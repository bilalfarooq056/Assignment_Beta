#!/usr/bin/env python3
"""Question 2: measured baseline, LoRA, QLoRA, DoRA, AdaLoRA + rank sweep.

Prerequisites: Python 3.10+, CUDA GPU, pytorch CUDA build, and:
 pip install 'transformers>=4.45,<5' 'peft>=0.15,<0.19' 'accelerate>=1,<2' \
             'bitsandbytes>=0.45' 'safetensors>=0.4' numpy matplotlib

Run from the package root:
 python code/prepare_q2_data.py
 python code/q2_finetune_experiments.py --run all
 python code/q2_finetune_experiments.py --run rank-sweep

The *same* 1800 train, 200 sealed test, model, modules, epochs, seed, AdamW,
linear LR schedule, sequence length and optimization batch size are used for
all methods. The one intentional variation is QLoRA's NF4 weight precision;
DoRA and AdaLoRA add their method-specific parameters. Paged optimizer is NOT
enabled, because changing optimizer only for QLoRA would confound comparison.

GPU runs can take substantial time. No measured results exist until this
script writes results/q2_metrics.json and per-example predictions files.
"""
import argparse, csv, gc, json, math, os, random, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data';RESULTS=ROOT/'results';RESULTS.mkdir(exist_ok=True)
MODEL_ID='Qwen/Qwen2.5-1.5B-Instruct'
SEED=2026
SYSTEM=('You are a customer-support function-call router. Respond with EXACTLY one '
        'JSON object, with keys "name" and "arguments"; no markdown or commentary. '
        'Valid names: check_order_status, initiate_refund, track_shipment, escalate_to_human. '
        'check_order_status arguments: {"order_id": string}; '
        'initiate_refund arguments: {"order_id": string, "reason": string}; '
        'track_shipment arguments: {"tracking_number": string}; '
        'escalate_to_human arguments: {"issue": string}. '
        'Never initiate a refund unless the customer explicitly requests one; '
        'backend refund execution always requires separate authorization.')

def load_split(filename):
    return [json.loads(line) for line in (DATA/filename).read_text(encoding='utf-8').splitlines() if line.strip()]

def seed_everything(torch):
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

def prompt_text(tokenizer,message):
    msgs=[{'role':'system','content':SYSTEM},{'role':'user','content':message}]
    return tokenizer.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)

def make_dataset(torch,tokenizer,rows,maxlen):
    class TrainDataset(torch.utils.data.Dataset):
        def __len__(self):return len(rows)
        def __getitem__(self,i):
            row=rows[i]
            prompt=tokenizer(prompt_text(tokenizer,row['message']),add_special_tokens=False).input_ids
            answer=json.dumps(row['answer'],separators=(',',':'),ensure_ascii=False)
            ans=tokenizer(answer,add_special_tokens=False).input_ids+[tokenizer.eos_token_id]
            if len(prompt)+len(ans)>maxlen:
                raise ValueError(f'example {row["id"]} length {len(prompt)+len(ans)} > maxlen {maxlen}; do not silently truncate answers')
            ids=prompt+ans
            return {'input_ids':ids,'labels':[-100]*len(prompt)+ans,'attention_mask':[1]*len(ids)}
    return TrainDataset()

def collate(torch,tokenizer,batch):
    longest=max(len(row['input_ids']) for row in batch)
    items={'input_ids':[], 'labels':[], 'attention_mask':[]}
    for row in batch:
        missing=longest-len(row['input_ids'])
        items['input_ids'].append(row['input_ids']+[tokenizer.pad_token_id]*missing)
        items['labels'].append(row['labels']+[-100]*missing)
        items['attention_mask'].append(row['attention_mask']+[0]*missing)
    return {key:torch.tensor(v,dtype=torch.long,device='cuda') for key,v in items.items()}

def train_method(torch,model,tokenizer,train,method,rank,epochs,batch_size,grad_accum,lr,maxlen):
    from peft import LoraConfig,AdaLoraConfig,get_peft_model
    from torch.utils.data import DataLoader
    data=make_dataset(torch,tokenizer,train,maxlen)
    loader=DataLoader(data,batch_size=batch_size,shuffle=True,generator=torch.Generator().manual_seed(SEED),
                      collate_fn=lambda batch:collate(torch,tokenizer,batch))
    steps_per_epoch=math.ceil(len(loader)/grad_accum);total_steps=steps_per_epoch*epochs
    modules=['q_proj','v_proj']
    if method=='adalora':
        cfg=AdaLoraConfig(task_type='CAUSAL_LM',init_r=max(12,int(rank*1.5)),target_r=rank,lora_alpha=16,
            tinit=max(1,int(.1*total_steps)),tfinal=max(1,int(.1*total_steps)),
            deltaT=10,total_step=total_steps,target_modules=modules)
    else:
        cfg=LoraConfig(task_type='CAUSAL_LM',r=rank,lora_alpha=16,
             lora_dropout=0,target_modules=modules,use_dora=(method=='dora'))
    model=get_peft_model(model,cfg)
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    total=sum(p.numel() for p in model.parameters())
    model.train();model.config.use_cache=False
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=0.0)
    warmup=max(1,round(.1*total_steps))
    def schedule(step):
        if step < warmup:return (step+1)/warmup
        return max(0.0,(total_steps-step)/max(1,total_steps-warmup))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=schedule)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter();step=0;running=0.
    for epoch in range(epochs):
        for ix,batch in enumerate(loader):
            loss=model(**batch).loss
            (loss/grad_accum).backward()
            running+=float(loss.item())
            if (ix+1)%grad_accum==0 or ix==len(loader)-1:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
                optimizer.step();scheduler.step()
                step+=1
                if method=='adalora':
                    # IMPORTANT: update rank importance while parameter gradients still exist.
                    # Calling zero_grad(set_to_none=True) first makes p.grad None.
                    model.base_model.update_and_allocate(step)
                optimizer.zero_grad(set_to_none=True)
                if step % 40==0:print(f'  {method} rank={rank} epoch={epoch+1}/{epochs} step={step}/{total_steps} loss={loss.detach().item():.4f}',flush=True)
    torch.cuda.synchronize()
    seconds=time.perf_counter()-start
    peak_gb=torch.cuda.max_memory_allocated()/2**30
    model.eval();model.config.use_cache=True
    return model,{'trainable_params':trainable,'total_params':total,'trainable_pct':100*trainable/total,
       'peak_training_gpu_gib':round(peak_gb,3),'training_minutes':round(seconds/60,3),
       'training_steps':step,'epochs':epochs,'rank':rank,'method':method,
       'final_train_batch_loss':round(float(loss.item()),5)}

def evaluate(torch,model,tokenizer,test,tag,maxlen=110):
    rows=[];valid=0;tool_correct=0;full_correct=0;inference_seconds=[]
    model.eval();model.config.use_cache=True
    for i,row in enumerate(test):
        p=tokenizer(prompt_text(tokenizer,row['message']),return_tensors='pt',add_special_tokens=False).to('cuda')
        torch.cuda.synchronize(); t_infer=time.perf_counter()
        with torch.no_grad():
            generation=model.generate(**p,max_new_tokens=maxlen,do_sample=False,num_beams=1,
                     pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize(); inference_seconds.append(time.perf_counter()-t_infer)
        raw=tokenizer.decode(generation[0,p['input_ids'].shape[1]:],skip_special_tokens=True).strip()
        obj=None;reason=''
        try:
            obj=json.loads(raw)
            if not isinstance(obj,dict) or not isinstance(obj.get('arguments'),dict) or not isinstance(obj.get('name'),str):
                raise ValueError('not expected JSON object shape')
            valid+=1
        except (json.JSONDecodeError,ValueError) as exc:reason='broken_json_or_schema'
        is_tool=(obj is not None and obj.get('name')==row['answer']['name'])
        is_full=(is_tool and obj.get('arguments')==row['answer']['arguments'] and set(obj)=={'name','arguments'})
        if is_tool:tool_correct+=1
        if is_full:full_correct+=1
        if not reason and not is_tool:reason='wrong_tool'
        if not reason and is_tool and not is_full:reason='wrong_or_extra_arguments'
        rows.append({'id':row['id'],'message':row['message'],'expected':row['answer'],'raw_output':raw,
                     'parsed':obj,'valid':obj is not None,'tool_correct':bool(is_tool),
                     'full_match':bool(is_full),'error_category':reason})
        if (i+1)%25==0:print('  evaluation',tag,i+1,'/',len(test),flush=True)
    path=RESULTS/f'q2_predictions_{tag}.jsonl'
    with path.open('w',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    return {'json_valid_n':valid,'json_valid_rate':valid/len(test),
            'tool_correct_n':tool_correct,'tool_accuracy':tool_correct/len(test),
            'full_match_n':full_correct,'full_match_rate':full_correct/len(test),
            'evaluation_n':len(test),'mean_inference_seconds':sum(inference_seconds)/len(inference_seconds),'predictions_file':str(path.relative_to(ROOT))}

def run(method,rank,args,train,test,metrics):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training
    seed_everything(torch);torch.cuda.empty_cache()
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,trust_remote_code=False)
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    if method=='qlora':
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quant_cfg=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                 bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype)
        model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={'':'cuda:0'},
                 quantization_config=quant_cfg,trust_remote_code=False)
        model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True)
    else:
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={'':'cuda:0'},
                   torch_dtype=dtype,trust_remote_code=False)
        if method!='base':
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
    if method=='base':
        info={'method':'base','rank':0,'trainable_params':0,'total_params':sum(p.numel() for p in model.parameters()),
              'trainable_pct':0,'peak_training_gpu_gib':None,'training_minutes':0,'training_steps':0,'epochs':0}
    else:
        model,info=train_method(torch,model,tokenizer,train,method,rank,args.epochs,args.batch_size,
              args.grad_accum,args.learning_rate,args.max_length)
    tag='base' if method=='base' else f'{method}_r{rank}'
    info.update(evaluate(torch,model,tokenizer,test,tag))
    metrics[tag]=info
    (RESULTS/'q2_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    del model;gc.collect();torch.cuda.empty_cache()
    print('RESULT',tag,json.dumps(info,indent=2),flush=True)

def plot_metrics(metrics):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        ranks=[(int(x.split('_r')[-1]),v['tool_accuracy']) for x,v in metrics.items() if '_r' in x and x.startswith(metrics.get('rank_sweep_method','!')+'_r')]
        if len(ranks)>=4:
            ranks.sort();x,y=zip(*ranks)
            plt.figure(figsize=(7,4.5));plt.plot(x,y,'o-',linewidth=2)
            plt.xlabel('Adapter rank');plt.ylabel('Held-out tool-call accuracy')
            plt.title('Q2 rank-sensitivity (measured)');plt.xticks(x);plt.ylim(0,1);plt.grid(alpha=.25)
            plt.tight_layout();plt.savefig(ROOT/'figures'/'q2_rank_accuracy_measured.png',dpi=180);plt.close()
    except ImportError:print('matplotlib not installed, rank data still in q2_metrics.json')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',choices=['all','rank-sweep','base','lora','qlora','dora','adalora'],default='all')
    parser.add_argument('--epochs',type=int,default=2)
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--grad-accum',type=int,default=8)
    parser.add_argument('--learning-rate',type=float,default=2e-4)
    parser.add_argument('--max-length',type=int,default=256)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    train=load_split('q2_train.jsonl');test=load_split('q2_test_sealed.jsonl')
    assert len(train)==1800 and len(test)==200
    assert not {x['id'] for x in train}&{x['id'] for x in test}
    if args.check_only:
        print('Dataset validated (1800 train, 200 sealed test); imports and GPU not tested');return
    import torch
    if not torch.cuda.is_available():raise SystemExit('CUDA GPU required for the specified NF4 comparison; no fake results will be generated')
    metrics_path=RESULTS/'q2_metrics.json'
    metrics=json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    if args.run=='all':
        for method in ['base','lora','qlora','dora','adalora']:
            tag='base' if method=='base' else f'{method}_r8'
            if tag in metrics:print('Already complete, skipping',tag);continue
            run(method,8,args,train,test,metrics)
    elif args.run=='rank-sweep':
        candidates=[(method,metrics[f'{method}_r8']) for method in ['lora','qlora','dora','adalora'] if f'{method}_r8' in metrics]
        if len(candidates)!=4:raise SystemExit('Run --run all first: results from all four methods are required')
        # Assignment explicitly asks to choose best after Part B; this uses the
        # same 200 test items for selection and rank plot, causing selection bias.
        best=max(candidates,key=lambda t:(t[1]['full_match_rate'],t[1]['tool_accuracy']))[0]
        metrics['rank_sweep_method']=best
        for r in [4,16,32]:
            tag=f'{best}_r{r}'
            if tag not in metrics:run(best,r,args,train,test,metrics)
        (RESULTS/'q2_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
        plot_metrics(metrics)
    else:
        method=args.run;rank=8;run(method,rank,args,train,test,metrics)
    print('Measured files:',RESULTS)

if __name__=='__main__':main()
