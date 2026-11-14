# Assignment Beta - Agentic AI

**Student:** Bilal Farooq  
**Roll number:** 22P-9288  
**Section:** BAI-7A  
**Course:** Agentic AI (AI4015)  
**Semester:** Fall 2026

## Overview

Assignment Beta covers five connected topics: selecting an LLM for an agent, improving tool calling with parameter-efficient fine-tuning, knowledge distillation, evaluating model predictions, and reviewing retrieval-augmented generation (RAG). The work combines **research, reproducible calculations, Jupyter/Colab notebooks, and measured experiments**.

The main approach was to break each problem into **inputs → formulas or experiment → evidence → interpretation**. This kept model-price assumptions separate from measurements and made each result easier to check.

## Questions at a glance

| Question | Task | Approach | Result / status |
|---|---|---|---|
| **Q1 - SaafHawa** | Choose between a hosted LLM API and self-hosting for a Karachi weather/air-quality agent | Compare 6 closed + 6 open-weight models; calculate tokens, cost, peak load and GPU memory; time four tools on two dates | Workload calculated; **160 successful tool measurements**; original five-call latency estimate exceeds the 9-second goal. Hosted small-model API is the provisional cost choice. |
| **Q2 - Tool-calling fine-tuning** | Improve structured function calls using LoRA, QLoRA, DoRA and AdaLoRA | Qwen2.5-1.5B-Instruct, 1,800 synthetic training examples + 200 held-out examples; baseline and adapter comparisons | Baseline, LoRA, QLoRA and DoRA measured; **AdaLoRA and rank sweep remain pending**. |
| **Q3 - Knowledge distillation** | Compute teacher/student probabilities, distillation loss, gradient and an update | Step-by-step softmax, KL, cross-entropy and gradient descent; Python verification | Loss decreased from **0.463717** to **0.349256** after one illustrative update. |
| **Q4 - Model evaluation** | Compare sentiment classification performance and calculate text metrics | Confusion matrices, Wilson intervals, per-class and aggregate F1, stability, BLEU, ROUGE, METEOR and perplexity | Accuracy improved **76.67% → 84.67%**; calculations reproduced in a notebook. |
| **Q5 - RAG paper review** | Summarize a comparative RAG architecture study | Compare zero-shot, sparse, dense, hybrid and fusion retrieval, including quality/latency trade-offs | Two-page research summary with findings, limitations and SaafHawa implications. |

## Q1 - SaafHawa: from an agent idea to an engineering decision

The question asks whether a WhatsApp assistant should call a paid model API or host an open-weight model. Using roll number **22P-9288**, the assignment parameters give:

- **3,700** normal daily citizen queries and **170,200** queries in a 30-day month with four heatwave days.
- **5** initial model calls per query, a **5%** invalid-tool-call probability, and expected retry-adjusted totals of **18,640 input** and **2,070 output tokens** per query.
- A **$2,080/month** inference budget and **9-second** response target.

The notebook compares twelve candidate models, then estimates API spending, caching, rate-limit pressure, Qwen2.5-7B model/KV memory, and monthly GPU rental. It also logs OpenAQ, Open-Meteo, a heat-index calculation and a small advisory lookup.

**Measured evidence:** 40 successful measurements for each tool across **11 and 14 November 2026** (160 total). The **pooled median of all 160 tool timings was 0.0137149325 seconds**. The optimistic original five-call Qwen latency estimate was approximately **15.27 seconds**, exceeding the 9-second target. A redesigned workflow that fetches independent information concurrently and generates a single final response appears promising, but has **not** been validated end to end.

**Important limitations:** The logged OpenAQ values were repeated **PM2.5-only** data with a **March 2025** observation timestamp; they establish API response time, **not** fresh PM2.5/PM10 coverage. Some endpoint-specific price, throughput, benchmark and rate-limit fields remain unverified. The GPU break-even graph is a **memory-only scenario**, not a demonstrated inference-throughput capacity or SLA.

**Keep with the submission:** the executed Q1 notebook, both original timing CSVs, the combined timing CSV, calculation plots/tables, and the Qwen configuration screenshot required by the question.

## Q2 - Tool calling: adapt only a small fraction of the model

The e-commerce assistant must choose among `check_order_status`, `initiate_refund`, `track_shipment` and `escalate_to_human`, returning valid JSON and correct arguments. Instead of updating every model weight, the experiment trains small adapters:

- **LoRA:** low-rank updates to frozen weights.
- **QLoRA:** LoRA adapters with a 4-bit NF4-quantized base model.
- **DoRA:** separate learning of weight magnitude and direction.
- **AdaLoRA:** adaptation rank reallocated according to estimated importance.

For a **4096 × 4096** matrix and LoRA rank **16**, the adapter adds **131,072 parameters**, or **0.78125%** of the original matrix's parameter count.

### Measured 200-example evaluation

| Model | Valid JSON | Correct tool | Full function + argument match | Peak training GPU | Training time |
|---|---:|---:|---:|---:|---:|
| Unmodified base | 200/200 | 186/200 (93%) | 120/200 (60%) | N/A | 0 min |
| LoRA, rank 8 | 200/200 | 200/200 (100%) | 200/200 (100%) | 3.588 GiB | 46.46 min |
| QLoRA, rank 8 | 200/200 | 200/200 (100%) | 200/200 (100%) | 2.253 GiB | 50.60 min |
| DoRA, rank 8 | 200/200 | 200/200 (100%) | 200/200 (100%) | 3.590 GiB | 48.05 min |
| AdaLoRA, 12 → 8 | - | - | - | - | **Not completed** |

The three completed fine-tuned models tied on this **synthetic, template-based** test set. LoRA was fastest to train among those three; QLoRA saved about **37%** of peak GPU allocation relative to LoRA. Neither 100% test accuracy nor the low final batch loss establishes production reliability.

**What interrupted the run:** AdaLoRA's importance update attempted to use gradients after they had been cleared, producing a `Parameter * NoneType` error. A corrected notebook/script was prepared, but **its AdaLoRA completion and the additional rank-4/16/32 sweep have not been verified**. The five-error-case discussion also cannot be fabricated when the fine-tuned models have no recorded errors on this test set; a separate challenging evaluation set is needed.

**Dataset note:** The provided 2,000-example set is **synthetic**, not the public Glaive dataset suggested as an example in the assignment. This should be disclosed in the report.

## Q3 - Distillation: a worked gradient update

Teacher logits `[5, 3, 2, 0]`, student logits `[3, 2.5, 1, 0.5]`, temperature `T = 2`, mixing weight `α = 0.5`, and learning rate `η = 0.5` were used to calculate softmax probabilities, the softened KL term, hard-label cross-entropy, the total loss and its gradient.

- Initial loss: **0.463717126**
- Updated logits: **[3.206290350, 2.366768329, 0.972535856, 0.454405465]**
- New loss: **0.349256040**

The reduction illustrates how the student prediction changes after one gradient-descent step. This is a **numerical exercise**, not a trained language model.

## Q4 - Evaluation: improve the prompt, then measure it

Starting from instructor-provided 300-review confusion matrices, the work calculates classification and text-generation evaluation metrics.

- Zero-shot accuracy: **76.67%**; improved few-shot accuracy: **84.67%** (**+8 percentage points**).
- Neutral recall: **62.5% → 80.0%**; a clearer neutral-class definition helped the weakest class.
- Other calculations: Wilson 95% confidence intervals, precision/recall/F1 for each class, macro/weighted/micro F1, prompt-order stability, BLEU-4, ROUGE-1/2/L, exact-token METEOR and perplexity.

The work was simplified by showing **formula → substituted values → final result** for every calculation, then checking numerical outputs in Python. The evaluation is based on **provided matrices and token sequences**, not a newly run sentiment model.

## Q5 - RAG: grounding versus speed

The literature review compares an ungrounded baseline with **BM25 sparse retrieval**, **dense retrieval**, **hybrid retrieval/reranking**, and **fusion** approaches. The supplied study's reported faithfulness scores favor hybrid/fusion retrieval, while additional retrieval work raises latency. The assignment summary recommends considering **hybrid retrieval for responsive SaafHawa guidance**, with freshness checks and traceable source evidence. These are **paper-derived interpretations**, not new experiments performed for Assignment Beta.

## How the work became simpler

1. **Parameterize once:** roll-number-derived values feed the Q1 workload and cost calculations.
2. **Separate data from assumptions:** API timings and GPU measurements are labeled as observations; model prices, serving rates and GPU break-even are labeled as sourced inputs or estimates.
3. **Reuse the same test conditions:** Q2 holds dataset, seed, epochs, target modules and evaluation criteria constant across adaptation methods.
4. **Make calculations auditable:** Q3 and Q4 show formulas, substitutions and final answers; notebooks provide reproducibility.
5. **Preserve evidence:** notebooks, raw CSVs, prediction logs when available, and dated links support the written reports.

## Repository contents and submission guidance

The intended submission consists of **five separate question reports**, supported by notebooks and data where required. File names may differ slightly depending on the downloaded package.

```text
Assignment_Beta/
├── README.md
├── Q1/                         # SaafHawa report, executed notebook, timing CSVs, charts
├── Q2/                         # PEFT report, notebook, training script, dataset, available logs
├── Q3/                         # Worked distillation solution and verification notebook
├── Q4/                         # Worked evaluation solution and verification notebook
└── Q5/                         # Two-page RAG paper summary
```

**Submission distinction:** Q1 requires a report, calculation notebook/spreadsheet and **raw tool-timing log**. Q2 requires the **actual fine-tuning and evaluation evidence** and a named notebook. Q3 is mathematical working; Q4 is calculation-based but also specifies a named notebook; Q5 is a research-paper summary. Thus, it would be inaccurate to describe the *last two* questions as calculations: **Q3 and Q4 are calculations, whereas Q5 is a literature review**.

### What remains open

- **Q1:** Verify live PM2.5 **and** PM10 freshness, add the required Qwen `config.json` screenshot, and retain clear caveats for unverified model-specific figures and untested integrated latency.
- **Q2:** Complete AdaLoRA and the **rank 4/16/32** experiments with a suitable GPU; retain real results and discuss limitations of the synthetic test set.
- **Across the assignment:** Submit the executed notebooks and raw logs alongside the formatted documents, and include the required declaration describing AI assistance and its purpose.

> **Integrity note:** “Prepared code” and “completed experiment” are not interchangeable. This README reports measured results only where execution evidence was shared; unfinished experiments are explicitly marked.
