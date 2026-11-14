# Question 2 — Resume safely

The four completed run numbers in the report are copied from the console output you supplied. AdaLoRA crashed because `zero_grad(set_to_none=True)` cleared `p.grad` before `update_and_allocate(step)`. The FIXED notebook changes this order.

1. In Colab with an available GPU, open `Assignment_Beta_Question_no_2_22P_9288_FIXED.ipynb`.
2. Execute setup, Drive mounting, dataset-generation, and script-writing cells. If you still have the original `results/q2_metrics.json` in Drive, use it (do NOT overwrite). If lost, use the **optional recovery cell**. It recreates only the four summary entries from the run log, marked as recovered, **not** the prediction files or checkpoints.
3. Run the corrected AdaLoRA launch cell (`--run adalora`) and ensure `RESULT adalora_r8` is printed and `q2_metrics.json` saved.
4. Run the rank sweep cell (`--run rank-sweep`) after all four methods are present. By current tie-breaking, LoRA is selected when LoRA/QLoRA/DoRA tie at full-match accuracy. It trains ranks 4, 16 and 32.
5. Save `results/q2_metrics.json`, every `results/q2_predictions_*.jsonl`, `figures/q2_rank_accuracy_measured.png`, the sealed data split, and the executed notebook. Original raw predictions for your lost runs cannot be recreated by editing JSON summaries.

**Do not claim Q2 is fully complete until AdaLoRA and rank sweep have actually run.**
