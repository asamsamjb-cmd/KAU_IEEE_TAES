# Step 45 formal encoder training

TRAIN/VALIDATION ONLY. Test evaluation is pending.

Five encoder-training seeds per condition; each selected encoder has a newly trained L10 TST.
TST/window-head/shuffle seeds are fixed. Pilot runs are excluded. No seed selection or ensemble.

| Encoder | Seeds | All-window val mean | Common-window val mean | L10 val mean | L10 seed SD |
|---|---:|---:|---:|---:|---:|
| CategoricalEmbedding_MLP | 5/5 | 0.404346 | 0.434067 | 0.838632 | 0.000913 |
| FieldToken_Encoder | 5/5 | 0.404535 | 0.434656 | 0.835361 | 0.001798 |
| DistilBERT_RandomInit_Train | 5/5 | 0.407104 | 0.437900 | 0.834342 | 0.003380 |
| DistilBERT_Pretrained_Finetune | 5/5 | 0.408507 | 0.438691 | 0.836315 | 0.000568 |

Prespecified finite budgets; no full-convergence, equal-compute, or equivalence claim.
Common-window and L10 scores share identical endpoints; the all-window metric selects the encoder checkpoint.
L10 encoder rankings are conditional on the fixed TST control; window/TST heads also differ in capacity and training.
Recorded timings are training/extraction times, not deployment inference latency.
Do not tune settings or pick seeds using these formal outcomes.
Retain the complete folder: review ZIP excludes checkpoints, embeddings and raw logits.
