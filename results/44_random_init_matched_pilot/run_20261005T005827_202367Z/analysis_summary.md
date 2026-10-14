# Step 44 matched random-init pilot

TRAIN/VALIDATION ONLY; one diagnostic seed; review before formal settings are declared.

Random-init matches the selected main Step 43 pretrained training rules and maximum budget.
Step 43 models are not retrained. Actual early stopping and optimizer updates can differ.

| Run | Encoder | All-window val F1 | Common-window val F1 | L10 val F1 | L10 minus common-window |
|---|---|---:|---:|---:|---:|
| step43_reference | DistilBERT_RandomInit_Train | 0.282595 | 0.305083 | 0.699031 | +0.393948 |
| step43_reference | DistilBERT_Pretrained_Finetune | 0.404001 | 0.434502 | 0.830566 | +0.396064 |
| step44_matched | DistilBERT_RandomInit_Train | 0.409869 | 0.439896 | 0.836204 | +0.396308 |

Common-window and L10 scores use the exact same validation endpoints.
All-window F1 uses all validation windows and selects the encoder checkpoint.
one diagnostic seed; exclude from formal seed statistics
same training rules and maximum budget, not necessarily equal realized optimizer steps or epochs
compared with the Step 43 random run BOTH learning rate and schedule horizon change
validation selects encoder/TST checkpoints and informs later settings; this is not held-out evidence
no full-convergence, equivalence, test-generalization or pure learning-rate causal claim
window and TST heads differ in capacity/training; common endpoints alone do not isolate time aggregation
all supplied shared-nominal episode labels are retained, including quiet attack windows; no evidence-based filtering or relabeling
overall validation F1 alone does not establish attack-evidence use or exclude generator signatures; interpret with the benchmark/evidence audits
Keep full 43/44 run folders; review ZIPs exclude weights, embeddings and logits.
