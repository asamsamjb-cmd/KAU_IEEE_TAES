# Step 17: fixed Rule comparisons

Delta = comparator minus reference. F1 and intervals use the 0-1 scale.

| Comparator | Reference | Delta | 95% CI | Bonferroni 97.5% CI |
|---|---|---:|---|---|
| Rule_L10 | Rule_L1 | +0.31162281 | [+0.29612812, +0.32633799] | [+0.29406866, +0.32828668] |
| TST_L10 | Rule_L10 | +0.19081626 | [+0.17116111, +0.21072277] | [+0.16838035, +0.21355549] |

The contrast definitions are carried forward from the earlier revision analysis; no prospective preregistration is claimed.
Rules/thresholds and the trained encoder/heads remain fixed; no selection is repeated in bootstrap.
Resampling is shared whole episodes within class. Head seeds are fixed and averaged after F1.
Rule L10 includes mean evidence, recurrence support and separate validation calibration.
It is not a pure mean-pooling ablation. One rule baseline cannot represent every possible rule.
Neither contrast establishes that DistilBERT or TST is essential.
The Bonferroni family contains only these two contrasts, not all revision analyses.
A CI containing zero is not evidence of equivalence; no p-values are computed.
Intervals require approximately independent/exchangeable episodes within each class.
No encoder-training, validation-selection or generator uncertainty is included.
Prediction hashes/alignment do not independently certify checkpoint provenance or operational validity.

Do not retune Rule thresholds or select a new aggregator because of these test scores.
