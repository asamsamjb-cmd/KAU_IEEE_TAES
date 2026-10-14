# Step 15: paired episode bootstrap

Delta = comparator minus TST. Values are Macro-F1 on the 0–1 scale.

| Comparator | Delta | 95% percentile CI | Bonferroni 98.75% CI |
|---|---:|---|---|
| Mean_MLP | +0.00033998 | [-0.00078177, +0.00142570] | [-0.00110309, +0.00172489] |
| Max_MLP | -0.00275052 | [-0.00444956, -0.00120138] | [-0.00492598, -0.00082808] |
| MeanMax_MLP | -0.00073344 | [-0.00213822, +0.00053058] | [-0.00253225, +0.00085985] |
| LSTM | +0.00023060 | [-0.00056044, +0.00098265] | [-0.00079259, +0.00118363] |

Intervals condition on the ten observed trained heads and fixed encoder/data.
Head seeds are averaged, not resampled. Seed SD is reported separately.
An interval containing zero does not establish equivalence.
Statistical direction does not by itself establish practical importance.
The four adjusted intervals target nominal 95% family coverage under the bootstrap approximation.
CSV alignment and hashes do not independently certify checkpoint provenance.
No operational satellite deployment or generator-shift claim follows from this analysis.
