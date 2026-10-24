**Table 3. Popularity-matched PopQA wrong (y=1) vs. correct (y=0), N = 147 per group**

| score | description | auroc | ci_95_low | ci_95_high | delta_vs_proj | delta_ci_95_low | delta_ci_95_high | p_value | p_value_test |
|---|---|---|---|---|---|---|---|---|---|
| proj_learned | x_{i*}^{(l*)} . r_hat (H4) | 0.6930 | 0.6364 | 0.7464 |  |  |  | 5.242e-09 | one-sided Mann-Whitney U |
| neg_maxp | -max first-token probability (H5 baseline) | 0.7736 | 0.7195 | 0.8254 | -0.0805 | -0.1426 | -0.0207 | 0.9960 | paired bootstrap, one-sided |
| entropy | first-token predictive entropy | 0.7774 | 0.7247 | 0.8291 | -0.0843 | -0.1418 | -0.0336 | 0.9990 | paired bootstrap, one-sided |
| abstention_log_odds | first-token abstention log-odds S(p) | 0.6858 | 0.6270 | 0.7421 | 0.0073 | -0.0399 | 0.0551 | 0.3866 | paired bootstrap, one-sided |
| random_99_mean | mean AUROC of 99 random unit directions | 0.4933 | 0.3793 | 0.6634 |  |  |  | 0.0100 | Monte Carlo rank of learned AUROC |
| shuffled_19_mean | mean AUROC of 19 label-shuffled directions | 0.4769 | 0.3072 | 0.7025 |  |  |  | 0.2000 | permutation rank of learned AUROC |
| r_concept | orthogonalized concept direction | 0.5682 |  |  |  |  |  |  |  |
