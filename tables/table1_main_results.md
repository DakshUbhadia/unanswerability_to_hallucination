**Table 1. Pre-registered tests H1-H5 (Holm-Bonferroni, alpha_FWER = 0.05) and secondary controls**

| test_id | hypothesis_or_control | metric | point_estimate | ci_95_low | ci_95_high | raw_p_value | holm_threshold | passed_holm_bonferroni | confirmed | mc_rank_out_of_100 | mc_p_value | mc_z_score | shuffled_p_value | shuffled_z_score | r_concept_effect_ratio | wikitext2_kl_nats | mmlu_accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H1 | Causal necessity: all-layer ablation on D_test^unans | mean log-odds shift (hypothesised direction > 0) | 6.8037 | 5.7226 | 7.8587 | 5.96e-08 | 0.0125 | True | True | 1 | 0.0100 | 59.6460 | 0.0500 | 1.8511 | 0.0059 | 0.0192 | 0.5700 |
| H2 | Causal sufficiency: +1.0 r at l* on D_test^ans | mean log-odds shift (hypothesised direction > 0) | 20.3088 | 19.4097 | 21.1753 | 5.96e-08 | 0.0167 | True | True | 1 | 0.0100 | 9.1601 | 0.0500 | 12.8168 | 0.1309 | 0.1060 | 0.3050 |
| H3 | Private-detail de-confounding (Family B unans - B-ctrl projection) | mean template projection difference | 4.1143 | 3.4125 | 4.7115 | 0.0002441 | 0.0250 | True | True |  |  |  |  |  |  |  |  |
| H3_bctrl_rate | B-ctrl generation abstention rate (criterion <= 0.15) | regex abstention rate | 0.0833 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| S1_H1b | [secondary] Single-layer subtraction -1.0 r on D_test^unans | mean log-odds shift (hypothesised direction > 0) | 8.0020 | 6.5968 | 9.4117 | 5.96e-08 |  |  |  | 1 | 0.0100 | 7.7504 | 0.0500 | 14.1763 | 0.1610 | 0.1025 | 0.4150 |
| S2_concept | [secondary] r_concept pre-orthogonalization cosine | cos(theta_pre) | -0.2362 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| H4 | Closed-book wrong vs. correct (popularity-matched PopQA) | AUROC_proj (criterion >= 0.60) | 0.6930 | 0.6364 | 0.7464 | 5.242e-09 | 0.0100 | True | True | 1 | 0.0100 | 2.6580 | 0.2000 | 1.5140 |  |  |  |
| H5 | Superiority over max first-token probability | AUROC_proj - AUROC_maxp | -0.0805 | -0.1426 | -0.0207 | 0.9960 | 0.0500 | False | False |  |  |  |  |  |  |  |  |
| S3_entropy | [secondary] AUROC_proj - AUROC_entropy | delta AUROC | -0.0843 | -0.1418 | -0.0336 | 0.9990 |  |  |  |  |  |  |  |  |  |  |  |
