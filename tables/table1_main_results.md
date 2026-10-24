**Table 1. Pre-registered tests H1-H5 (Holm-Bonferroni, alpha_FWER = 0.05) and secondary controls**

| test_id | hypothesis_or_control | metric | point_estimate | ci_95_low | ci_95_high | raw_p_value | holm_threshold | passed_holm_bonferroni | confirmed | mc_rank_out_of_100 | mc_p_value | mc_z_score | shuffled_p_value | shuffled_z_score | r_concept_effect_ratio | wikitext2_kl_nats | mmlu_accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H1 | Causal necessity: all-layer ablation on D_test^unans | mean log-odds shift (hypothesised direction > 0) | 6.8037 | 5.7226 | 7.8587 | 5.96e-08 |  |  |  | 1 | 0.0100 | 59.6460 | 0.0500 | 1.8511 | 0.0059 | 0.0192 | 0.5700 |
| H2 | Causal sufficiency: +1.0 r at l* on D_test^ans | mean log-odds shift (hypothesised direction > 0) | 20.3088 | 19.4097 | 21.1753 | 5.96e-08 |  |  |  | 1 | 0.0100 | 9.1601 | 0.0500 | 12.8168 | 0.1309 | 0.1060 | 0.3050 |
| H3 | Private-detail de-confounding (Family B unans - B-ctrl projection) | mean template projection difference | 4.1143 | 3.4125 | 4.7115 | 0.0002441 |  |  |  |  |  |  |  |  |  |  |  |
| H3_bctrl_rate | B-ctrl generation abstention rate (criterion <= 0.15) | regex abstention rate | 0.0833 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| S1_H1b | [secondary] Single-layer subtraction -1.0 r on D_test^unans | mean log-odds shift (hypothesised direction > 0) | 8.0020 | 6.5968 | 9.4117 | 5.96e-08 |  |  |  | 1 | 0.0100 | 7.7504 | 0.0500 | 14.1763 | 0.1610 | 0.1025 | 0.4150 |
| S2_concept | [secondary] r_concept pre-orthogonalization cosine | cos(theta_pre) | -0.2362 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
