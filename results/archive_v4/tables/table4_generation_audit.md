**Table 4. First-token log-odds vs. 16-token greedy generation audit (held-out test)**

| condition | dataset_partition | n_prompts | mean_log_odds | log_odds_ci_95_low | log_odds_ci_95_high | mean_p_unknown | gen_abstention_rate_strict | gen_abstention_rate_regex | gen_rate_ci_95_low | gen_rate_ci_95_high | discrepancy_flagged |
|---|---|---|---|---|---|---|---|---|---|---|---|
| test_unans_baseline | unans | 96 | 7.0197 | 6.1783 | 7.6663 | 0.9672 | 0.9583 | 0.9583 | 0.8750 | 1.0000 | False |
| test_unans_ablate_learned | unans | 96 | 0.2160 | -1.1360 | 1.6706 | 0.5168 | 0.5208 | 0.5208 | 0.3542 | 0.6979 | False |
| test_unans_steer_neg0p5_learned | unans | 96 | 2.0639 | 0.2184 | 3.8853 | 0.6296 | 0.6250 | 0.6250 | 0.4479 | 0.7917 | False |
| test_unans_steer_neg1_learned | unans | 96 | -0.9823 | -2.3800 | 0.6312 | 0.3664 | 0.4062 | 0.4062 | 0.2083 | 0.6042 | False |
| test_unans_steer_neg1p5_learned | unans | 96 | -1.0922 | -1.7028 | -0.4830 | 0.3203 | 0.4062 | 0.4062 | 0.2083 | 0.5940 | False |
| test_unans_steer_neg2_learned | unans | 96 | -2.1056 | -2.4081 | -1.8074 | 0.1348 | 0.2812 | 0.2812 | 0.1456 | 0.4375 | False |
| test_unans_ablate_concept | unans | 96 | 6.9796 | 6.4176 | 7.4764 | 0.9907 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | False |
| test_unans_steer_neg1_concept | unans | 96 | 5.7313 | 5.3910 | 6.0757 | 0.9937 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | False |
| test_ans_baseline | ans | 96 | -11.4620 | -12.3178 | -10.5573 | 0.0003941 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| test_ans_steer_pos0p5_learned | ans | 96 | 5.5074 | 4.4990 | 6.4396 | 0.9306 | 0.9271 | 0.9271 | 0.8229 | 1.0000 | False |
| test_ans_steer_pos1_learned | ans | 96 | 8.8468 | 8.7713 | 8.9235 | 0.9999 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | False |
| test_ans_steer_pos1p5_learned | ans | 96 | 3.1331 | 2.7530 | 3.5345 | 0.9356 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | False |
| test_ans_steer_pos2_learned | ans | 96 | -7.1671 | -7.5844 | -6.7311 | 0.0013 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| test_ans_steer_pos1_concept | ans | 96 | -8.8042 | -9.7054 | -7.8905 | 0.0031 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| test_ans_ablate_learned | ans | 96 | -11.8831 | -12.6672 | -11.1293 | 0.0001838 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| b_ctrl_baseline | bctrl | 48 | -9.1725 | -12.0071 | -5.5493 | 0.0933 | 0.0833 | 0.0833 | 0.0000 | 0.2500 | False |
