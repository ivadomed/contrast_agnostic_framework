# Evaluation Summary

| model_id | family | source_contrast | ckpt_exists | flair | t1w | t1gd | t2w | in_domain_dice | ood_mean_dice | ood_worst_dice |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_baseline_flair | baseline | flair | 1 | 0.8028 | 0.0206 | 0.1825 | 0.3745 | 0.8028 | 0.1925 | 0.0206 |
| baseline_baseline_t1gd | baseline | t1gd | 1 | 0.3140 | 0.6598 | 0.7428 | 0.0943 | 0.7428 | 0.3560 | 0.0943 |
| baseline_baseline_t1w | baseline | t1w | 1 | 0.2138 | 0.7308 | 0.6413 | 0.1091 | 0.7308 | 0.3214 | 0.1091 |
| baseline_baseline_t2w | baseline | t2w | 1 | 0.5908 | 0.0721 | 0.0923 | 0.8276 | 0.8276 | 0.2518 | 0.0721 |
| v2_generator_t1w | generator | t1w | 1 | 0.4465 | 0.7182 | 0.6894 | 0.5070 | 0.7182 | 0.5476 | 0.4465 |
| v2_generator_t2w | generator | t2w | 1 | 0.5796 | 0.1553 | 0.2361 | 0.7995 | 0.7995 | 0.3237 | 0.1553 |
