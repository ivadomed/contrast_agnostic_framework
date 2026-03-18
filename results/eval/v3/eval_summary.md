# Evaluation Summary

| model_id | family | source_contrast | ckpt_exists | flair | t1w | t1gd | t2w | in_domain_dice | ood_mean_dice | ood_worst_dice |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_baseline_flair | baseline | flair | 1 | 0.8028 | 0.0206 | 0.1825 | 0.3745 | 0.8028 | 0.1925 | 0.0206 |
| baseline_baseline_t1gd | baseline | t1gd | 1 | 0.3140 | 0.6598 | 0.7428 | 0.0943 | 0.7428 | 0.3560 | 0.0943 |
| baseline_baseline_t1w | baseline | t1w | 1 | 0.2138 | 0.7308 | 0.6413 | 0.1091 | 0.7308 | 0.3214 | 0.1091 |
| baseline_baseline_t2w | baseline | t2w | 1 | 0.5908 | 0.0721 | 0.0923 | 0.8276 | 0.8276 | 0.2518 | 0.0721 |
| v3_fullyartificial_t1w | fullyartificial | t1w | 1 | 0.6495 | 0.6242 | 0.6445 | 0.6882 | 0.6242 | 0.6608 | 0.6445 |
| v3_fullyartificial_t2w | fullyartificial | t2w | 1 | 0.6013 | 0.3323 | 0.4469 | 0.6925 | 0.6925 | 0.4601 | 0.3323 |
| v3_generator_t1w | generator | t1w | 1 | 0.5222 | 0.6905 | 0.6635 | 0.5823 | 0.6905 | 0.5893 | 0.5222 |
| v3_generator_t2w | generator | t2w | 1 | 0.6163 | 0.2029 | 0.3255 | 0.7815 | 0.7815 | 0.3816 | 0.2029 |
