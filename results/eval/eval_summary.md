# Evaluation Summary

| model_id | family | source_contrast | ckpt_exists | flair | t1w | t1gd | t2w | in_domain_dice | ood_mean_dice | ood_worst_dice |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_flair | baseline | flair | 1 | 0.8267 | 0.0448 | 0.2453 | 0.3869 | 0.8267 | 0.2257 | 0.0448 |
| baseline_t1gd | baseline | t1gd | 1 | 0.3140 | 0.6598 | 0.7428 | 0.0943 | 0.7428 | 0.3560 | 0.0943 |
| baseline_t1w | baseline | t1w | 1 | 0.2138 | 0.7308 | 0.6413 | 0.1091 | 0.7308 | 0.3214 | 0.1091 |
| baseline_t2w | baseline | t2w | 1 | 0.5908 | 0.0721 | 0.0923 | 0.8276 | 0.8276 | 0.2518 | 0.0721 |
| generator_t1w | generator | t1w | 1 | 0.4493 | 0.7486 | 0.7047 | 0.5012 | 0.7486 | 0.5517 | 0.4493 |
| generator_t2w | generator | t2w | 1 | 0.5648 | 0.0956 | 0.2099 | 0.7636 | 0.7636 | 0.2901 | 0.0956 |
