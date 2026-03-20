# Evaluation Summary

| model_id | family | source_contrast | ckpt_exists | flair | t1w | t1gd | t2w | in_domain_dice | ood_mean_dice | ood_worst_dice |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_flair | baseline | flair | 1 | 0.8314 | 0.0334 | 0.2174 | 0.4104 | 0.8314 | 0.2204 | 0.0334 |
| baseline_t1gd | baseline | t1gd | 1 | 0.3513 | 0.6107 | 0.7326 | 0.0795 | 0.7326 | 0.3472 | 0.0795 |
| baseline_t1w | baseline | t1w | 1 | 0.1889 | 0.6987 | 0.5806 | 0.0959 | 0.6987 | 0.2885 | 0.0959 |
| baseline_t2w | baseline | t2w | 1 | 0.6066 | 0.0730 | 0.0983 | 0.8128 | 0.8128 | 0.2593 | 0.0730 |
| v4_fullyartificial_t1w | fullyartificial | t1w | 1 | 0.6551 | 0.6403 | 0.6414 | 0.6543 | 0.6403 | 0.6503 | 0.6414 |
| v4_fullyartificial_t2w | fullyartificial | t2w | 1 | 0.6975 | 0.3345 | 0.4775 | 0.6775 | 0.6775 | 0.5032 | 0.3345 |
