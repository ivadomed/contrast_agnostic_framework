# Evaluation Summary

| model_id | family | source_contrast | ckpt_exists | flair | t1w | t1gd | t2w | in_domain_dice | ood_mean_dice | ood_worst_dice |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_baseline_flair | baseline | flair | 1 | 0.8028 | 0.0206 | 0.1825 | 0.3746 | 0.8028 | 0.1925 | 0.0206 |
| baseline_baseline_t1gd | baseline | t1gd | 1 | 0.3136 | 0.6599 | 0.7427 | 0.0941 | 0.7427 | 0.3559 | 0.0941 |
| baseline_baseline_t1w | baseline | t1w | 1 | 0.2139 | 0.7307 | 0.6413 | 0.1091 | 0.7307 | 0.3214 | 0.1091 |
| baseline_baseline_t2w | baseline | t2w | 1 | 0.5908 | 0.0722 | 0.0923 | 0.8276 | 0.8276 | 0.2518 | 0.0722 |
| v3_fullyartificial_t1w | fullyartificial | t1w | 1 | 0.6496 | 0.6242 | 0.6445 | 0.6882 | 0.6242 | 0.6608 | 0.6445 |
| v3_fullyartificial_t2w | fullyartificial | t2w | 1 | 0.5813 | 0.3024 | 0.3751 | 0.6691 | 0.6691 | 0.4196 | 0.3024 |
| v3_generator_t1w | generator | t1w | 1 | 0.4717 | 0.6834 | 0.6463 | 0.5304 | 0.6834 | 0.5495 | 0.4717 |
| v3_generator_t2w | generator | t2w | 1 | 0.7110 | 0.0897 | 0.2865 | 0.7455 | 0.7455 | 0.3624 | 0.0897 |
