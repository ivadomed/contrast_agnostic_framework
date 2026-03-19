# Evaluation Summary

| model_id | family | source_contrast | ckpt_exists | flair | t1w | t1gd | t2w | in_domain_dice | ood_mean_dice | ood_worst_dice |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_flair | baseline | flair | 1 | 0.8380 | 0.0460 | 0.2305 | 0.4090 | 0.8380 | 0.2285 | 0.0460 |
| baseline_t1gd | baseline | t1gd | 1 | 0.3456 | 0.6251 | 0.7241 | 0.0832 | 0.7241 | 0.3513 | 0.0832 |
| generator_t1w | generator | t1w | 1 | 0.4489 | 0.7486 | 0.7048 | 0.5009 | 0.7486 | 0.5515 | 0.4489 |
| baseline_t2w | baseline | t2w | 1 | 0.5348 | 0.0954 | 0.1055 | 0.8205 | 0.8205 | 0.2452 | 0.0954 |
| baseline_baseline_flair | baseline | flair | 1 | 0.8380 | 0.0460 | 0.2305 | 0.4090 | 0.8380 | 0.2285 | 0.0460 |
| baseline_baseline_t1w | baseline | t1w | 1 | 0.1571 | 0.6987 | 0.5764 | 0.0974 | 0.6987 | 0.2770 | 0.0974 |
| baseline_baseline_t2w | baseline | t2w | 1 | 0.5348 | 0.0954 | 0.1055 | 0.8205 | 0.8205 | 0.2452 | 0.0954 |
| v4_fullyartificial_t1w | fullyartificial | t1w | 1 | 0.5908 | 0.5747 | 0.5897 | 0.5955 | 0.5747 | 0.5920 | 0.5897 |
| v4_fullyartificial_t2w | fullyartificial | t2w | 1 | 0.5327 | 0.1422 | 0.3541 | 0.5080 | 0.5080 | 0.3430 | 0.1422 |
