# Evaluation Summary

| model_id | family | source_contrast | ckpt_exists | flair | t1w | t1gd | t2w | in_domain_dice | ood_mean_dice | ood_worst_dice |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| segmenter_baseline_t1w | baseline | t1w | 1 | 0.2141 | 0.7158 | 0.5885 | 0.1187 | 0.7158 | 0.3071 | 0.1187 |
| segmenter_baseline_t2w | baseline | t2w | 1 | 0.5177 | 0.0863 | 0.0914 | 0.8128 | 0.8128 | 0.2318 | 0.0863 |
| segmenter_fullyartificial_t1w | fullyartificial | t1w | 1 | 0.5476 | 0.6146 | 0.6140 | 0.5985 | 0.6146 | 0.5867 | 0.5476 |
| segmenter_fullyartificial_t2w | fullyartificial | t2w | 1 | 0.5020 | 0.2497 | 0.3478 | 0.6021 | 0.6021 | 0.3665 | 0.2497 |
