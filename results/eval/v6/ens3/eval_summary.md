# Evaluation Summary

| model_id | family | source_contrast | ckpt_exists | flair | t1w | t1gd | t2w | in_domain_dice | ood_mean_dice | ood_worst_dice |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| segmenter_baseline_t1w | baseline | t1w | 1 | 0.2167 | 0.7158 | 0.5889 | 0.1197 | 0.7158 | 0.3084 | 0.1197 |
| segmenter_baseline_t2w | baseline | t2w | 1 | 0.5161 | 0.0875 | 0.0934 | 0.8106 | 0.8106 | 0.2323 | 0.0875 |
| segmenter_fullyartificial_t1w | fullyartificial | t1w | 1 | 0.5480 | 0.6122 | 0.6134 | 0.5960 | 0.6122 | 0.5858 | 0.5480 |
| segmenter_fullyartificial_t2w | fullyartificial | t2w | 1 | 0.5053 | 0.2451 | 0.3519 | 0.6000 | 0.6000 | 0.3674 | 0.2451 |
