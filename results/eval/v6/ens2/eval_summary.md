# Evaluation Summary

| model_id | family | source_contrast | ckpt_exists | flair | t1w | t1gd | t2w | in_domain_dice | ood_mean_dice | ood_worst_dice |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| segmenter_baseline_t1w | baseline | t1w | 1 | 0.2162 | 0.7162 | 0.5889 | 0.1208 | 0.7162 | 0.3086 | 0.1208 |
| segmenter_baseline_t2w | baseline | t2w | 1 | 0.5212 | 0.0855 | 0.0934 | 0.8089 | 0.8089 | 0.2334 | 0.0855 |
| segmenter_fullyartificial_t1w | fullyartificial | t1w | 1 | 0.5436 | 0.6134 | 0.6127 | 0.5960 | 0.6134 | 0.5841 | 0.5436 |
| segmenter_fullyartificial_t2w | fullyartificial | t2w | 1 | 0.5218 | 0.2384 | 0.3693 | 0.6375 | 0.6375 | 0.3765 | 0.2384 |
