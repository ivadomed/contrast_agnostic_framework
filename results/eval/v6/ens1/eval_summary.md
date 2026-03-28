# Evaluation Summary

| model_id | family | source_contrast | ckpt_exists | flair | t1w | t1gd | t2w | in_domain_dice | ood_mean_dice | ood_worst_dice |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| segmenter_baseline_flair | baseline | flair | 1 | 0.7930 | 0.0334 | 0.1871 | 0.3857 | 0.7930 | 0.2021 | 0.0334 |
| segmenter_baseline_t1gd | baseline | t1gd | 1 | 0.3582 | 0.6016 | 0.7042 | 0.0850 | 0.7042 | 0.3483 | 0.0850 |
| segmenter_baseline_t1w | baseline | t1w | 1 | 0.2162 | 0.7162 | 0.5889 | 0.1208 | 0.7162 | 0.3086 | 0.1208 |
| segmenter_baseline_t2w | baseline | t2w | 1 | 0.5212 | 0.0855 | 0.0934 | 0.8089 | 0.8089 | 0.2334 | 0.0855 |
| segmenter_fullyartificial_t1w | fullyartificial | t1w | 1 | 0.5436 | 0.6134 | 0.6127 | 0.5960 | 0.6134 | 0.5841 | 0.5436 |
| segmenter_fullyartificial_t2w | fullyartificial | t2w | 1 | 0.5162 | 0.2443 | 0.3550 | 0.6271 | 0.6271 | 0.3719 | 0.2443 |
