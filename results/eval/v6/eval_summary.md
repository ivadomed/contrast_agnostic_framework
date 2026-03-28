# Evaluation Summary

| model_id | family | source_contrast | ckpt_exists | flair | t1w | t1gd | t2w | in_domain_dice | ood_mean_dice | ood_worst_dice |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| segmenter_baseline_t1w | baseline | t1w | 1 | 0.2144 | 0.7165 | 0.5897 | 0.1187 | 0.7165 | 0.3076 | 0.1187 |
| segmenter_baseline_t2w | baseline | t2w | 1 | 0.5133 | 0.0869 | 0.0908 | 0.8170 | 0.8170 | 0.2304 | 0.0869 |
| segmenter_fullyartificial_t1w | fullyartificial | t1w | 1 | 0.5487 | 0.6189 | 0.6156 | 0.6008 | 0.6189 | 0.5883 | 0.5487 |
| segmenter_fullyartificial_t2w | fullyartificial | t2w | 1 | 0.4898 | 0.2678 | 0.3522 | 0.5801 | 0.5801 | 0.3699 | 0.2678 |
