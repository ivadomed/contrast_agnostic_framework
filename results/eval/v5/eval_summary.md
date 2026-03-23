# Evaluation Summary

| model_id | family | source_contrast | ckpt_exists | flair | t1w | t1gd | t2w | in_domain_dice | ood_mean_dice | ood_worst_dice |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| segmenter_baseline_t1gd | baseline | t1gd | 1 | 0.2615 | 0.2933 | 0.3654 | 0.0774 | 0.3654 | 0.2107 | 0.0774 |
| segmenter_baseline_t1w | baseline | t1w | 1 | 0.1456 | 0.3789 | 0.2876 | 0.0910 | 0.3789 | 0.1747 | 0.0910 |
| segmenter_baseline_t2w | baseline | t2w | 1 | 0.3376 | 0.0887 | 0.0900 | 0.4746 | 0.4746 | 0.1721 | 0.0887 |
| segmenter_fullyartificial_t1w | fullyartificial | t1w | 1 | 0.3193 | 0.2121 | 0.2336 | 0.2212 | 0.2121 | 0.2580 | 0.2212 |
| segmenter_fullyartificial_t2w | fullyartificial | t2w | 1 | 0.4044 | 0.1273 | 0.2124 | 0.4782 | 0.4782 | 0.2480 | 0.1273 |
