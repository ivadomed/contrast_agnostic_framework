# Evaluation Summary

| model_id | family | source_contrast | ckpt_exists | flair | t1w | t1gd | t2w | in_domain_dice | ood_mean_dice | ood_worst_dice |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| segmenter_baseline_t1w | baseline | t1w | 1 | 0.1825 | 0.6698 | 0.4995 | 0.0823 | 0.6698 | 0.2548 | 0.0823 |
| segmenter_baseline_t2w | baseline | t2w | 1 | 0.3368 | 0.0986 | 0.0980 | 0.4562 | 0.4562 | 0.1778 | 0.0980 |
| segmenter_fullyartificial_t1w | fullyartificial | t1w | 1 | 0.0470 | 0.1424 | 0.1647 | 0.0363 | 0.1424 | 0.0827 | 0.0363 |
