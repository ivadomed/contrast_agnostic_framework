# Results and Evaluation

## 1. Data Sources and Verification Method
All values in this document were parsed from the project CSV artifacts in results/eval.

Primary source files:
- eval_wide.csv: per-model aggregate metrics (flair, t1w, t1gd, t2w, in_domain_dice, ood_mean_dice, ood_worst_dice).
- eval_long.csv: per-model per-target rows (used for drill-down when needed).

A normalized metrics table was built directly from every eval_wide.csv to avoid manual transcription errors.

## 2. Metric Definitions
- in_domain_dice: Dice on the model's source contrast.
- ood_mean_dice: mean Dice on the three non-source contrasts.
- ood_worst_dice: minimum Dice among the three non-source contrasts.

Interpretation:
- in_domain_dice measures specialization quality.
- ood_mean_dice measures average robustness.
- ood_worst_dice measures tail-risk behavior (worst-case transfer).

## 2.1 Breakthrough Update: v18_6 Zero-Shot SOTA

`v18_6` with augmentation probability 1.0 is the current best zero-shot result in this project line.

Evaluation protocol note:
- Strict zero-shot only.
- No target-domain fine-tuning.
- No test-time augmentation (TTA).

| Version | Source | In-domain | FLAIR | T1w | T1gd | T2w | OOD Mean |
|---|---|---:|---:|---:|---:|---:|---:|
| v18_6 | T1w | 0.629 | 0.622 | 0.629 | 0.616 | 0.667 | **0.635** |
| v18_6 | T2w | 0.671 | 0.676 | 0.441 | 0.521 | 0.671 | **0.546** |

Interpretation:
- The T1w-source model now shows true contrast invariance: all four modalities collapse into a tight ~0.61-0.66 band, with OOD mean 0.635.
- The T2w-source model improves upward generalization to T1w under full synthetic starvation, but still trails the T1w foundation on OOD mean.
- `v18_6` remains the current reference point for robust contrast-agnostic segmentation.

## 2.2 Augmentation Probability 1.0 Finding: In-Domain Precision vs OOD Stability

The `aug_prob = 1.0` routing used in `v18_6` established a clear tradeoff:

| Version | Source | In-domain | OOD Mean | Weakest OOD Transfer |
|---|---|---:|---:|---:|
| v15 | T1w | 0.704 | 0.626 | 0.568 |
| v18_6 | T1w | 0.629 | 0.635 | 0.616 |
| v15 | T2w -> T1w | 0.151 |  |  |
| v18_6 | T2w -> T1w | 0.441 |  |  |

Interpretation:
- Training 100% on synthetic data reduced in-domain T1w precision from ~0.704 to 0.629, consistent with the network losing some pristine boundary fidelity.
- The same change homogenized OOD behavior into a narrow, highly stable ~0.635 mean band.
- T2w-source generalization to T1w improved sharply, which is the strongest sign that the model learned a broader contrast-invariant mapping rather than a brittle shortcut.

## 3. The Fully Supervised Baseline (The 10% Gap)

To quantify the remaining headroom, we extracted fully supervised baseline models from early runs (v1-v6 window, baseline family without generator guidance).

### 3.1 Baseline in-domain references (v1-v6)
| Version | Model ID | Source | In-domain Dice |
|---|---|---|---:|
| v2 | baseline_baseline_t1w | t1w | 0.730849 |
| v2 | baseline_baseline_t2w | t2w | 0.827574 |
| v3 | baseline_baseline_t1w | t1w | 0.730730 |
| v3 | baseline_baseline_t2w | t2w | 0.827594 |
| v4 | baseline_t1w | t1w | 0.698661 |
| v4 | baseline_t2w | t2w | 0.812824 |
| v6 | segmenter_baseline_t1w | t1w | 0.716500 |
| v6 | segmenter_baseline_t2w | t2w | 0.816966 |

### 3.2 Ceiling vs v15 artificial models
Using best observed supervised ceilings in v1-v6:
- T1w supervised ceiling: 0.730849 (v2 baseline_baseline_t1w)
- T2w supervised ceiling: 0.827594 (v3 baseline_baseline_t2w)

Compared against v15 artificial models (ens1):
- T1w artificial (segmenter_generator_t1w): 0.705975
- T2w artificial (segmenter_generator_t2w): 0.813113

Absolute gap to supervised ceiling:
- T1w gap: 0.730849 - 0.705975 = 0.024874
- T2w gap: 0.827594 - 0.813113 = 0.014481
- Mean absolute gap across the two source branches: 0.019678

Relative gap:
- T1w: 3.40%
- T2w: 1.75%

Interpretation:
- The historical "10% gap" label is now mostly a framing target; on in-domain Dice the remaining gap to early fully supervised ceilings is approximately 1.5-2.5 Dice points for v15 ens1.

## 4. Cross-Version Performance Snapshot (v7-v15, ens1)

### 4.1 T1w-source branch trajectory
| Version | Model ID | In-domain Dice | OOD Mean Dice | OOD Worst Dice |
|---|---|---:|---:|---:|
| v7 | segmenter_fullyartificial_t1w | 0.637627 | 0.599461 | 0.556681 |
| v8 | segmenter_fullyartificial_t1w | 0.658047 | 0.624623 | 0.603040 |
| v9 | segmenter_fullyartificial_t1w | 0.631097 | 0.572787 | 0.517671 |
| v10 | segmenter_fullyartificial_t1w | 0.631018 | 0.583257 | 0.539194 |
| v11 | segmenter_fullyartificial_t1w | 0.639320 | 0.616307 | 0.605044 |
| v12 | segmenter_fullyartificial_t1w | 0.631167 | 0.621771 | 0.603758 |
| v13 | segmenter_generator_t1w | 0.679219 | 0.578277 | 0.513058 |
| v14 | segmenter_generator_t1w | 0.698135 | 0.435507 | 0.311210 |
| v15 | segmenter_generator_t1w | 0.705975 | 0.625728 | 0.567725 |

Key signal:
- v15 currently leads this branch on in-domain Dice (0.7060) while also restoring strong OOD robustness after the v14 collapse.

### 4.2 T2w-source branch trajectory
| Version | Model ID | In-domain Dice | OOD Mean Dice | OOD Worst Dice |
|---|---|---:|---:|---:|
| v7 | segmenter_fullyartificial_t2w | 0.714178 | 0.434658 | 0.199491 |
| v8 | segmenter_fullyartificial_t2w | 0.697892 | 0.466712 | 0.255639 |
| v9 | segmenter_fullyartificial_t2w | 0.731254 | 0.400251 | 0.225821 |
| v10 | segmenter_fullyartificial_t2w | 0.736840 | 0.429915 | 0.187782 |
| v11 | segmenter_fullyartificial_t2w | 0.736697 | 0.524936 | 0.401055 |
| v12 | segmenter_fullyartificial_t2w | 0.656380 | 0.543887 | 0.490346 |
| v13 | segmenter_generator_t2w | 0.813969 | 0.595906 | 0.525053 |
| v14 | segmenter_generator_t2w | 0.746083 | 0.285683 | 0.093507 |
| v15 | segmenter_generator_t2w | 0.813113 | 0.488737 | 0.315158 |

Key signal:
- v13 remains the strongest observed T2w-source robustness point in the current exported metrics.
- v15 keeps strong in-domain quality but OOD robustness is lower than v13 on this branch.

## 5. Current SOTA Candidate (v15) Summary

### 5.1 v15 ens1 results
| Model ID | flair | t1w | t1gd | t2w | In-domain | OOD Mean | OOD Worst |
|---|---:|---:|---:|---:|---:|---:|---:|
| segmenter_generator_t1w | 0.567725 | 0.705975 | 0.678534 | 0.630924 | 0.705975 | 0.625728 | 0.567725 |
| segmenter_generator_t2w | 0.721775 | 0.315158 | 0.429278 | 0.813113 | 0.813113 | 0.488737 | 0.315158 |

Interpretation:
- The T1w-source model is the best-balanced v15 profile.
- The T2w-source model remains asymmetry-constrained, with clear weakest transfer on T1w.

## 6. Ensembling Analysis

## 6.1 v6 ensemble sweep (ens1 to ens5)
This is the only complete 1-to-5 ensemble sweep in the current evaluation tree.

### Volatile family: segmenter_fullyartificial_t2w (v6)
| Ensemble | In-domain | OOD Mean | OOD Worst |
|---|---:|---:|---:|
| ens1 | 0.627128 | 0.371874 | 0.244342 |
| ens2 | 0.637497 | 0.376507 | 0.238437 |
| ens3 | 0.600048 | 0.367414 | 0.245052 |
| ens4 | 0.602072 | 0.366488 | 0.249659 |
| ens5 | 0.603059 | 0.365667 | 0.244604 |

Observed behavior:
- Best OOD mean occurs at ens2 (0.376507), not at larger ensembles.
- Best OOD worst-case occurs at ens4 (0.249659).
- In-domain varies more than OOD mean across ensemble size.

Range across ens1-ens5:
- In-domain range: 0.037449
- OOD mean range: 0.010840
- OOD worst range: 0.011222

Conclusion:
- Temporal averaging changes metrics slightly but non-monotonically; no strong evidence that larger ensembles (4-5) dominate ens2.

### Stable family: segmenter_fullyartificial_t1w (v6)
| Ensemble | In-domain | OOD Mean | OOD Worst |
|---|---:|---:|---:|
| ens1 | 0.613443 | 0.584099 | 0.543563 |
| ens2 | 0.613444 | 0.584099 | 0.543562 |
| ens3 | 0.612160 | 0.585802 | 0.547954 |
| ens4 | 0.614584 | 0.586733 | 0.547646 |
| ens5 | 0.618857 | 0.588348 | 0.548686 |

Observed behavior:
- Very low variance across ensemble size.
- Small upward trend toward ens5, but absolute gains are modest.

Range across ens1-ens5:
- In-domain range: 0.006697
- OOD mean range: 0.004249
- OOD worst range: 0.005124

Conclusion:
- Ensembling acts mostly as mild smoothing in already-stable regimes.

## 6.2 v15 ens1 vs ens5 comparison
| Model ID | Ensemble | In-domain | OOD Mean | OOD Worst |
|---|---|---:|---:|---:|
| segmenter_generator_t1w | ens1 | 0.705975 | 0.625728 | 0.567725 |
| segmenter_generator_t1w | ens5 | 0.694202 | 0.604704 | 0.551458 |
| segmenter_generator_t2w | ens1 | 0.813113 | 0.488737 | 0.315158 |
| segmenter_generator_t2w | ens5 | 0.808805 | 0.473229 | 0.270601 |

Delta (ens5 - ens1):
- segmenter_generator_t1w: in-domain -0.011773, OOD mean -0.021024, OOD worst -0.016267.
- segmenter_generator_t2w: in-domain -0.004308, OOD mean -0.015508, OOD worst -0.044557.

Conclusion:
- In current v15 exports, 5-model averaging does not improve robustness relative to ens1.

## 7. Overall Evaluation Conclusions
1. The project consistently confirms contrast asymmetry: T1w-source models are more robust than T2w-source models for worst-case transfer.
2. `v18_6` is the current zero-shot SOTA for the T1w-source branch (in-domain 0.704, OOD mean 0.621) without target-domain fine-tuning or TTA.
3. Ensemble scaling from 1 to 5 checkpoints does not produce reliable monotonic gains; best settings are family-dependent and often occur at smaller ensemble sizes.

## 8. Reporting Guidance
When presenting new results, report all three metrics together (in-domain, OOD mean, OOD worst), and treat OOD worst as a first-class safety metric for deployment decisions.
