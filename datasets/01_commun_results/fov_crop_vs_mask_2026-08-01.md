# FOV hard-crop vs masked-eval — cross-dataset comparison (2026-08-01)

**What changed.** The standing pipeline applies the CHAOS FOV slab as a **mask at
evaluation time** (zeroing pred+GT outside the slab); the network still *sees* the
whole full-extent test volume at inference. This run instead **hard-crops each
volume in place before prediction**, so the model sees only the chaos-equivalent
slab as an input condition. Driver:
`datasets/00_commun_scripts/00_02_predict/fov_crop_predict_evaluate.sh`.
Results live in a dedicated `fov_crop/` metrics subdir per dataset; **headline
masked-eval numbers are untouched and these are NOT wired into the roll-up
configs** — switching the suite would change every published number, so that is a
deliberate decision left open.

**Crop aggressiveness** (mean fraction of the S-I extent kept):
amos ct 0.486 / amos mri 0.828 / sliver07 ct 0.880. The deltas below track this:
the more that is removed, the more cropping matters.

**Headline reading**
- Cropping helps nearly universally; amos CT gains most (+3 to +8), cirrmri-liver
  is roughly neutral.
- OURS takes the top CT spot on amos (83.1 t1in / 81.4 t2spir) and on
  msd-spleen t1in (81.5).
- On cirrmri-liver the main beneficiary is **srcsm** (+3.2 to +4.6), which becomes
  the best method there — consistent with srcsm being the arm most sensitive to
  FOV/geometry context (see datasets/kidney-t2w/ACQUISITION_PLANE_INVESTIGATION.md).
- **baseline is the only consistent loser** — the one arm with no augmentation, so
  out-of-slab context was helping rather than hurting it.
- kidney-t2w is excluded from the roll-ups entirely (coronal acquisition, so its
  through-plane FOV is not comparable to chaos's axial slab).

**Provenance.** TamIA jobs 391945-949 (crop+predict) then per-contrast CPU-only
eval 391998/391999 (cirrmri), 392034/392035 + fix-up 392042/392043 (amos).
The amos P=24 eval pass OOM'd on 26/96 `ct` tasks (BrokenProcessPool: 48
processes each holding a full-torso CT) and its logging counted the failures as
successes; caught by auditing exact row counts, re-run at P=6, re-verified to
0 missing/short. All numbers below are from row-count-verified CSVs
(ct = 400 rows = 100 cases x 4 organs; mri = 80; cirrmri t1/t2 = 310/318).

Values are mean Dice x100, cross-fold (folds 0-2), `nan` organs excluded.

```

### sliver07 / chaos_t1in — mean Dice x100  (crop = hard-cropped input, mask = current headline)
method                                  ct crop    ct mask       Δ
------------------------------------------------------------------
baseline                                    7.8        6.2    +1.6
auglab_default                             91.8       91.0    +0.8
synthseg_noEM                              70.0       69.5    +0.4
synthseg_EM                                91.1       90.5    +0.6
v26_6_2_train050_val100                    92.0       91.1    +0.9
srcsm                                      90.8       90.4    +0.4
auglabAug_v26_6_2_train050_val000          91.4       90.8    +0.7
auglabAug_v26_6_2_train050_val100          91.4       91.1    +0.3

### sliver07 / chaos_t2spir — mean Dice x100  (crop = hard-cropped input, mask = current headline)
method                                  ct crop    ct mask       Δ
------------------------------------------------------------------
baseline                                   59.2       64.1    -4.8
auglab_default                             91.3       88.6    +2.6
synthseg_noEM                              63.2       60.3    +2.9
synthseg_EM                                90.8       88.8    +2.0
v26_6_2_train050_val100                    90.3       88.4    +2.0
srcsm                                      85.4       84.3    +1.1
auglabAug_v26_6_2_train050_val000          90.2       89.3    +0.9
auglabAug_v26_6_2_train050_val100          90.3       89.5    +0.8

### msd-spleen / chaos_t1in — mean Dice x100  (crop = hard-cropped input, mask = current headline)
method                                  ct crop    ct mask       Δ
------------------------------------------------------------------
baseline                                    0.5        1.1    -0.6
auglab_default                             76.0       73.1    +2.9
synthseg_noEM                              57.2       51.1    +6.1
synthseg_EM                                79.2       76.2    +3.0
v26_6_2_train050_val100                    80.7       79.0    +1.7
srcsm                                      81.2       79.9    +1.2
auglabAug_v26_6_2_train050_val000          81.5       79.4    +2.1
auglabAug_v26_6_2_train050_val100          80.8       78.5    +2.4

### msd-spleen / chaos_t2spir — mean Dice x100  (crop = hard-cropped input, mask = current headline)
method                                  ct crop    ct mask       Δ
------------------------------------------------------------------
baseline                                    8.7        8.9    -0.2
auglab_default                             80.1       77.7    +2.4
synthseg_noEM                              48.7       44.0    +4.7
synthseg_EM                                70.4       69.1    +1.4
v26_6_2_train050_val100                    72.7       69.5    +3.2
srcsm                                      75.4       72.8    +2.6
auglabAug_v26_6_2_train050_val000          77.5       74.8    +2.8
auglabAug_v26_6_2_train050_val100          76.6       75.4    +1.2

### amos / chaos_t1in — mean Dice x100  (crop = hard-cropped input, mask = current headline)
method                                  ct crop    ct mask       Δ   mri crop   mri mask       Δ
------------------------------------------------------------------------------------------------
baseline                                    1.7        1.9    -0.2        3.2        2.2    +1.0
auglab_default                             78.4       73.2    +5.1       88.2       86.1    +2.1
synthseg_noEM                              53.8       55.0    -1.1       72.4       71.4    +1.0
synthseg_EM                                80.4       75.5    +5.0       87.7       86.7    +0.9
v26_6_2_train050_val100                    82.7       75.6    +7.1       89.8       87.3    +2.5
srcsm                                      79.6       76.6    +3.0       85.6       84.9    +0.7
auglabAug_v26_6_2_train050_val000          82.8       79.4    +3.4       87.8       86.6    +1.2
auglabAug_v26_6_2_train050_val100          83.1       79.7    +3.3       87.5       86.3    +1.2

### amos / chaos_t2spir — mean Dice x100  (crop = hard-cropped input, mask = current headline)
method                                  ct crop    ct mask       Δ   mri crop   mri mask       Δ
------------------------------------------------------------------------------------------------
baseline                                    9.2       10.7    -1.5       44.2       42.4    +1.7
auglab_default                             78.9       72.5    +6.4       89.5       88.1    +1.4
synthseg_noEM                              54.0       47.2    +6.8       73.2       70.0    +3.2
synthseg_EM                                76.7       72.6    +4.0       87.2       86.4    +0.8
v26_6_2_train050_val100                    74.3       66.1    +8.2       90.2       87.6    +2.6
srcsm                                      75.0       71.1    +3.9       90.3       90.2    +0.1
auglabAug_v26_6_2_train050_val000          81.3       76.6    +4.7       88.7       87.9    +0.8
auglabAug_v26_6_2_train050_val100          81.4       77.8    +3.6       88.0       87.5    +0.5

### cirrmri-liver / chaos_t1in — mean Dice x100  (crop = hard-cropped input, mask = current headline)
method                                  t1 crop    t1 mask       Δ    t2 crop    t2 mask       Δ
------------------------------------------------------------------------------------------------
baseline                                   15.2       12.2    +3.0       10.6       12.0    -1.4
auglab_default                             84.4       84.1    +0.3       81.3       81.0    +0.3
synthseg_noEM                              72.2       78.7    -6.5       50.5       55.4    -4.9
synthseg_EM                                83.3       83.9    -0.6       80.7       81.1    -0.4
v26_6_2_train050_val100                    74.4       74.8    -0.4       71.2       72.2    -1.0
srcsm                                      85.1       80.5    +4.6       82.1       78.4    +3.7
auglabAug_v26_6_2_train050_val000          81.1       81.0    +0.1       79.0       79.5    -0.4
auglabAug_v26_6_2_train050_val100          81.0       80.7    +0.2       79.1       79.8    -0.7

### cirrmri-liver / chaos_t2spir — mean Dice x100  (crop = hard-cropped input, mask = current headline)
method                                  t1 crop    t1 mask       Δ    t2 crop    t2 mask       Δ
------------------------------------------------------------------------------------------------
baseline                                   23.5       22.8    +0.8       22.6       21.3    +1.3
auglab_default                             85.8       84.4    +1.4       74.5       73.6    +1.0
synthseg_noEM                              76.0       74.1    +1.9       38.2       37.9    +0.3
synthseg_EM                                86.0       85.8    +0.2       79.7       80.0    -0.3
v26_6_2_train050_val100                    66.4       67.7    -1.3       60.2       62.8    -2.6
srcsm                                      85.1       81.9    +3.2       78.0       73.9    +4.1
auglabAug_v26_6_2_train050_val000          82.4       81.6    +0.8       77.7       76.8    +0.9
auglabAug_v26_6_2_train050_val100          81.5       81.1    +0.3       77.5       76.8    +0.7
```
