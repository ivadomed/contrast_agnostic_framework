# ToothFairy2 splits

- cases: 479  test: 71  train-pool: 408
- excluded (no mandible): 1 ['toothfairy2_ToothFairy2P077']
- folds: 3 (CLAUDE.md fold policy: 0 1 2 only)
- stratifier: cohort (F/P) then lower-teeth voxel count; median burden test=31209 pool=31386

- test set: 71 cases (9 F / 62 P)

| fold | train | val | F in val | median lower-teeth voxels (val) |
|---|---|---|---|---|
| 0 | 272 | 136 | 18 | 35684 |
| 1 | 272 | 136 | 18 | 30362 |
| 2 | 272 | 136 | 18 | 30376 |
