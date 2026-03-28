### NEW ISSUE: v4 Migration to a Highly Fixed and Reproducible Training Environment
**Description:**
This issue captures the main v4 objective: hardening the project into a reproducible research system rather than introducing a new contrast-selection philosophy.

**Context:**
- The scientific direction of using artificial contrasts had already been established earlier (Issue 4 and follow-ups).
- v4 was about making experiments stable, replayable, and auditable end-to-end.

**What changed in v4:**
- Moved to PyTorch Lightning modules for both synthesis and segmentation training.
- Introduced Hydra configuration hierarchy to centralize data/model/training/logging controls.
- Standardized seeding, logging, checkpointing, and resume behavior.
- Added reproducibility metadata capture (config snapshots, run context) and test coverage around core histogram operators.

**Why v4 matters:**
- Reduced script-level variance and operator-dependent run drift.
- Made result comparisons across runs and versions credible.
- Created the foundation required for later speed work (v5/v6) to be measured fairly.

**Takeaway:**
- v4 is the reproducibility and systems-discipline milestone of the project.

---

### NEW ISSUE: v5 Speed-First Optimization Phase and Throughput Engineering
**Description:**
This issue tracks the v5 phase as a performance engineering push focused on reducing epoch time and removing avoidable compute overhead.

**Primary v5 focus:**
- Throughput optimization rather than redefining the core scientific hypothesis.
- Elimination of obvious host-device sync penalties and expensive non-vectorized paths.

**Key optimization themes introduced in v5 window:**
- GPU vectorization in histogram and augmentation-adjacent paths.
- Separable 1D Gaussian convolutions replacing dense 3D blur kernels where mathematically equivalent.
- Logging and data-pipeline pressure reduction to improve steady-state GPU utilization.
- Compile-boundary and wrapper cleanup to reduce graph churn and runtime overhead.

**Important caveat:**
- v5 also contained instability periods in downstream metrics, so this should be documented as a speed-focused phase with mixed quality outcomes.

**Takeaway:**
- v5 is the first major acceleration cycle that exposed both performance wins and fragility points later addressed in v6.

---

### NEW ISSUE: v6 Speed Consolidation Plus Critical Kornia Bug Fixes
**Description:**
This issue consolidates the v6 lifecycle as a continuation of speed work plus critical correctness fixes uncovered during the Kornia migration.

**Part A: Critical correctness bugs fixed in v6**

We finished the forensic pass on the Kornia migration and confirmed the three severe failure modes that temporarily invalidated training dynamics.

**1) Mask Annihilation bug (bilinear interpolation on binary labels):**
- Symptom: Dice collapsed toward near-zero while CE could still decrease, with white-mask/empty-mask pathological predictions.
- Root cause: label masks were passing through bilinear-like interpolation paths, shrinking small tumor regions below threshold and eroding categorical boundaries.
- Fix: enforce synchronized image/mask warps but with **nearest-neighbor mask resampling** and explicit mask-safe affine handling.

**2) Pipeline execution-order bug (noise before generator path):**
- Symptom: generator-guided supervision became mathematically inconsistent with intended target-histogram guidance assumptions.
- Root cause: augmentation/noise operations were applied in an order that changed the input distribution before the generator stage in a way the loss design did not assume.
- Fix: re-established the intended operation order so generator synthesis/guidance math is computed on the proper representation.

**3) Nearest-neighbor aliasing during low-resolution simulation:**
- Symptom: staircase/block artifacts and unstable supervision signals during heavy low-res augmentation.
- Root cause: low-resolution down/up-sampling path introduced aliasing that damaged anatomy fidelity.
- Fix: corrected sampling strategy and return contracts so image/mask paths stay synchronized and categorical semantics are preserved.

**Additional hardening done in same window:**
- Removed Python `random` branching from compiled/distributed-critical augmentation paths and replaced with tensor RNG to avoid host-side nondeterministic control flow.
- Fixed inconsistent tuple-vs-tensor augmentation return signatures that previously caused runtime unpack failures in segmenter mode.

**Part B: Continued speed work and profiling in v6**

We completed deeper profiling to explain why `segmenter+generator` training could still inflate to around 18s/epoch while baseline segmenter remained much faster.

**Initial bottlenecks identified:**
- `aten::copy_` dominated (~30% GPU time), indicating heavy memory-format and copy churn.
- 3D conv backward/forward remained a major expected cost center (generator U-Net overhead).
- `DifferentiableHistogram3D` consumed a large fixed slice (~9-10% in trace).
- InstanceNorm and sequential multi-loss graph execution added non-trivial overhead.

**Optimizations implemented in this cycle:**
- Eliminated redundant guidance-map recomputation by reusing generated guidance tensors.
- Removed avoidable `clone()` / forced `contiguous()` calls in histogram and transfer paths.
- Vectorized histogram/guidance internals and retained strided-quantile sampling to cap sort pressure.
- Applied separable 1D Gaussian convolutions in place of dense 3D kernels in critical blur paths.
- Hardened compile boundaries and wrapper modules to reduce Dynamo/Inductor graph fragmentation.

**Measured effect from the profiling pass:**
- Copy/clone counts dropped (fewer `aten::copy_` and `aten::clone` events).
- Aggregate GPU trace improved modestly (~1-2% in the profiled window), but not enough alone to explain the full 18s to sub-10s gap.

**Research conclusion:**
- The remaining slowdown is mostly **architectural**, not a single hidden bug: generator forward/backward + dual histogram passes + multi-term losses are inherently expensive in 3D.
- v6 improved robustness and retained speed-oriented optimizations from earlier waves, but the current generator-coupled path still needs algorithmic simplification (e.g., lighter histogram/model/loss footprint) to robustly stay under 10s/epoch across settings.

**Next optimization candidates already identified:**
- Histogram dimensionality reduction (bins and/or spatial downsampling with calibration).
- Loss-kernel fusion to reduce sequential launch overhead.
- Generator capacity trim experiments with quality guardrails.

**Takeaway:**
- v6 should be framed as speed consolidation plus correctness recovery, not as a purely new scientific direction.

---

### REPLY TO EXISTING ISSUE: v6 Ensembling Evaluation Results
**Comment:**
### Update: v6 Ensemble Sweep (1 to 5 Models) and Heuristic Validation

Because validation is fully artificial, we explicitly questioned whether the single "best" checkpoint might simply be a model that benefited from an easier synthetic validation slice rather than being truly robust.

To test this, we evaluated temporal ensembling from 1 to 5 checkpoints (saved at separate training times) and compared each ensemble to the single selected best model.

**Primary quantitative signal (most volatile model family: `segmenter_fullyartificial_t2w`):**

| Ensemble Size | flair | t1w | t1gd | t2w | in_domain_dice | ood_mean_dice | ood_worst_dice |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.5409 | 0.2203 | 0.3452 | 0.6406 | 0.6406 | 0.3688 | 0.2203 |
| 2 | 0.5218 | 0.2384 | 0.3693 | 0.6375 | 0.6375 | **0.3765** | 0.2384 |
| 3 | 0.5053 | 0.2451 | 0.3519 | 0.6000 | 0.6000 | 0.3674 | 0.2451 |
| 4 | 0.5020 | **0.2497** | 0.3478 | 0.6021 | 0.6021 | 0.3665 | **0.2497** |
| 5 | 0.5042 | 0.2446 | 0.3482 | 0.6031 | 0.6031 | 0.3657 | 0.2446 |

**Interpretation:**
- The measured changes are small and non-monotonic across ensemble sizes.
- Best OOD mean appears at **2-model ensemble** (`0.3765`, +0.0077 vs single model), while other settings do not maintain consistent gains.
- Worst-case OOD improves in some larger ensembles, but this is accompanied by in-domain drift and no clear overall dominance pattern.

**Stable regime check (`segmenter_fullyartificial_t1w`):**
- Metrics are already stable and high; ensembling changes remain minimal (small oscillations around `ood_mean_dice ~0.584-0.588`).

**Verdict on heuristic success:**
- **Not a significant win.** Temporal ensembling in this form does not provide a strong enough robustness improvement to justify continued use as a default strategy.
- We will stop using this ensembling approach for now.
- This result also increases confidence that the current validation approach is not systematically selecting "easy-validation winners": the single best checkpoint does not appear to be an artifact of an unusually easy artificial validation draw.

**Next action point (deferred):**
- A plausible follow-up is to train on fully artificial data and then fine-tune only the decoder on a chosen target contrast.
- We are **not** pursuing this now because it would partially sacrifice the strict zero-shot framing that this line of work is designed to test.

---

### NEW ISSUE: v6 ens1 Evaluation Snapshot (Single-Checkpoint Reference)
**Description:**
This issue logs the v6 `ens1` results as the reference point for all temporal ensembling comparisons, now completed with the newly available baseline references for `flair` and `t1gd` source contrasts.

**Why this issue exists:**
- `ens1` is the selected single-checkpoint baseline.
- All `ens2` to `ens5` conclusions depend on an explicit, frozen `ens1` reference.

**v6 ens1 key results (current ensemble export):**

| model_id | flair | t1w | t1gd | t2w | in_domain_dice | ood_mean_dice | ood_worst_dice |
|---|---:|---:|---:|---:|---:|---:|---:|
| segmenter_baseline_t1w | 0.2162 | 0.7162 | 0.5889 | 0.1208 | 0.7162 | 0.3086 | 0.1208 |
| segmenter_baseline_t2w | 0.5212 | 0.0855 | 0.0934 | 0.8089 | 0.8089 | 0.2334 | 0.0855 |
| segmenter_fullyartificial_t1w | 0.5436 | 0.6134 | 0.6127 | 0.5960 | 0.6134 | 0.5841 | 0.5436 |
| segmenter_fullyartificial_t2w | 0.5290 | 0.2443 | 0.3562 | 0.6431 | 0.6431 | 0.3765 | 0.2443 |

**Additional baseline references (newly available):**

| model_id | flair | t1w | t1gd | t2w | in_domain_dice | ood_mean_dice | ood_worst_dice |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_flair | 0.8267 | 0.0448 | 0.2453 | 0.3869 | 0.8267 | 0.2257 | 0.0448 |
| baseline_t1gd | 0.3140 | 0.6598 | 0.7428 | 0.0943 | 0.7428 | 0.3560 | 0.0943 |

**Interpretation:**
- The strongest cross-contrast profile remains `segmenter_fullyartificial_t1w` with balanced scores and high `ood_mean_dice`.
- `segmenter_fullyartificial_t2w` remains the more volatile family and is therefore the critical stress-test target for ensemble analysis.
- `baseline_flair` and `baseline_t1gd` complete the baseline source-contrast coverage and improve interpretability of source-dependent generalization asymmetry.
- This `ens1`-anchored snapshot is now the canonical baseline for future robustness checks.

**Actionable use:**
- Keep this issue as the stable anchor when reporting any new ensemble policy, checkpoint selection rule, or zero-shot validation changes.
