# AI Copilot Guidelines

This file is a strict operational rulebook for all AI agents working in this repository.

## Mandatory Rules

- [ ] **Rule 1 - Strict Adherence Before Any Code Change**
Read and respect both docs/AI_COPILOT_GUIDELINES.md and docs/REFACTOR_LOG.md before writing or modifying any code.

- [ ] **Rule 2 - Optimization SLO Compliance Is Required**
All implementation work must target highly optimized execution.
After launch, verify:
- Generator runtime: < 1.5 mins/epoch
- Segmenter runtime (with consistency loss): < 14s/epoch

- [ ] **Rule 3 - Training Must Run in tmux**
All training runs must be started inside a tmux session to prevent hang-ups and accidental session loss.

- [ ] **Rule 4 - Use Only Approved Launch Scripts**
Never launch training with raw python commands.
Always use repository launch scripts that manage set_slot and CUDA visibility.

Approved usage patterns:
- bash scripts/run_generators.sh <SLOT_ID> <VERSION> <CONTRAST>
- bash scripts/run_segmenters.sh <SLOT_ID> <VERSION> <CONTRAST>

Default contrast policy:
- For routine generator runs, use <CONTRAST>=t1w.
- Do not schedule routine generator jobs with <CONTRAST>=t2w.

- [ ] **Rule 5 - Active Monitoring Protocol (First 8 Minutes Minimum)**
After starting a training script, actively monitor terminal output for approximately 8 minutes.
Do not mark the task complete until:
1. MONAI cache loading is fully finished.
2. Epoch 1 completes successfully.
3. No CUDA OOM or fatal runtime error appears.

- [ ] **Rule 6 - Dependency Chain Enforcement**
For every new version:
1. Finish generator training to 100% completion first.
2. Only then launch the corresponding segmenter training for that same version.

- [ ] **Rule 7 - Generator Source Policy (T1w-Only)**
Generator training launches are T1w-only by default.
Do not launch new T2w-source generator trainings unless an explicit, pre-approved experiment plan says otherwise.

Rationale:
- Saves compute slots for the highest-yield branch.
- Reflects established physics asymmetry and the project pivot to T1w-first generation.

## Required Operational Workflow

1. Open or attach to tmux session.
2. Review docs/REFACTOR_LOG.md and this guideline file.
3. Launch generator with approved script.
4. Monitor for at least 8 minutes and confirm first epoch success.
5. Confirm generator training completion.
6. Launch segmenter with approved script.
7. Monitor for at least 8 minutes and confirm first epoch success.
8. Report measured epoch times against SLOs.

## Quality Gate Before Declaring Completion

Do not finalize a training task unless all conditions are true:
- Training launched via approved bash script.
- Run occurred inside tmux.
- Monitoring protocol completed.
- No OOM/fatal startup failure.
- Epoch-time SLO checks recorded.
- Generator-to-segmenter dependency order respected.

## Non-Compliance Policy

If any mandatory rule is violated, the run is considered non-compliant and must be restarted under this policy.
