"""
Single source of truth for the ToothFairy2 label reduction.

The released ToothFairy2 dataset carries 42 used ids out of a 0-48 numbering (see
LABELS_SOURCE below, verbatim from the challenge's own dataset.json). This project
does NOT train on all 42. The reduction here is deliberate and is the thing to
change if the task definition is ever revisited — every converter, split script,
evaluator and results config imports from this module rather than restating a map.

WHY REDUCE AT ALL — AND WHY *THIS* REDUCTION
--------------------------------------------
The reduction below is NOT a matter of taste. It was derived from a full per-source-id
voxel histogram over all 480 volumes (00_utils/00_01_audit_source_labels.py ->
<BIDS_ROOT>/source_label_audit.json), which showed that ToothFairy2 ships TWO cohorts
with materially different annotation completeness:

                       F cohort (63)      P cohort (417)
  Lower Jawbone            63/63             416/417
  lower teeth (31-48)      62/63             393/417
  Pharynx                  63/63             417/417
  Upper Jawbone            61/63             124/417   <-- 30%
  upper teeth (11-28)      60/63             296/417   <-- 71%
  Maxillary Sinus          45/63               4/417   <-- 1%

The upper block is annotated unreliably in P, which is the bulk of the release. The
proof that this is an ANNOTATION gap and not anatomy: of P's 293 maxilla-missing
cases, 173 nonetheless have upper TEETH labelled — you cannot have upper teeth
without an upper jawbone, so the missing Upper Jawbone label is an omission, not an
edentulous maxilla. Training on such a class would be actively harmful: the maxilla
is plainly visible in the image while the label says "background", so the network is
explicitly supervised to miss it in ~70% of training cases.

So the task keeps ONLY the structures both cohorts annotate consistently — the
MANDIBULAR BLOCK plus the AIRWAY:

  1 mandible     (id 1)        479/480 cases
  2 lower_teeth  (ids 31-48)   455/480 cases  (the ~25 absent are plausibly
                               edentulous mandibles — a real clinical phenotype,
                               and rarer than maxillary edentulism, as expected)
  3 pharynx      (id 7)        480/480 cases

Consistency was checked on EXTENT, not just presence: mandible and lower_teeth median
volumes agree between cohorts to within ~10%. Pharynx raw volume looked 2x larger in
F — but that is field-of-view, not annotation style: F volumes have a median 82 mm
z-extent vs P's 51 mm, and as a FRACTION of the imaged volume the pharynx annotation
matches closely (F 1.63%, P 1.88%). That FOV heterogeneity is real and is why
01_create_splits stratifies on cohort.

All three survivors are boundary-defined targets — cortical bone against air and soft
tissue, enamel/dentine against bone, airway air against mucosa — which is exactly the
pole of the paper's dissociation this dataset was onboarded to populate.

WHY MERGE THE 16 LOWER TEETH INTO ONE CLASS
-------------------------------------------
Per-tooth FDI identity is a localization/instance problem — telling a first molar
from a second is about position along the arch, not about the tissue boundary — so 16
separate classes would add severe class imbalance and per-class variance while
measuring something other than what this project asks. One merged `lower_teeth` class
keeps the hard interface and keeps the results table readable.

WHAT IS DROPPED, AND WHY (all mapped to background)
---------------------------------------------------
* Upper Jawbone (2), Maxillary Sinuses (5, 6), upper teeth (11-28) — the unreliably
  annotated upper block, per the table above. This is the entire reason the task is
  mandible-centric.
* Inferior alveolar canals (3, 4). Excluded even though they are ToothFairy's
  signature structure: a ~2-3 mm neurovascular canal whose walls are frequently not
  visible as a distinct cortical boundary, so it is inferred as much as seen. That is
  the ambiguous, non-boundary-defined character this task exists to avoid, and it
  would let one thin high-variance class dominate a macro-averaged Dice. (A scoping
  decision, not a claim the annotations are poor — they are expert 3D annotations.)
* Prosthetics: Bridge (8), Crown (9), Implant (10). Manufactured material, not
  tissue, and the dominant source of metal beam-hardening streaks. A method that
  synthesizes TISSUE appearance has no notion of "titanium", so scoring it on
  implants would measure something it never claims to model. Mapping them to
  background rather than into `lower_teeth` keeps that class meaning "natural
  dentition" consistently across restored and unrestored patients.

EXTERNAL-COMPATIBILITY NOTE — this reduction is what makes hanseg work
----------------------------------------------------------------------
HaN-Seg (head-and-neck CT + MR-T1, 42 patients) annotates a single `Bone_Mandible`
structure that INCLUDES the lower dentition. This task's `mandible` and `lower_teeth`
are precisely the two halves of that: their UNION is HaN-Seg's class, exactly, with
no leftover. Cross-dataset evaluation therefore compares
(predicted mandible ∪ lower_teeth) against HaN-Seg's Bone_Mandible — an exact
correspondence rather than the approximate one the original 5-class reduction would
have forced. See MANDIBLE_UNION_* below. This is the same one-class-overlap shape as
chaos -> sliver07 (liver only).
"""
from __future__ import annotations

# Verbatim from the ToothFairy2 challenge dataset.json (release 20/04/2024).
LABELS_SOURCE: dict[str, int] = {
    "background": 0,
    "Lower Jawbone": 1,
    "Upper Jawbone": 2,
    "Left Inferior Alveolar Canal": 3,
    "Right Inferior Alveolar Canal": 4,
    "Left Maxillary Sinus": 5,
    "Right Maxillary Sinus": 6,
    "Pharynx": 7,
    "Bridge": 8,
    "Crown": 9,
    "Implant": 10,
}
# Teeth use FDI-ordered contiguous blocks with 2 unused ids after each quadrant's
# wisdom tooth (19/20, 29/30, 39/40 are "NA" placeholders that never occur).
UPPER_RIGHT_TEETH = list(range(11, 19))
UPPER_LEFT_TEETH = list(range(21, 29))
LOWER_LEFT_TEETH = list(range(31, 39))
LOWER_RIGHT_TEETH = list(range(41, 49))
ALL_TEETH_IDS = UPPER_RIGHT_TEETH + UPPER_LEFT_TEETH + LOWER_LEFT_TEETH + LOWER_RIGHT_TEETH
LOWER_TEETH_IDS = LOWER_LEFT_TEETH + LOWER_RIGHT_TEETH
UPPER_TEETH_IDS = UPPER_RIGHT_TEETH + UPPER_LEFT_TEETH

# ── the reduced task ─────────────────────────────────────────────────────────
# name -> the source ids that collapse into it. Order defines the target ids
# (1..5), which is what lands in the nnU-Net dataset.json and in every metrics
# CSV column, so DO NOT reorder without regenerating everything downstream.
TARGET_LABELS: dict[str, list[int]] = {
    "mandible":    [1],
    "lower_teeth": LOWER_TEETH_IDS,
    "pharynx":     [7],
}
DROPPED_TO_BACKGROUND: dict[str, list[int]] = {
    # The unreliably-annotated upper block (see the table in the module docstring).
    "upper_jawbone":           [2],
    "maxillary_sinus":         [5, 6],
    "upper_teeth":             UPPER_TEETH_IDS,
    "inferior_alveolar_canal": [3, 4],
    "prosthetics":             [8, 9, 10],
}

# target id -> name, and the nnU-Net dataset.json "labels" block (name -> id).
TARGET_ID_TO_NAME: dict[int, str] = {i + 1: n for i, n in enumerate(TARGET_LABELS)}
NNUNET_LABELS: dict[str, int] = {"background": 0, **{n: i + 1 for i, n in enumerate(TARGET_LABELS)}}

# For the hanseg cross-dataset comparison: the UNION of these two target classes is
# exactly HaN-Seg's single Bone_Mandible structure (see module docstring).
MANDIBLE_UNION_IDS_SOURCE: list[int] = [1] + LOWER_TEETH_IDS
MANDIBLE_UNION_TARGET_IDS: list[int] = [NNUNET_LABELS["mandible"], NNUNET_LABELS["lower_teeth"]]


def build_remap() -> dict[int, int]:
    """source label id -> reduced target id (everything unlisted -> 0)."""
    remap: dict[int, int] = {}
    for tgt_id, name in TARGET_ID_TO_NAME.items():
        for src in TARGET_LABELS[name]:
            remap[src] = tgt_id
    return remap


if __name__ == "__main__":  # quick human check: bash -c '.venv/bin/python <this>'
    remap = build_remap()
    print(f"{len(remap)} source ids -> {len(TARGET_LABELS)} foreground classes")
    for tid, name in TARGET_ID_TO_NAME.items():
        print(f"  {tid}: {name:16s} <- {TARGET_LABELS[name]}")
    for name, ids in DROPPED_TO_BACKGROUND.items():
        print(f"  0: (dropped) {name} <- {ids}")
