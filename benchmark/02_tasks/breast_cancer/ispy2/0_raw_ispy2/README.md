# 0_raw_ispy2 — provenance (PRISTINE, do not modify)

Source: TCIA (The Cancer Imaging Archive) collection "I-SPY 2 Breast Dynamic
Contrast Enhanced MRI Trial (ISPY2)". Collection page:
https://www.cancerimagingarchive.net/collection/ispy2/

Li, W., Newitt, D. C., Gibbs, J., et al. (2022). I-SPY 2 Breast Dynamic
Contrast Enhanced MRI Trial (ISPY2) (Version 1) [Data set]. The Cancer
Imaging Archive. DOI: https://doi.org/10.7937/TCIA.D8Z0-9T85

## License — verified before any download (2026-09-02)

**CC BY 4.0** (Creative Commons Attribution 4.0 International). An earlier
Version 1 of this collection used CC BY-NC 4.0; the TCIA wiki collection page
(https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230072)
states the data submitter switched to CC BY 4.0 on 2022-07-08: *"On 7/8/2022
the data submitter signed off on using the CC BY 4.0 instead of the CC
BY-NC 4.0."* Verified independently three ways, matching the methodology
used for AMBL, none showing any residual NC restriction:
1. TCIA wiki collection page license statement (CC BY 4.0, with the
   2022-07-08 switch-over note above).
2. **Every single one of the 2688 SEG series records** returned by the live
   TCIA/NBIA REST API for this collection's T0 (baseline) timepoint carries
   its own license fields directly: `"LicenseName": "Creative Commons
   Attribution 4.0 International License"`, `"LicenseURI":
   "https://creativecommons.org/licenses/by/4.0/"` — checked exhaustively
   (all records via a Python set() over the full `ispy2_seg_series.json`,
   confirmed exactly one distinct (LicenseName, LicenseURI) pair across all
   2688 records).
3. The `LICENSE` file bundled inside every downloaded series zip states the
   same, verbatim: *"License Information. The ISPY2 collection is
   distributed under the CC BY 4.0 at
   https://creativecommons.org/licenses/by/4.0/. By downloading the data,
   you agree to abide by terms of this license."*

## Collection structure — verified via the API's own series metadata

- **719 subjects** in the collection (matches the collection page).
- **717 of 719 (99.7%) have a T0 (baseline, pre-treatment) DICOM-SEG
  "Analysis Mask"** — checked directly (enumerated all `Modality=SEG` series
  for the collection via `getSeries?Collection=ISPY2&Modality=SEG`, grouped
  by `StudyDesc`, counted `ISPY2_MRI_T0` rows: 717). The collection page's
  own claim ("Functional Tumor Volume analysis masks are provided for all
  patients") is close to true at baseline but not exactly ("all" vs
  717/719) — noted, not a material discrepancy on the scale of AMBL's
  "for all cases" claim (632 claimed vs 99 real, 15.7%).
- **717 is NOT the final usable-N for this project's two-modality
  (`t1wce`/`t2w`) convention.** Per-site VOLSER-crop variant and T2-sequence
  naming are genuinely inconsistent across the trial's 22 sites/vendors (see
  below) — after requiring BOTH a recognized VOLSER DCE series AND a real,
  correctly-identified (non-localizer, non-T1, non-fieldmap, non-fat-only)
  T2-weighted series, the final, directly-counted usable population is
  **561 of 719 (78.0% of all subjects; 78.2% of the 717 baseline-masked
  subjects)**. This is the real N used for download/conversion — reported
  here instead of the page-claimed "masks for all patients" figure, matching
  this project's standing rule (see AMBL's own 632-vs-99 correction) to
  directly count rather than trust collection-page language.
  - 443 patients: "VOLSER: uni-lateral cropped: original DCE" (crop to the
    affected breast).
  - 118 patients: "VOLSER: bi-lateral: original DCE" (no unilateral crop was
    computed for these cases/sites; the bilateral VOLSER variant was used
    instead — spot-checked one bilateral case's SEG object and confirmed
    identical `SegmentAlgorithmType=SEMIAUTOMATIC`/`SegmentationType=
    FRACTIONAL OCCUPANCY`/inverse-mask semantics as the unilateral variant,
    same VOLSER pipeline just without the single-breast crop; tumor volume
    5.2mL on that sample, sane).
  - The excluded 156 (of the 717 baseline-masked) patients either lack any
    VOLSER DCE series recognized here, lack a genuine T2-weighted series, or
    both — not downloaded, per this project's "only pull what's genuinely
    usable" rule.

## Modality / sequence identification — verified via DICOM headers, not names

- **T1wce**: temporal position 1 (first post-contrast, of the VOLSER
  pipeline's resampled 0-6 seven-position DCE stack) of the "VOLSER:
  uni-lateral cropped: original DCE" / "VOLSER: bi-lateral: original DCE"
  series. All 7 temporal positions of this DERIVED/resampled series share
  one identical voxel grid — confirmed by checking the SEG object's
  `SourceImageSequence` `ReferencedSOPInstanceUID`s against this series'
  instances (matched 64/64 on the onboarding sample; they reference temporal
  position 0 specifically, but the shared grid means the mask applies
  unchanged to position 1 too — no registration needed for the T1wce side,
  unlike AMBL).
- **T2**: naming is **not uniform across sites** (confirmed directly, not
  assumed) — dozens of distinct SeriesDescription variants observed, e.g.
  "AX T2 FSE BILATERAL PRE" (GE), "T2W_SPAIR ACRIN" (Philips, fat-sat
  SPAIR), "T2fseidealarc_BP"/"WATER: Ax T2 FSE-IDEAL" (GE IDEAL water-fat
  separated), "T2 TIRM AX"/"Ax T2 STIR ASSET"/"STIR AXIAL" (inversion-
  recovery T2-ish fat-sat variants), etc. Selection went through **two
  correction passes**, each caught by noticing an implausible result rather
  than trusting the first heuristic:
  1. **First pass** (`T2|FSE|SPAIR` substring match, no word boundaries):
     mis-selected a genuinely **T1** fat-sat post-contrast series
     ("t1_fl3d_tra_SPAIR_POST" / "t1_fl3d_tra_vibe spair post", 29
     patients) as "T2" purely because "SPAIR" appeared in the name (these
     names literally also contain "t1"/"vibe"/"fl3d", all T1 gradient-echo
     markers, which the unbounded regex never checked). The same pass also
     had an un-anchored "T2" match firing inside unrelated substrings (e.g.
     "post2" contains the literal characters "t2"). Fixed with a
     boundary-aware, T1-excluding regex (require a T2-indicative token —
     t2/t2w/fse/tirm/stir/spair — with a word-ish boundary, AND exclude any
     T1-indicative token — t1/t1w/vibe/fl3d/vibrant).
  2. **Second pass**, after the T1 fix, widened T2 candidacy to STIR/TIRM
     (legitimate T2-ish fat-sat sequences, +129 more eligible patients) but
     surfaced two further, more subtle mis-selections on GE's 4-way IDEAL
     (Dixon-style) water/fat/in-phase/field-map reconstruction, both caught
     by manually inspecting the actual candidate list rather than trusting
     "largest ImageCount wins": (a) **6 patients** had a
     `FieldMap:...T2FSE+ASSET` series selected — a field map is a phase/
     frequency calibration image, not an anatomical image at all; (b) **9
     patients** had a `FAT:...T2FSE+ASSET` series selected on an ImageCount
     tie — the fat-only reconstruction, which is close to the *inverse* of
     a usable diagnostic image for tumor visualization (tumor/glandular
     tissue signal is suppressed, not fat). Fixed with an explicit priority
     order: prefer any WATER-labeled reconstruction first, then any other
     non-FAT/non-FieldMap candidate, then FAT only as an unused last
     resort (0 patients ended up needing it). All 561 final selections were
     re-verified to contain no fieldmap/FAT-only choice.
  `ImageType`/slice-count is re-verified again at conversion time
  (`../1_BIDS_ispy2/breast-ispy2/code/00_00_convert_patient.py`) as an
  independent third check, refusing to proceed if the selected series turns
  out to be a PROJECTION/localizer or has too few slices.

## Segmentation format — bit-encoded, INVERSE polarity (critical finding)

**DICOM-SEG**, `SeriesDescription="ISPY2: VOLSER: [uni-lateral cropped|
bi-lateral]: Analysis Mask"`, `SegmentAlgorithmType=SEMIAUTOMATIC`,
`SegmentAlgorithmName="Background Threshold, PE threshold and connectivity
filter"`, `SegmentationType=FRACTIONAL`/`SegmentationFractionalType=
OCCUPANCY` (an 8-bit container, `MaximumFractionalValue=255`, repurposed to
carry bit-encoded flags rather than a literal occupancy fraction). This is
the VOLSER pipeline's Functional Tumor Volume (FTV) analysis mask — a
**semi-automated threshold+connectivity-filter output, not a radiologist
freehand contour** (unlike AMBL's manually delineated "Mass" ROI) — flagged
clearly here, not silently presented as an equivalent annotation type.

**Verified directly against TCIA's own documentation, not assumed from the
object's name** ("Analysis mask files description",
https://wiki.cancerimagingarchive.net/download/attachments/50135447/Analysis%20mask%20files%20description.v20211020.docx):
*"The masks are INVERSE masks, in that a mask value of 0 indicates that a
voxel was included in the measured FTV."* A naive nonzero-as-tumor read (the
natural first guess from the SeriesDescription "Analysis Mask") was tried
first on the onboarding sample and produced a mask covering **99.7% of the
entire cropped image volume on every single slice, including the crop's own
edge slices** — physically impossible for a discrete breast tumor, and
exactly the class of bug this project's onboarding process is designed to
catch before it reaches training/eval data (see CLAUDE.md's AMBL
632-vs-99 precedent). `value==0` gives physiologically sane tumor volumes:
8.5mL and 24.0mL on the two Philips/GE onboarding samples, 5.2mL on a
bi-lateral-variant sample — all consistent with the trial's own >=2.5cm-
tumor-diameter eligibility criterion (an 8.2mL sphere is exactly 2.5cm
across).

Documented bit values for the nonzero (excluded) codes, per the same source
(written primarily for the older I-SPY1 2D mask format — the I-SPY2 3D SEG
values observed here also include an undocumented bit 16, not resolved
further since only value==0 matters for extraction): 1=PE Threshold,
2=MNC/connectivity filter, 8=Background mask, 32=Manual VOI, 64=OMIT
regions.

## Protocol consistency — real inter-site variation, not per-patient chaos

Checked directly via DICOM headers, following the same methodology used to
catch ATLAS-Liver-HCC's uncontrolled per-patient contrast-phase mix (which
led to that dataset's exclusion — see CLAUDE.md).

- **22 distinct trial sites** confirmed via the `PatientName` field
  (`ISPY2-XXXXXX^Site-YYY`), matching the collection's documented "22+
  sites" — site population ranges from 1 to 100 patients; the top 12 sites
  cover ~89% of the 719-patient collection.
- **Raw (pre-VOLSER) DCE series naming/structure differs by site/vendor** —
  e.g. one Philips 1.5T site (AAA) stores all 7 temporal positions as ONE
  combined "ACRIN DYN" series; one GE 3T site (AAN) stores each temporal
  position as a SEPARATE series ("Ph1/Ax dyn mp 326 2.4 62 NO DELAY" ...
  "Ph5/..."). T2 sequence choice differs too (SPAIR fat-sat vs conventional
  FSE vs STIR/TIRM vs IDEAL water-fat separation) — see above. **This is
  expected, documented multi-vendor implementation of one trial protocol,
  not the kind of undocumented per-patient chaos that sank ATLAS-Liver-HCC.**
  Reassuringly, the VOLSER-derived series actually used by this pipeline
  ARE uniformly named ("VOLSER: uni-lateral cropped: ..." / "VOLSER:
  bi-lateral: ...") across every site checked (Philips and GE alike), which
  is why the pipeline keys off those, not the raw per-vendor series.
- **Pre-contrast -> first-post-contrast timing, measured directly from
  `AcquisitionTime` on the ORIGINAL (non-cropped) per-site DCE series** (the
  VOLSER-derived "uni-lateral cropped"/"bi-lateral" series' own
  `AcquisitionTime`/`ContentTime` fields are frozen/near-identical across all
  7 temporal positions — a reconstruction-pipeline artifact, verified
  directly and NOT usable for this check — e.g. `ContentTime` differences
  between positions were only 2-7 SECONDS, consistent with processing-
  pipeline write timestamps, not real ~100s contrast-timing intervals):
  - Site AAA (Philips 1.5T, "ACRIN DYN" combined series): phase1(pre)
    07:50:31.50 -> phase2(first-post) 07:52:18.07 = **107s**.
  - Site AAN (GE 3T, separate Ph1/Ph2 series): Ph1 07:59:10 -> Ph2 08:00:47
    = **97s**.
  Both in the same order of magnitude, consistent with the trial's
  documented "~1.5-2min early post-contrast" design intent and with the
  task's own prior reference point (~145-442s across the wider MAMA-MIA-
  documented cohort family — our two spot-checks fall somewhat below that
  range's low end, plausibly because MAMA-MIA's figure spans a different/
  wider phase-pairing convention across its whole multi-collection family;
  not investigated further as out of scope for this onboarding).
  **Only 2 of 12 originally-sampled sites were checked this way** (cost:
  each check requires a ~72-215MB per-phase raw-series download, vs a few
  KB for the VOLSER-derived series' own — unusable — time fields) — this is
  real evidence of same-order-of-magnitude timing across 2 different
  vendors/field-strengths (1.5T Philips, 3T GE), not an exhaustive per-site
  sweep. No sign of anything resembling ATLAS's 33-arterial/10-portal/
  8-delayed/7-unknown/2-none per-patient chaos was found in the (limited)
  sample checked. A fuller multi-site timing sweep is a reasonable follow-up
  if this ever becomes load-bearing for a causal claim (the way it did for
  ATLAS), but was not pursued further here given per-sample download cost
  and the strong qualitative evidence already gathered from vendor/protocol
  naming heterogeneity itself (which independently rules out a single
  undocumented per-patient-arbitrary protocol, since the heterogeneity maps
  cleanly onto site/vendor, not onto individual patients within a site).

## Clinical data

`ISPY2-Imaging-Cohort-1-Clinical-Data.xlsx` (downloaded directly from the
TCIA collection page) — 985 subjects (719 ISPY2 + 266 companion ACRIN-6698,
combined cohort spreadsheet) with age at screening, trial arm, HR/HER2
receptor status, pCR outcome, race, menopausal status, ethnicity. Used to
build `../1_BIDS_ispy2/breast-ispy2/participants.tsv` (only the ISPY2
subjects actually converted here get a row).

## Download method

TCIA/NBIA REST API (`services.cancerimagingarchive.net/nbia-api`), OAuth2
password grant with the public `nbia_guest` account (no credentials needed
for CC BY data), `getImage?SeriesInstanceUID=...` per series — same pattern
as AMBL. Run from the Vulcan login node (compute nodes have no internet),
staged to `$SCRATCH/paulh/ispy2_staging/` per the many-small-files rule,
backgrounded (`nohup`), single-threaded (concurrent workers were tried for
AMBL/CBIS-DDSM earlier this session and TCIA's guest API rejected the added
concurrency outright — not repeated here).

## Files in this directory

- `ispy2_seg_series.json` — full raw response of
  `getSeries?Collection=ISPY2&Modality=SEG` (2688 series records), the
  source of truth for the license/N claims above.
- `build_manifest.py` — the (3rd-iteration, corrected) script that queried
  each of the 717 baseline-masked patients' T0 study for a usable DCE + T2
  series pair; kept for provenance/reproducibility.
- `ispy2_final_manifest.csv` — the final 561 eligible patients actually
  downloaded, with their DCE/T2/SEG SeriesInstanceUIDs and which series
  variant was used for each.
- `ISPY2-Imaging-Cohort-1-Clinical-Data.xlsx` — clinical spreadsheet.
- `dicom/<PatientID>/` (populated by the download step, not committed to
  git — many-small-files rule) — per-patient raw DICOM (VOLSER-cropped-or-
  bilateral DCE + chosen T2 + SEG only, the 3 series actually used; the full
  719-patient/multi-series-per-timepoint collection was NOT bulk-downloaded).
