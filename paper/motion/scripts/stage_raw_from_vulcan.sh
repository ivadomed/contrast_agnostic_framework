#!/bin/bash
# Run on VULCAN. TamIA's scratch holds only preprocessed copies of BraTS / CHAOS / I-SPY2, so the raw
# cases used by extract_assets.py are copied from Vulcan into TamIA's palette_motion/raw_stage/,
# mirroring the TamIA-relative roots in extract_assets.py's CASES (re-run if TamIA scratch was purged).
set -euo pipefail
T="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/benchmark/02_tasks"
ST=$SCRATCH/palette_motion_stage
rm -rf "$ST"; mkdir -p "$ST"
cp_case(){ local src=$1 dst=$2 ds=$3 c=$4
  mkdir -p "$ST/$dst/$ds/imagesTr" "$ST/$dst/$ds/labelsTr"
  cp "$T/$src/$ds/imagesTr/${c}_0000.nii.gz" "$ST/$dst/$ds/imagesTr/"
  cp "$T/$src/$ds/labelsTr/${c}.nii.gz" "$ST/$dst/$ds/labelsTr/"; }
for ds in Dataset051_BraTS2024GliomaT1n Dataset052_BraTS2024GliomaT2w Dataset053_BraTS2024GliomaT2f Dataset054_BraTS2024GliomaT1c; do
  cp_case brain_tumor/brats2024-glioma/2_nnUNet_brats2024-glioma/raw brats2024-glioma/2_nnUNet/raw $ds BraTSGLI00005100; done
for ds in Dataset060_CHAOS_MR_T1in Dataset061_CHAOS_MR_T2spir; do
  cp_case abdomen_healthy/chaos/2_nnUNet_chaos/raw chaos/2_nnUNet_chaos/raw $ds MR01; done
for ds in Dataset100_ISPY2T1wce Dataset101_ISPY2T2w; do
  cp_case breast_cancer/ispy2/2_nnUNet_ispy2/raw ispy2/2_nnUNet/raw $ds ispy2_104384_uni; done
tar cf - -C "$ST" . | ssh tamia.alliancecan.ca 'mkdir -p /scratch/${USER:0:1}/${USER}/palette_motion/raw_stage && cd /scratch/${USER:0:1}/${USER}/palette_motion/raw_stage && tar xf - && find . -name "*.nii.gz" | wc -l'
