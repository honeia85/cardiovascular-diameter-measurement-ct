# Cardiovascular Diameter Auto-Measurement Pipeline

Takes an axis-aligned LPS+ chest CT (NIfTI) and produces a JSON file with the diameters of 8 structures
plus the derived `PT_AA_ratio` and `RV_LV_ratio`.

Measured structures: `PT`, `RPA`, `LPA`, `AA`, `LA`, `LV`, `RA`, `RV`

**Guided Colab reproduction:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/honeia85/cardiovascular-diameter-measurement-ct/blob/v1.0.4/reproduce_colab.ipynb)
&nbsp;— `reproduce_colab.ipynb` runs the frozen pipeline end-to-end on a public, license-clean sample CT and
verifies the model-weight checksums. It uses **your own** TotalSegmentator academic license. The
`totalseg_set_license` command writes that key to `~/.totalsegmentator/config.json` inside the ephemeral
Colab runtime; it is not committed to this repository or included in pipeline outputs, and deleting the
runtime removes it. This repository does not redistribute weights or data.

This is the measurement-pipeline code accompanying the manuscript:

> **Rule-Based Automated Measurement of Eight Cardiovascular Diameters from Pretrained
> Deep-Learning Segmentation on Non-ECG-Gated Contrast-Enhanced Chest CT: A Retrospective
> Reader-Referenced Agreement Study.** (BMC Medical Imaging, under revision)

Public study release: **v1.0.4** — [Zenodo DOI: 10.5281/zenodo.22761761](https://doi.org/10.5281/zenodo.22761761). This is the release that
produced the deposited measurements; it differs from v1.0.2 (commit 8be2f50) in serialising
segmentation-mask export and in accepting a label map without a CT (see [CHANGELOG.md](CHANGELOG.md)).
Companion evaluation dataset (reader markups, automated measurements, the merged label map of each
examination, run logs, TCIA crosswalk, overlap flag): **v1.0.1**, [Zenodo DOI: 10.5281/zenodo.22842865](https://doi.org/10.5281/zenodo.22842865). See [Citation](#citation).

Dataset v1.0.1 corrects documentation only. Measurements, markups, label maps, analysis scripts,
and numerical results are unchanged from v1.0. It clarifies that the archive contains 48
pre-correction coordinate files from six examinations and adds `SUPPLEMENT_TABLE_MAP.md`
for the final 14-table supplement. The archived v1.0.4 source README predates this correction;
use the current dataset README for public-data fields and reproduction commands. The
measurement code and v1.0.4 tag are unchanged.

The repository contains **code only**. It contains no imaging data, no derived measurement
tables, and no model weights. See [Reproducibility](#reproducibility) for the pinned
environment and model-weight checksums, and [Licenses and data access](#licenses-and-data-access)
for what you must obtain separately. See [SECURITY.md](SECURITY.md) before running the
pipeline on any data.

> **Development and evaluation separation — with a known overlap.** All rule parameters in this
> repository were fixed using the LungCT-Diagnosis development cohort and the same implementation was
> applied without tuning to the reader-referenced evaluation cohort. **27 of the 118 assembled evaluation
> examinations (all from QIN LUNG CT) are the same CT series as development examinations** (26 identical
> volumes and one partial copy): LungCT-Diagnosis and QIN LUNG CT both originate from the H. Lee Moffitt
> Cancer Center and were de-identified separately by TCIA, so identifiers do not reveal the overlap. The
> overlap was established by pixel-level comparison; the primary analysis of the manuscript therefore uses
> the 91 examinations without overlap, with the assembled 118 and the MIDRC-RICORD-only 76 as sensitivity
> sets. The affected examinations are flagged in the companion dataset
> (`crosswalk_118_pixel_verified.csv`, column `dev_overlap_match_type`). Development PatientID
> values are omitted from the public crosswalk. See
> [Evaluation data and known development–evaluation overlap](#evaluation-data-and-known-developmentevaluation-overlap).

## Pipeline

```
CT (.nii/.nii.gz)
  → TotalSegmentator (heartchambers_highres)   individual masks (installed separately)
  → labelmap                                   merged label volume (PA=1, AA=2, LA=3, LV=4, RA=5, RV=6)
  → measure_chamber / measure_vessel           slice selection + diameter measurement
  → <case>_measurements.json                   diameters, LPS endpoints, and ratios
```

There is **no author-trained neural network**. The only deep-learning model is the public,
pretrained TotalSegmentator `heartchambers_highres` model, used without fine-tuning. All of
this study's tuning produced **fixed scalar rule parameters** (e.g. the cardiac slice-selection
thresholds in `measure_chamber.py`, optimized on the development cohort with Optuna). Those
values are hard-coded in the measure modules, so this release reproduces application of the
finalized rules. It does not reproduce parameter derivation. The rule-selection code, fold
assignments, fold-level scores, chamber trial records, and development-cohort R1 measurements
were not retained and are unavailable.

## Setup

Requires **Python 3.12.x**. The study environment used 3.12.12 and the clean Colab smoke test
used 3.12.13; the patch version is not fixed.

```bash
conda create -n med_seg python=3.12 -y
conda activate med_seg

# Exact GPU stack used for the published measurements (CUDA 11.8):
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

# heartchambers_highres is a LICENSED TotalSegmentator task — obtain your own license key
# from the TotalSegmentator authors and register it:
totalseg_set_license -l <YOUR_LICENSE>
```

The commands above reproduce the study's CUDA 11.8 environment. The Colab notebook instead checks for
the separately validated Colab-compatible torch 2.5.1 CUDA build (the clean T4 smoke test used
`torch 2.5.1+cu124`); that mechanics demonstration is not a claim of bit-identical CUDA inference.
`totalseg_set_license` persists the supplied key in `~/.totalsegmentator/config.json`. On a local
workstation, protect and remove that file according to your credential policy; in Colab, delete the
ephemeral runtime when finished.

If `TotalSegmentator` is not on `PATH`, point to it explicitly with the `TOTALSEG_EXE`
environment variable. Segmentation and measurement may live in separate environments — the
pipeline invokes TotalSegmentator as a subprocess.

## Usage

```bash
# Convert DICOM externally, then make the NIfTI axis-aligned LPS+ orientation explicit.
dcm2niix -z y -o ./converted -f ct_raw /path/to/dicom
python reorient_nifti_lps.py ./converted/ct_raw.nii.gz ./ct_lps.nii.gz

python run_pipeline.py -i ct_lps.nii.gz -o ./out        # single case
python run_pipeline.py -i ./ct_folder -o ./out          # batch over a folder
python run_pipeline.py -i ct_lps.nii.gz -o ./out -d cpu # CPU, same high-res task (very slow)

# Skip segmentation if a merged label volume already exists
python run_pipeline.py -i ct_lps.nii.gz -o ./out --seg seg_lps.nii.gz

# Measure a label map with no CT present: no GPU, no TotalSegmentator, measurement dependencies only
python run_pipeline.py --seg case_T_031_seg.nii.gz -o ./out
```

The last form is how the companion dataset's deposited label maps regenerate the published diameters:
the measurement modules read only the label map and its NIfTI header. Over the 118 deposited label maps
it returns all 944 diameters identical to the deposited values at four decimal places (Linux, central
processing unit, Python 3.12 with the pinned measurement dependencies, 30-70 s per examination).

The logged v1.0.4 evaluation rerun had a median runtime of 73.7 s per examination.
CPU mode does not add TotalSegmentator's `--fast` flag because that mode is incompatible with
`heartchambers_highres`; it therefore runs the same task but is substantially slower than GPU mode.

## Input requirements and a known limitation

- Input is a 3D chest CT in **NIfTI** (`.nii`/`.nii.gz`) with **axis-aligned LPS+** voxel
  orientation. Convert from DICOM beforehand, then run `reorient_nifti_lps.py INPUT OUTPUT`;
  the helper performs only axis permutation/flipping and does not resample voxel values.
  It therefore cannot correct an oblique or sheared acquisition: resample such an image
  separately to an axis-aligned LPS+ grid before running the pipeline.
- The pipeline rejects a CT or supplied merged segmentation that is not axis-aligned LPS+,
  contains a non-finite affine, or does not match the other volume's first-three-axis shape
  and affine (absolute tolerance `1e-5`). It does not silently repair geometry.
- **In-plane pixel spacing must be isotropic (spacing[0] == spacing[1]).** This code converts
  in-plane pixel lengths to millimeters using the first spacing element only. Every examination
  in the study had equal x- and y-axis spacing, so this did not affect the published results,
  but on an **anisotropic in-plane** input the reported diameters would be wrong. Check your
  header spacing before trusting any output on new data.
- `qc_low_voxel` in the output lists structures whose segmentation has fewer than 100 voxels —
  do not trust measurements for those. If one structure fails, the rest still run and the failed
  one is recorded under `missing`.

## Output

```
out/<case>/
├── <case>_measurements.json     ← diameters (mm), LPS endpoints, PT/AA and RV/LV ratios
├── PT.mrk.json  RPA.mrk.json …  ← 3D Slicer markups (per structure)
├── <case>_seg.nii.gz            ← merged label volume
└── totalseg/                    ← raw TotalSegmentator masks (for debugging)
```

Run artifacts (logs, `pipeline_out/`) embed local input/output paths and are excluded by
`.gitignore`; do not commit them.

The top-level JSON field `measurements` contains each available structure's positive finite
`diameter_mm`, two three-coordinate LPS endpoints, and markup filename. `ratios` contains
`PT_AA_ratio` and `RV_LV_ratio`, rounded to six decimals; a ratio is JSON `null` if either
component measurement is unavailable. Invalid or absent structures are listed under `missing`.

## Layout

| File | Role |
|---|---|
| `run_pipeline.py` | orchestrator (entry point) |
| `reorient_nifti_lps.py` | lossless axis permutation/flipping toward axis-aligned LPS+; no oblique resampling |
| `labelmap.py` | individual masks → merged label volume |
| `measure_chamber.py` | LA/LV/RA/RV measurement |
| `measure_vessel.py` | PT/RPA/LPA/AA measurement |
| `utils.py` | shared geometry/IO helpers |
| `verify_weights.py` | check the installed model weights against the published ones |
| `SECURITY.md` | research-use security and dependency boundary |

The measure modules expose the finalized study parameters directly in code and document that
they were fixed in the development cohort and used unchanged in evaluation. They reproduce
application of the frozen rules, not parameter derivation.

## Reproducibility

Environment used to produce the published results:

| Item | Value |
|---|---|
| Python | 3.12 |
| Measurement | numpy 2.2.6, scipy 1.16.2, scikit-learn 1.7.2, scikit-image 0.25.2, pandas 2.3.3, nibabel 5.3.2, opencv-python 4.12.0.88 |
| Segmentation | TotalSegmentator 2.12.0, nnunetv2 2.6.4, torch 2.5.1+cu118 |
| GPU | NVIDIA RTX 4090, CUDA 11.8 |
| Task | `heartchambers_highres` (task_id 301), 3d_fullres, fold 0, nnUNetTrainer |

Version pins in `requirements.txt` are exact (`==`) for this reason. Relax them to `>=` only if
reproducing the published numbers is not a requirement.

### Model weights

Two weight sets are required. `heartchambers_highres` first runs the total (3 mm robust) model
to crop to the heart before the high-res pass. Both download automatically on first run into
`~/.totalsegmentator/nnunet/results/` and are **not redistributed here** (they are covered by
the TotalSegmentator license). SHA256 of the checkpoints used for the published results:

| Model | File | SHA256 |
|---|---|---|
| `Dataset301_heart_highres_1559subj`<br>`nnUNetTrainer__nnUNetPlans__3d_fullres` | `fold_0/checkpoint_final.pth` | `ca616ae0e9e2f4d7e8b01bc8fca00e430c4392d6a449ccdc2f9dfc4bc03f62eb` |
| | `plans.json` | `3359f3da44ee822bb49eb72b1248c531774ac6227a35d90055ab4533a18efb32` |
| `Dataset297_TotalSegmentator_total_3mm_1559subj`<br>`nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres` | `fold_0/checkpoint_final.pth` | `1e38e40356adc2706a662e365405a97f862d62a7da65f6bf81d025aee1b979ac` |
| | `plans.json` | `d1cb3c15f53dc36fdb618e0f9082573d1f50b74a9b5fa73094ac8680845839f1` |

Run `python verify_weights.py` after the first pipeline run to confirm your weights match.
`plans.json` is included in the check because it fixes preprocessing (spacing, patch size,
normalization); identical checkpoints with different plans do not reproduce. TotalSegmentator
inference is not bit-deterministic (repeated runs differ by a few voxels), which has not changed
measured diameters in our checks.

### Reproducing the published statistics

Two levels of reproduction are possible:

1. **Statistical reproduction (no images needed).** The derived evaluation-cohort data — corrected
   reader markups, automated measurements, measurement table, data dictionary, exclusion log, the
   118-row case-to-TCIA-series crosswalk, the development-overlap flag, and scripts for the
   evaluation-cohort agreement analyses — are deposited on Zenodo under CC BY-NC 4.0
   (dataset v1.0.1 DOI: 10.5281/zenodo.22842865). They are not in this repository.
   Development-cohort R1 measurements and separate PE Challenge data are unavailable in the
   archive. Historical source-image audits require CT volumes obtained separately.
2. **End-to-end reproduction (CT → measurements).** Install this pipeline and TotalSegmentator,
   obtain the 118 source series yourself from The Cancer Imaging Archive using the SeriesInstanceUIDs
   in the deposited crosswalk, convert and reorient them, then run `run_pipeline.py` and compare with
   the deposited automated markups.

## Evaluation data and known development–evaluation overlap

The evaluation cohort comprises 118 examinations: MIDRC-RICORD-1A (26), MIDRC-RICORD-1B (50), and
QIN LUNG CT (42). Twenty-seven of the QIN LUNG CT examinations are the same CT series as
LungCT-Diagnosis examinations used for rule development — 26 pixel-identical volumes and one partial
copy — under different TCIA identifiers. The manuscript's primary analysis is therefore restricted to
the 91 examinations without overlap, and reports the assembled 118 and the MIDRC-RICORD-only 76 as
sensitivity sets. Exclude the 27 flagged examinations to reproduce the primary cohort without
identified acquisition overlap. This does not establish patient-level independence or exclude
overlap with nonpublic segmentation-model training data. The public crosswalk retains the match
type, not development identifiers; source-image fingerprint checks require the source CT volumes.

The pipeline runs each examination once, without automatic retry, and logs per-examination success or
failure (`run_pipeline.py`); the batch summary at the end of a run lists successes and failures. Logs of
the original study runs were not retained, so all 118 evaluation examinations were re-run under v1.0.4
with logging: every one returned all eight diameters on the first pass (`run_logs/run_summary.json` of
the companion dataset).

## Licenses and data access

**This repository's code:** MIT — see [LICENSE](LICENSE). The code contains no imaging data and
no derived measurement tables.

You must obtain the following **separately**; none of them are redistributed here:

| Item | How to obtain | License / terms |
|---|---|---|
| TotalSegmentator + `heartchambers_highres` weights | Install `TotalSegmentator`; register a license key (`totalseg_set_license`) | TotalSegmentator's own license (the `heartchambers_highres` task requires a key) |
| MIDRC-RICORD-1a / -1b CT | The Cancer Imaging Archive — doi:10.7937/VTW4-X588, doi:10.7937/31V8-4A40 | CC BY-NC 4.0 (+ TCIA Data Usage Policy) |
| QIN LUNG CT | The Cancer Imaging Archive — doi:10.7937/K9/TCIA.2015.NPGZYZBZ | CC BY 3.0 (+ TCIA Data Usage Policy) |
| LungCT-Diagnosis (development) | The Cancer Imaging Archive — doi:10.7937/K9/TCIA.2015.A6V7JIWX | CC BY 3.0 (+ TCIA Data Usage Policy) |

TotalSegmentator citation: Wasserthal et al., *Radiology: Artificial Intelligence* (2023),
<https://pubs.rsna.org/doi/10.1148/ryai.230024>

## Citation

If you use this code, please cite the archived release and the manuscript; if you use the evaluation
data, please also cite the companion dataset:

> Je J, Shim H, Nam Y, Kim Y, Kim BW, Hong P. Cardiovascular Diameter Auto-Measurement Pipeline
> (v1.0.4) [Software]. Zenodo. DOI: 10.5281/zenodo.22761761.

> Je J, Shim H, Nam Y, Kim Y, Kim BW, Hong P. Evaluation-cohort data for "Rule-Based Automated
> Measurement of Eight Cardiovascular Diameters from Pretrained Deep-Learning Segmentation on
> Non-ECG-Gated Contrast-Enhanced Chest CT" (Version 1.0.1) [Data set]. Zenodo.
> DOI: 10.5281/zenodo.22842865.

A `CITATION.cff` file is included; GitHub renders it under "Cite this repository". Changes between
releases are listed in `CHANGELOG.md`.

## Notes

- The class order in `labelmap.build_multiclass_labelmap` (later class wins on overlap) and the
  parameters in the measure modules are the finalized study settings. Changing them changes the
  implementation and its results.
- `measure_chamber.run` takes labels explicitly via `cd`. Omitting it falls back to inferring
  labels from `np.unique(vol)[3:]`, which is only correct when labels 0–6 are all present, so the
  pipeline always passes `cd`.
