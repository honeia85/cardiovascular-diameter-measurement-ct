# Changelog

All notable changes to this repository are documented here. Measurement code (`labelmap.py`,
`measure_chamber.py`, `measure_vessel.py`, `utils.py`, `reorient_nifti_lps.py`, `verify_weights.py`)
has not changed since v1.0.2. `run_pipeline.py` changed at v1.0.4 in how it invokes
TotalSegmentator and in accepting a label map without a CT; the measurements reported in the
manuscript were produced with v1.0.4.

## v1.0.4 — 2026-09 (segmentation call and CT-free measurement)

- `run_totalseg` now passes `--nr_thr_saving 1` for every examination. With the package default
  (6) TotalSegmentator writes the per-structure masks from a worker pool in which each worker
  loads the full-resolution label volume as float64, so peak host memory scales with the number
  of workers; on the largest examination of the evaluation cohort (512 x 512 x 564 = 147.8 M
  voxels) this raised `MemoryError` during mask export on one host while the same code and
  examination completed on another. TotalSegmentator applies the same serialisation itself above
  512 x 512 x 1000 voxels, so this setting extends upstream's own guard to every input size and
  makes the run independent of host memory, including CPU-only and hosted (Colab) environments.
  The serial path also writes the masks from the array that TotalSegmentator's `remove_outside`
  postprocessing has been applied to, which the worker-pool path does not; the two paths
  therefore differ in the mask, not only in memory use. Re-running all 118 evaluation
  examinations changed 13 of 936 diameters relative to the worker-pool path (all in the
  pulmonary arteries and the ascending aorta; 12 of 13 below 0.5 mm, largest 1.52 mm).
- `-i/--input` is now optional when `--seg` is given, so a deposited label map can be measured
  with no CT, no GPU and no TotalSegmentator installation: `run_pipeline.py --seg
  case_T_031_seg.nii.gz -o ./out`. The measurement rules read only the label map and its NIfTI
  header. The CT/segmentation geometry check is replaced in this mode by the same validation
  applied to the label map alone.
- `<case>_measurements.json` records `pipeline_version`; `input_ct` is `null` in CT-free mode.
- Colab badge and `reproduce_colab.ipynb` re-pinned to v1.0.4; the notebook runs the same end-to-end path on a sample CT.
- First-pass behavior with this release: 118 of 118 evaluation examinations completed on the
  first pass with all eight diameters.

## v1.0.3 — 2026-09-15 (documentation and archiving only; code identical to v1.0.2)

- Archived on Zenodo (DOI recorded in README.md and CITATION.cff).
- README: manuscript title updated to the revised title; disclosure of the development–evaluation
  overlap (27 of the 118 assembled evaluation examinations are the same CT series as LungCT-Diagnosis
  development examinations under different TCIA identifiers: 26 identical volumes and one partial copy)
  with a pointer to the 91-examination primary analysis and the flag in the companion dataset; "Reproducing the published statistics" now points to the deposited
  evaluation dataset (Zenodo) instead of "available on request"; Citation section added.
- Added `CITATION.cff`, `.zenodo.json`, and this `CHANGELOG.md`.
- Colab badge pinned to v1.0.3.

## v1.0.2 — 2026-07

- Release cited in the submitted manuscript (commit 8be2f50ef7fcc72283095689bf08ee4f3c2a60c6).
