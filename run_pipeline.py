"""
End-to-end pipeline: CT NIfTI -> TotalSegmentator -> label merge -> measurement -> JSON

Every step calls the finalized study module, so the measurements match running each
step by hand. The finalized rules were applied uniformly to the evaluation cohort.
    1) TotalSegmentator (heartchambers_highres)   7 individual masks
    2) labelmap.build_multiclass_labelmap         individual masks -> single volume, labels 1-6
    3) measure_chamber.run                        LA/LV/RA/RV
       measure_vessel.measure_artery              PT/RPA/LPA
       measure_vessel.measure_aorta               AA
    4) collect 8 .mrk.json files -> <case>_measurements.json

Environment:
    The measurement dependencies (sklearn/cv2/...) and TotalSegmentator may live in
    separate environments. This script runs in the measurement environment and invokes
    TotalSegmentator as a subprocess. Set the TOTALSEG_EXE environment variable if the
    executable is not on PATH.

Usage:
    python run_pipeline.py -i ct.nii.gz -o ./out
    python run_pipeline.py -i ./ct_folder -o ./out                  # every case in a folder
    python run_pipeline.py -i ct.nii.gz -o ./out --seg seg.nii.gz   # reuse a segmentation
    python run_pipeline.py --seg case_seg.nii.gz -o ./out            # measure a label map without CT
    python run_pipeline.py -i ct.nii.gz -o ./out -d cpu
"""
import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np

from labelmap import build_multiclass_labelmap, MEASURE_CLASS, CHAMBER_CD
from measure_chamber import run as run_chambers
from measure_vessel import measure_artery, measure_aorta

# ======================
# Configuration
# ======================
# Paths differ per deployment, so TOTALSEG_EXE can override. Otherwise look on PATH.
TOTALSEG_EXE = (
    os.environ.get("TOTALSEG_EXE")
    or shutil.which("TotalSegmentator")
    or "TotalSegmentator"
)
TASK = "heartchambers_highres"
PIPELINE_VERSION = "v1.0.4"
DEVICE = "gpu:0"
MIN_VOXELS = 100          # QC threshold: below this, the structure is unusable

# The 8 structures collected into the final JSON
STRUCTURES = ["PT", "RPA", "LPA", "AA", "LA", "LV", "RA", "RV"]

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT,
                    handlers=[logging.StreamHandler()])
log = logging.getLogger("pipeline")


def add_file_log(out_root: Path):
    """Write the run log into the output folder.

    Not the working directory: the log records input CT paths, so keeping it next
    to the results avoids leaving patient paths inside the deployed package.
    """
    fh = logging.FileHandler(out_root / "pipeline.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.getLogger().addHandler(fh)


def case_name_of(path: Path) -> str:
    """Extract the case name from either ct.nii.gz or ct.nii."""
    name = path.name
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return path.stem


def _validate_volume(image, label: str) -> None:
    """Reject a volume that is not a 3-D, axis-aligned LPS+ NIfTI."""
    if len(image.shape) != 3:
        raise ValueError(f"{label} must be a 3D NIfTI volume; got {image.shape}")
    if not np.all(np.isfinite(image.affine)):
        raise ValueError(f"{label} affine contains non-finite values")
    codes = tuple(nib.aff2axcodes(image.affine))
    if codes != ("L", "P", "S"):
        raise ValueError(f"{label} must use LPS+ voxel orientation; got {codes}")
    linear = np.asarray(image.affine[:3, :3], dtype=float)
    column_norms = np.linalg.norm(linear, axis=0)
    if not np.all(np.isfinite(column_norms)) or np.any(column_norms <= 0):
        raise ValueError(f"{label} affine has invalid spatial axes")
    directions = linear / column_norms
    if not np.allclose(
        directions, np.diag([-1.0, -1.0, 1.0]), rtol=0.0, atol=1e-5
    ):
        raise ValueError(
            f"{label} affine must be axis-aligned LPS+; oblique/sheared input is unsupported"
        )
    spacing = np.asarray(image.header.get_zooms()[:3], dtype=float)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError(f"{label} has invalid voxel spacing: {spacing.tolist()}")
    if not np.isclose(spacing[0], spacing[1], rtol=1e-5, atol=1e-6):
        raise ValueError(
            f"{label} requires isotropic in-plane spacing; got "
            f"{spacing[0]:.9g} x {spacing[1]:.9g} mm"
        )


def validate_geometry_single(seg_path: Path) -> None:
    """Reject a supplied label map that is not a valid measurement volume."""
    _validate_volume(nib.load(str(seg_path)), "segmentation")


def validate_geometry_pair(ct_path: Path, seg_path: Path) -> None:
    """Reject CT/segmentation pairs that are not matching LPS+ volumes."""
    ct_img = nib.load(str(ct_path))
    seg_img = nib.load(str(seg_path))
    _validate_volume(ct_img, "CT")
    _validate_volume(seg_img, "segmentation")
    if ct_img.shape[:3] != seg_img.shape[:3]:
        raise ValueError(
            f"CT/segmentation shape mismatch: {ct_img.shape[:3]} vs {seg_img.shape[:3]}"
        )
    if not np.allclose(ct_img.affine, seg_img.affine, rtol=0.0, atol=1e-5):
        raise ValueError("CT/segmentation affine mismatch")


# ──────────────────────────────────────────────────────────────
# 1) TotalSegmentator
# ──────────────────────────────────────────────────────────────
def run_totalseg(ct_path: Path, seg_dir: Path, device: str = DEVICE) -> Path:
    """Write the individual masks (heart_atrium_left.nii.gz etc.) into seg_dir."""
    if not Path(TOTALSEG_EXE).exists() and shutil.which(TOTALSEG_EXE) is None:
        raise RuntimeError(
            f"TotalSegmentator executable not found: {TOTALSEG_EXE}\n"
            f"Set the TOTALSEG_EXE environment variable to its path."
        )
    seg_dir.mkdir(parents=True, exist_ok=True)
    cmd = [TOTALSEG_EXE, "-i", str(ct_path), "-o", str(seg_dir), "-ta", TASK]
    # Serialize mask export to avoid one full-resolution float64 label volume per
    # worker. This is also the setting used for the reported v1.0.4 measurements.
    cmd += ["--nr_thr_saving", "1"]
    if device == "cpu":
        # heartchambers_highres is incompatible with TotalSegmentator --fast.
        cmd += ["-d", "cpu"]
    else:
        cmd += ["-d", device]

    log.info("CMD: " + " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    (seg_dir / "stdout.txt").write_text(p.stdout or "", encoding="utf-8")
    (seg_dir / "stderr.txt").write_text(p.stderr or "", encoding="utf-8")
    if p.returncode != 0:
        raise RuntimeError(
            f"TotalSegmentator failed (exit {p.returncode}). "
            f"stderr tail:\n{(p.stderr or '')[-1500:]}"
        )
    return seg_dir


# ──────────────────────────────────────────────────────────────
# 2) Label merge + QC
# ──────────────────────────────────────────────────────────────
def merge_and_qc(seg_dir: Path, merged_path: Path):
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    counts = build_multiclass_labelmap(folder=str(seg_dir), out_path=str(merged_path))
    weak = [n for n, v in counts.items() if v < MIN_VOXELS]
    for n, v in counts.items():
        log.info(f"  QC {n:>3}: {v} voxels" + ("  [low]" if v < MIN_VOXELS else ""))
    if weak:
        log.warning(f"Too few voxels: {weak} (measurement for these may fail)")
    return counts, weak


# ──────────────────────────────────────────────────────────────
# 3) Measurement
# ──────────────────────────────────────────────────────────────
def clear_stale_markups(out_dir: Path) -> None:
    """Remove prior-run structure markups before a case is remeasured."""
    removed = []
    for name in STRUCTURES:
        path = out_dir / f"{name}.mrk.json"
        if path.exists():
            path.unlink()
            removed.append(path.name)
    if removed:
        log.info(f"  Removed stale markups: {removed}")


def measure_all(merged_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    clear_stale_markups(out_dir)
    seg_str, out_str = str(merged_path), str(out_dir)

    log.info("  [1/3] Chambers (LA/LV/RA/RV)")
    # Pass cd explicitly. The default np.unique(vol)[3:] inference is only correct when
    # labels 0-6 are all present; if any structure is missing the mapping silently
    # shifts and the wrong chamber gets measured.
    run_chambers(seg_str, out_str, cd=dict(CHAMBER_CD))

    log.info("  [2/3] Pulmonary arteries (PT/RPA/LPA)")
    head_idx = measure_artery(seg_str, out_str)

    log.info("  [3/3] Ascending aorta (AA)")
    measure_aorta(seg_str, out_str, head_idx)
    return head_idx


# ──────────────────────────────────────────────────────────────
# 4) Collect results
# ──────────────────────────────────────────────────────────────
def collect_results(out_dir: Path):
    """Gather diameter and both endpoints (LPS) from each structure's .mrk.json."""
    measurements, missing = {}, []
    for name in STRUCTURES:
        f = out_dir / f"{name}.mrk.json"
        if not f.exists():
            missing.append(name)
            continue
        try:
            mk = json.loads(f.read_text(encoding="utf-8"))["markups"][0]
            diameter_mm = round(float(mk["measurements"][0]["value"]), 4)
            p1_lps = np.asarray(mk["controlPoints"][0]["position"], dtype=float)
            p2_lps = np.asarray(mk["controlPoints"][1]["position"], dtype=float)
            if not np.isfinite(diameter_mm) or diameter_mm <= 0:
                raise ValueError(f"diameter must be finite and > 0; got {diameter_mm}")
            if p1_lps.shape != (3,) or p2_lps.shape != (3,):
                raise ValueError("each control-point position must contain three coordinates")
            if not np.all(np.isfinite(p1_lps)) or not np.all(np.isfinite(p2_lps)):
                raise ValueError("control-point position contains non-finite values")
            endpoint_distance = float(np.linalg.norm(p2_lps - p1_lps))
            if not np.isfinite(endpoint_distance) or endpoint_distance <= 0:
                raise ValueError("control points do not define a positive finite line")
            measurements[name] = {
                "diameter_mm": diameter_mm,
                "p1_lps": p1_lps.tolist(),
                "p2_lps": p2_lps.tolist(),
                "markup_file": f.name,
            }
        except Exception as e:
            missing.append(name)
            f.unlink(missing_ok=True)
            log.error(f"  failed to parse {name}.mrk.json: {e}")
    return measurements, missing


def calculate_ratios(measurements):
    """Return the prespecified ratios, or null when a component is unavailable."""
    ratios = {}
    for ratio_name, numerator, denominator in (
        ("PT_AA_ratio", "PT", "AA"),
        ("RV_LV_ratio", "RV", "LV"),
    ):
        if numerator not in measurements or denominator not in measurements:
            ratios[ratio_name] = None
            continue
        value = (
            measurements[numerator]["diameter_mm"]
            / measurements[denominator]["diameter_mm"]
        )
        ratios[ratio_name] = round(float(value), 6) if np.isfinite(value) else None
    return ratios


def process_case(ct_path: Path | None, out_root: Path, seg_override: Path = None,
                 device: str = DEVICE):
    source_path = ct_path if ct_path is not None else seg_override
    if source_path is None:
        raise ValueError("either a CT input or a merged segmentation is required")
    case = case_name_of(source_path)
    if ct_path is None and case.endswith("_seg"):
        case = case[:-4]
    out_dir = out_root / case
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log.info(f"===== {case} =====")

    if seg_override is not None:
        merged_path = seg_override
        counts, weak = {}, []
        log.info(f"Using existing segmentation: {merged_path}")
    else:
        seg_dir = out_dir / "totalseg"
        log.info("Running TotalSegmentator")
        run_totalseg(ct_path, seg_dir, device=device)
        merged_path = out_dir / f"{case}_seg.nii.gz"
        log.info("Merging labels")
        counts, weak = merge_and_qc(seg_dir, merged_path)

    if ct_path is None:
        validate_geometry_single(merged_path)
    else:
        validate_geometry_pair(ct_path, merged_path)

    log.info("Measuring")
    measure_all(merged_path, out_dir)

    measurements, missing = collect_results(out_dir)
    if missing:
        log.warning(f"Missing measurements: {missing}")

    img = nib.load(str(merged_path))
    summary = {
        "case": case,
        "pipeline_version": PIPELINE_VERSION,
        "input_ct": str(ct_path) if ct_path is not None else None,
        "segmentation": str(merged_path),
        "task": TASK,
        "created": datetime.now().isoformat(timespec="seconds"),
        "spacing_mm": [round(float(z), 6) for z in img.header.get_zooms()[:3]],
        "label_map": MEASURE_CLASS,
        "seg_voxel_counts": counts,
        "qc_low_voxel": weak,
        "measurements": measurements,
        "ratios": calculate_ratios(measurements),
        "missing": missing,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out_json = out_dir / f"{case}_measurements.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
                        encoding="utf-8")
    log.info(f"Done: {out_json}  ({len(measurements)}/{len(STRUCTURES)} structures, "
             f"{summary['elapsed_sec']}s)")
    return summary


def find_cts(input_path: Path):
    if input_path.is_file():
        return [input_path]
    cts = sorted(list(input_path.glob("*.nii.gz")) + list(input_path.glob("*.nii")))
    return cts


def main():
    ap = argparse.ArgumentParser(
        description="Measure 8 cardiac structures from a chest CT NIfTI and write JSON")
    ap.add_argument("-i", "--input", required=False, help="CT .nii/.nii.gz file or a folder")
    ap.add_argument("-o", "--output", default="./pipeline_out", help="output folder")
    ap.add_argument("--seg", default=None,
                    help="merged label (1-6) segmentation file; skips TotalSegmentator "
                         "(single case only)")
    ap.add_argument("-d", "--device", default=DEVICE, help="gpu:0 | cpu")
    args = ap.parse_args()

    input_path = Path(args.input) if args.input else None
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    add_file_log(out_root)

    seg_override = Path(args.seg) if args.seg else None
    if input_path is None and seg_override is None:
        log.error("either --input or --seg is required")
        sys.exit(2)

    cts = find_cts(input_path) if input_path is not None else [None]
    if input_path is not None and not cts:
        log.error(f"No CT files found: {input_path}")
        sys.exit(1)

    if seg_override is not None and len(cts) > 1:
        log.error("--seg can only be used with a single case.")
        sys.exit(1)

    log.info(f"{len(cts)} case(s)")
    ok, fail = [], []
    for ct in cts:
        display_path = ct if ct is not None else seg_override
        display_name = case_name_of(display_path)
        if ct is None and display_name.endswith("_seg"):
            display_name = display_name[:-4]
        try:
            process_case(ct, out_root, seg_override=seg_override, device=args.device)
            ok.append(display_name)
        except Exception as e:
            log.error(f"Failed: {display_name}: {e}")
            fail.append({"case": display_name, "error": str(e)})

    log.info(f"All done: {len(ok)} succeeded / {len(fail)} failed")
    if fail:
        log.error(f"Failed cases: {[f['case'] for f in fail]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
