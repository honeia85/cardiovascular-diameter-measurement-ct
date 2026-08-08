#!/usr/bin/env python3
"""Reorient a NIfTI volume to LPS+ voxel order without resampling.

Usage:
    python reorient_nifti_lps.py INPUT OUTPUT

Only axis permutation and flips are applied. The affine is updated so that voxel
values retain their original physical coordinates. Oblique/sheared inputs and
anisotropic in-plane output are rejected because this helper does not resample
and the measurement pipeline requires equal x/y spacing.
"""

import argparse
import itertools
import sys
from pathlib import Path

if sys.version_info[:2] != (3, 12):
    raise RuntimeError("This release requires Python 3.12.x")

import nibabel as nib
import numpy as np


TARGET_AXCODES = ("L", "P", "S")


def _world_corners(shape, affine):
    """Return spatial corner coordinates in a deterministic row order."""
    corners = np.asarray(
        list(itertools.product(*((0, int(size) - 1) for size in shape[:3]))),
        dtype=float,
    )
    world = nib.affines.apply_affine(affine, corners)
    order = np.lexsort((world[:, 2], world[:, 1], world[:, 0]))
    return world[order]


def _validate_lps_output(image, label):
    codes = tuple(nib.aff2axcodes(image.affine))
    if codes != TARGET_AXCODES:
        raise ValueError(f"{label} is not LPS+ after reorientation; got {codes}")
    if not np.all(np.isfinite(image.affine)):
        raise ValueError(f"{label} affine contains non-finite values")
    linear = np.asarray(image.affine[:3, :3], dtype=float)
    column_norms = np.linalg.norm(linear, axis=0)
    if not np.all(np.isfinite(column_norms)) or np.any(column_norms <= 0):
        raise ValueError(f"{label} affine has invalid spatial axes")
    directions = linear / column_norms
    if not np.allclose(
        directions, np.diag([-1.0, -1.0, 1.0]), rtol=0.0, atol=1e-5
    ):
        raise ValueError(
            f"{label} remains oblique/sheared; this helper only reorients axes and "
            "does not resample"
        )
    spacing = np.asarray(image.header.get_zooms()[:3], dtype=float)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError(f"{label} has invalid voxel spacing: {spacing.tolist()}")
    if not np.isclose(spacing[0], spacing[1], rtol=1e-5, atol=1e-6):
        raise ValueError(
            f"{label} has anisotropic in-plane spacing "
            f"({spacing[0]:.9g} x {spacing[1]:.9g} mm); reorientation cannot resample it"
        )


def reorient_nifti_lps(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"input NIfTI not found: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("INPUT and OUTPUT must be different paths")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    if not str(output_path).lower().endswith((".nii", ".nii.gz")):
        raise ValueError("OUTPUT must end in .nii or .nii.gz")

    source = nib.load(str(input_path))
    if len(source.shape) != 3:
        raise ValueError(f"input NIfTI must be a 3D volume; got {source.shape}")
    if not np.all(np.isfinite(source.affine)):
        raise ValueError("input affine contains non-finite values")

    source_ornt = nib.orientations.io_orientation(source.affine)
    if source_ornt.shape != (3, 2) or not np.all(np.isfinite(source_ornt)):
        raise ValueError(f"cannot determine a 3D spatial orientation: {source_ornt}")
    target_ornt = nib.orientations.axcodes2ornt(TARGET_AXCODES)
    transform = nib.orientations.ornt_transform(source_ornt, target_ornt)
    data = nib.orientations.apply_orientation(np.asanyarray(source.dataobj), transform)
    affine = source.affine @ nib.orientations.inv_ornt_aff(transform, source.shape[:3])

    output = nib.Nifti1Image(data, affine, header=source.header.copy())
    qform_code = int(source.header["qform_code"]) or 1
    sform_code = int(source.header["sform_code"]) or 1
    output.set_qform(affine, code=qform_code)
    output.set_sform(affine, code=sform_code)
    _validate_lps_output(output, "output")

    if not np.allclose(
        _world_corners(source.shape, source.affine),
        _world_corners(output.shape, output.affine),
        rtol=0.0,
        atol=1e-5,
    ):
        raise RuntimeError("physical corner coordinates changed during reorientation")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output, str(output_path))

    saved = nib.load(str(output_path))
    _validate_lps_output(saved, "saved output")
    qform, saved_qform_code = saved.get_qform(coded=True)
    sform, saved_sform_code = saved.get_sform(coded=True)
    if int(saved_qform_code) == 0 or int(saved_sform_code) == 0:
        raise RuntimeError("saved output must have nonzero qform and sform codes")
    if not np.allclose(qform, saved.affine, rtol=0.0, atol=1e-4):
        raise RuntimeError("saved qform does not match the selected affine")
    if not np.allclose(sform, saved.affine, rtol=0.0, atol=1e-5):
        raise RuntimeError("saved sform does not match the selected affine")
    if not np.allclose(
        _world_corners(source.shape, source.affine),
        _world_corners(saved.shape, saved.affine),
        rtol=0.0,
        atol=1e-4,
    ):
        raise RuntimeError("saved output does not preserve physical corner coordinates")
    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Reorient a NIfTI image to LPS+ voxel order without resampling"
    )
    parser.add_argument("input", help="source .nii or .nii.gz")
    parser.add_argument("output", help="new LPS+ .nii or .nii.gz")
    args = parser.parse_args()
    result = reorient_nifti_lps(args.input, args.output)
    spacing = tuple(float(value) for value in result.header.get_zooms()[:3])
    print(
        f"saved {args.output} | axis={tuple(nib.aff2axcodes(result.affine))} "
        f"| shape={result.shape} | spacing_mm={spacing}"
    )


if __name__ == "__main__":
    main()
