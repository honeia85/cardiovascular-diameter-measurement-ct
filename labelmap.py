"""
TotalSegmentator (heartchambers_highres) individual masks -> merged label volume.

The measure modules expect a single volume with this label scheme:
    1 = PA  (PT/RPA/LPA as one blob)
    2 = AA
    3 = LA, 4 = LV, 5 = RA, 6 = RV
heart_myocardium is not measured and is dropped to background.

On overlapping voxels the later class in `ordered` wins (overwrite). This fixed
deterministic composition order was used for the reported analysis; it was not selected
on evaluation performance.
"""
from pathlib import Path

import numpy as np
import nibabel as nib

# Label scheme expected by the measure modules
MEASURE_CLASS = {"PA": 1, "AA": 2, "LA": 3, "LV": 4, "RA": 5, "RV": 6}
# Explicit labels passed to measure_chamber.run (instead of inferring with np.unique)
CHAMBER_CD = {"LA": 3, "LV": 4, "RA": 5, "RV": 6}


def load_bin_mask(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    data = img.get_fdata()
    # Binary masks are usually 0/1 or 0/255, so threshold at > 0
    return (data > 0)


def build_multiclass_labelmap(folder: str, out_path: str):
    folder = Path(folder)

    ordered = [
        ("pulmonary_artery.nii.gz", 1),
        ("aorta.nii.gz", 2),
        ("heart_atrium_left.nii.gz", 3),
        ("heart_ventricle_left.nii.gz", 4),
        ("heart_atrium_right.nii.gz", 5),
        ("heart_ventricle_right.nii.gz", 6),
    ]

    # Use the first file as the affine/header reference.
    ref_path = folder / ordered[0][0]
    if not ref_path.exists():
        raise FileNotFoundError(f"missing: {ref_path}")
    ref_img = nib.load(str(ref_path))
    if len(ref_img.shape) != 3:
        raise ValueError(f"input masks must be 3D; got {ref_img.shape} for {ref_path}")
    if not np.all(np.isfinite(ref_img.affine)):
        raise ValueError(f"mask affine contains non-finite values: {ref_path}")
    ref_codes = tuple(nib.aff2axcodes(ref_img.affine))
    if ref_codes != ("L", "P", "S"):
        raise ValueError(f"input masks must use LPS+ voxel orientation; got {ref_codes}")
    ref_linear = np.asarray(ref_img.affine[:3, :3], dtype=float)
    ref_column_norms = np.linalg.norm(ref_linear, axis=0)
    if not np.all(np.isfinite(ref_column_norms)) or np.any(ref_column_norms <= 0):
        raise ValueError(f"mask affine has invalid spatial axes: {ref_path}")
    ref_directions = ref_linear / ref_column_norms
    if not np.allclose(
        ref_directions, np.diag([-1.0, -1.0, 1.0]), rtol=0.0, atol=1e-5
    ):
        raise ValueError(
            f"input masks must be axis-aligned LPS+; oblique/sheared input: {ref_path}"
        )
    ref_spacing = np.asarray(ref_img.header.get_zooms()[:3], dtype=float)
    if not np.all(np.isfinite(ref_spacing)) or np.any(ref_spacing <= 0):
        raise ValueError(f"invalid voxel spacing for {ref_path}: {ref_spacing.tolist()}")
    if not np.isclose(ref_spacing[0], ref_spacing[1], rtol=1e-5, atol=1e-6):
        raise ValueError(
            f"in-plane spacing must be isotropic; got "
            f"{ref_spacing[0]:.9g} x {ref_spacing[1]:.9g} mm"
        )
    label = np.zeros(ref_img.shape, dtype=np.uint8)

    for fname, cls in ordered:
        mpath = folder / fname
        if not mpath.exists():
            raise FileNotFoundError(f"missing: {mpath}")

        image = nib.load(str(mpath))
        if len(image.shape) != 3:
            raise ValueError(f"input mask must be 3D; got {image.shape}: {mpath}")
        if not np.all(np.isfinite(image.affine)):
            raise ValueError(f"mask affine contains non-finite values: {mpath}")
        if tuple(nib.aff2axcodes(image.affine)) != ("L", "P", "S"):
            raise ValueError(f"mask must use LPS+ voxel orientation: {mpath}")
        linear = np.asarray(image.affine[:3, :3], dtype=float)
        column_norms = np.linalg.norm(linear, axis=0)
        if not np.all(np.isfinite(column_norms)) or np.any(column_norms <= 0):
            raise ValueError(f"mask affine has invalid spatial axes: {mpath}")
        directions = linear / column_norms
        if not np.allclose(
            directions, np.diag([-1.0, -1.0, 1.0]), rtol=0.0, atol=1e-5
        ):
            raise ValueError(f"mask must be axis-aligned LPS+: {mpath}")
        if image.shape != ref_img.shape:
            raise ValueError(
                f"mask shape does not match reference: {image.shape} vs {ref_img.shape}: {mpath}"
            )
        if not np.allclose(image.affine, ref_img.affine, rtol=0.0, atol=1e-5):
            raise ValueError(f"mask affine does not match reference: {mpath}")
        spacing = np.asarray(image.header.get_zooms()[:3], dtype=float)
        if not np.allclose(spacing, ref_spacing, rtol=1e-5, atol=1e-6):
            raise ValueError(f"mask voxel spacing does not match reference: {mpath}")
        mask_data = image.get_fdata()
        if not np.all(np.isfinite(mask_data)):
            raise ValueError(f"mask contains non-finite values: {mpath}")
        mask = mask_data > 0

        # Overlap handling: the class applied later wins (overwrite)
        label[mask] = cls

    out_img = nib.Nifti1Image(label, ref_img.affine, ref_img.header)
    # Set dtype explicitly since this is a label map
    out_img.set_data_dtype(np.uint8)

    nib.save(out_img, out_path)

    return {name: int((label == cls).sum()) for name, cls in MEASURE_CLASS.items()}
