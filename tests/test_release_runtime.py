import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import labelmap
import reorient_nifti_lps as reorient
import run_pipeline
import utils


LPS_AFFINE = np.diag([-1.0, -1.0, 2.0, 1.0])


def save_nifti(path, shape=(4, 5, 6), affine=LPS_AFFINE, dtype=np.float32):
    data = np.zeros(shape, dtype=dtype)
    image = nib.Nifti1Image(data, np.asarray(affine, dtype=float))
    image.set_qform(image.affine, code=1)
    image.set_sform(image.affine, code=1)
    nib.save(image, str(path))


def write_markup(path, value, p1=(0.0, 0.0, 0.0), p2=(1.0, 0.0, 0.0)):
    data = {
        "markups": [
            {
                "measurements": [{"value": value}],
                "controlPoints": [{"position": list(p1)}, {"position": list(p2)}],
            }
        ]
    }
    path.write_text(json.dumps(data), encoding="utf-8")


class ReorientationTests(unittest.TestCase):
    def test_reorientation_preserves_data_and_world_corners(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source_path = tmp / "source.nii.gz"
            output_path = tmp / "output.nii.gz"
            data = np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4)
            source = nib.Nifti1Image(data, np.diag([1.0, 1.0, 2.0, 1.0]))
            source.set_qform(source.affine, code=1)
            source.set_sform(source.affine, code=2)
            nib.save(source, str(source_path))

            saved = reorient.reorient_nifti_lps(source_path, output_path)
            transform = nib.orientations.ornt_transform(
                nib.orientations.io_orientation(source.affine),
                nib.orientations.axcodes2ornt(reorient.TARGET_AXCODES),
            )
            expected = nib.orientations.apply_orientation(data, transform)

            self.assertEqual(tuple(nib.aff2axcodes(saved.affine)), ("L", "P", "S"))
            np.testing.assert_array_equal(np.asanyarray(saved.dataobj), expected)
            np.testing.assert_allclose(
                reorient._world_corners(source.shape, source.affine),
                reorient._world_corners(saved.shape, saved.affine),
                rtol=0.0,
                atol=1e-4,
            )
            self.assertNotEqual(int(saved.header["qform_code"]), 0)
            self.assertNotEqual(int(saved.header["sform_code"]), 0)

    def test_reorientation_rejects_anisotropic_in_plane_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source_path = tmp / "source.nii.gz"
            output_path = tmp / "output.nii.gz"
            save_nifti(source_path, affine=np.diag([1.0, 2.0, 3.0, 1.0]))
            with self.assertRaisesRegex(ValueError, "anisotropic in-plane spacing"):
                reorient.reorient_nifti_lps(source_path, output_path)
            self.assertFalse(output_path.exists())

    def test_reorientation_rejects_oblique_and_non_3d_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source_path = tmp / "source.nii.gz"
            output_path = tmp / "output.nii.gz"
            oblique = np.diag([1.0, 1.0, 2.0, 1.0])
            oblique[0, 1] = 0.1
            save_nifti(source_path, affine=oblique)
            with self.assertRaisesRegex(ValueError, "oblique/sheared"):
                reorient.reorient_nifti_lps(source_path, output_path)
            self.assertFalse(output_path.exists())

            save_nifti(source_path, shape=(4, 5, 6, 1))
            with self.assertRaisesRegex(ValueError, "must be a 3D volume"):
                reorient.reorient_nifti_lps(source_path, output_path)
            self.assertFalse(output_path.exists())


class GeometryTests(unittest.TestCase):
    def test_geometry_pair_accepts_matching_lps_isotropic_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ct_path, seg_path = tmp / "ct.nii.gz", tmp / "seg.nii.gz"
            save_nifti(ct_path)
            save_nifti(seg_path, dtype=np.uint8)
            run_pipeline.validate_geometry_pair(ct_path, seg_path)

    def test_geometry_single_accepts_label_map_without_ct(self):
        with tempfile.TemporaryDirectory() as tmp:
            seg_path = Path(tmp) / "case_seg.nii.gz"
            save_nifti(seg_path, dtype=np.uint8)
            run_pipeline.validate_geometry_single(seg_path)

    def test_geometry_single_rejects_anisotropic_label_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            seg_path = Path(tmp) / "case_seg.nii.gz"
            save_nifti(seg_path, affine=np.diag([-1.0, -2.0, 2.0, 1.0]), dtype=np.uint8)
            with self.assertRaisesRegex(ValueError, "isotropic in-plane spacing"):
                run_pipeline.validate_geometry_single(seg_path)

    def test_geometry_pair_rejects_shape_affine_orientation_and_anisotropy(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ct_path = tmp / "ct.nii.gz"
            seg_path = tmp / "seg.nii.gz"
            save_nifti(ct_path)

            save_nifti(seg_path, shape=(5, 5, 6), dtype=np.uint8)
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                run_pipeline.validate_geometry_pair(ct_path, seg_path)

            shifted = LPS_AFFINE.copy()
            shifted[0, 3] = 1.0
            save_nifti(seg_path, affine=shifted, dtype=np.uint8)
            with self.assertRaisesRegex(ValueError, "affine mismatch"):
                run_pipeline.validate_geometry_pair(ct_path, seg_path)

            save_nifti(seg_path, affine=np.diag([1.0, 1.0, 2.0, 1.0]), dtype=np.uint8)
            with self.assertRaisesRegex(ValueError, "must use LPS"):
                run_pipeline.validate_geometry_pair(ct_path, seg_path)

            save_nifti(seg_path, affine=np.diag([-1.0, -2.0, 2.0, 1.0]), dtype=np.uint8)
            with self.assertRaisesRegex(ValueError, "isotropic in-plane spacing"):
                run_pipeline.validate_geometry_pair(ct_path, seg_path)

            oblique = LPS_AFFINE.copy()
            oblique[0, 1] = 0.1
            save_nifti(seg_path, affine=oblique, dtype=np.uint8)
            with self.assertRaisesRegex(ValueError, "axis-aligned LPS"):
                run_pipeline.validate_geometry_pair(ct_path, seg_path)

            save_nifti(ct_path, shape=(4, 5, 6, 1))
            save_nifti(seg_path, dtype=np.uint8)
            with self.assertRaisesRegex(ValueError, "must be a 3D NIfTI volume"):
                run_pipeline.validate_geometry_pair(ct_path, seg_path)

    def test_measurement_loader_rejects_oblique_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "oblique.nii.gz"
            oblique = LPS_AFFINE.copy()
            oblique[0, 1] = 0.1
            save_nifti(path, affine=oblique)
            with self.assertRaisesRegex(ValueError, "axis-aligned LPS"):
                utils.load_nii(str(path))

    def test_measurement_loader_rejects_4d_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "four_dimensional.nii.gz"
            save_nifti(path, shape=(4, 5, 6, 1))
            with self.assertRaisesRegex(ValueError, "must be a 3D volume"):
                utils.load_nii(str(path))

    def test_non_override_path_validates_merged_segmentation_before_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ct_path = tmp / "case.nii.gz"
            out_root = tmp / "out"
            merged_path = out_root / "case" / "case_seg.nii.gz"
            merged_path.parent.mkdir(parents=True)
            save_nifti(ct_path)
            save_nifti(merged_path, shape=(5, 5, 6), dtype=np.uint8)
            with (
                mock.patch.object(run_pipeline, "run_totalseg"),
                mock.patch.object(run_pipeline, "merge_and_qc", return_value=({}, [])),
                mock.patch.object(run_pipeline, "measure_all") as measure_all,
            ):
                with self.assertRaisesRegex(ValueError, "shape mismatch"):
                    run_pipeline.process_case(ct_path, out_root)
            measure_all.assert_not_called()


class OutputSafetyTests(unittest.TestCase):
    def test_cpu_command_omits_fast(self):
        with tempfile.TemporaryDirectory() as tmp:
            completed = SimpleNamespace(returncode=0, stdout="", stderr="")
            with (
                mock.patch.object(run_pipeline, "TOTALSEG_EXE", "/usr/bin/true"),
                mock.patch.object(run_pipeline.subprocess, "run", return_value=completed) as run,
            ):
                run_pipeline.run_totalseg(Path("ct.nii.gz"), Path(tmp), device="cpu")
            command = run.call_args.args[0]
            self.assertIn("cpu", command)
            self.assertNotIn("--fast", command)
            self.assertIn("--nr_thr_saving", command)
            self.assertEqual(command[command.index("--nr_thr_saving") + 1], "1")

    def test_ct_free_process_writes_version_and_null_ct(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            seg_path = tmp / "case_T_031_seg.nii.gz"
            out_root = tmp / "out"
            save_nifti(seg_path, dtype=np.uint8)
            expected = {
                name: {
                    "diameter_mm": float(index + 1),
                    "p1_lps": [0.0, 0.0, 0.0],
                    "p2_lps": [1.0, 0.0, 0.0],
                    "markup_file": f"{name}.mrk.json",
                }
                for index, name in enumerate(run_pipeline.STRUCTURES)
            }
            with (
                mock.patch.object(run_pipeline, "measure_all"),
                mock.patch.object(run_pipeline, "collect_results", return_value=(expected, [])),
            ):
                summary = run_pipeline.process_case(None, out_root, seg_override=seg_path)

            self.assertEqual(summary["case"], "case_T_031")
            self.assertEqual(summary["pipeline_version"], "v1.0.4")
            self.assertIsNone(summary["input_ct"])
            saved = json.loads(
                (out_root / "case_T_031" / "case_T_031_measurements.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(saved["pipeline_version"], "v1.0.4")
            self.assertIsNone(saved["input_ct"])

    def test_stale_markups_are_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for name in run_pipeline.STRUCTURES:
                (tmp / f"{name}.mrk.json").write_text("stale", encoding="utf-8")
            keep = tmp / "keep.json"
            keep.write_text("keep", encoding="utf-8")
            run_pipeline.clear_stale_markups(tmp)
            self.assertTrue(keep.exists())
            self.assertFalse(any((tmp / f"{name}.mrk.json").exists() for name in run_pipeline.STRUCTURES))

    def test_invalid_measurements_become_missing_and_are_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_markup(tmp / "PT.mrk.json", 10.0)
            write_markup(tmp / "AA.mrk.json", 0.0)
            write_markup(tmp / "RV.mrk.json", float("nan"))
            measurements, missing = run_pipeline.collect_results(tmp)
            self.assertEqual(list(measurements), ["PT"])
            self.assertIn("AA", missing)
            self.assertIn("RV", missing)
            self.assertFalse((tmp / "AA.mrk.json").exists())
            self.assertFalse((tmp / "RV.mrk.json").exists())

    def test_ratios_use_stored_diameters_and_null_missing_components(self):
        measurements = {
            "PT": {"diameter_mm": 20.0},
            "AA": {"diameter_mm": 40.0},
            "RV": {"diameter_mm": 50.0},
            "LV": {"diameter_mm": 25.0},
        }
        self.assertEqual(
            run_pipeline.calculate_ratios(measurements),
            {"PT_AA_ratio": 0.5, "RV_LV_ratio": 2.0},
        )
        del measurements["AA"]
        self.assertIsNone(run_pipeline.calculate_ratios(measurements)["PT_AA_ratio"])

    def test_markup_writer_rejects_zero_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for name, value in (("zero", 0.0), ("nan", float("nan"))):
                path = tmp / f"{name}.mrk.json"
                with self.assertRaisesRegex(ValueError, "finite and > 0"):
                    utils.save_line_markup_json(path, (0, 0, 0), (1, 0, 0), value)
                self.assertFalse(path.exists())


class LabelMapTests(unittest.TestCase):
    def test_labelmap_rejects_mask_geometry_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            names = [
                "pulmonary_artery.nii.gz",
                "aorta.nii.gz",
                "heart_atrium_left.nii.gz",
                "heart_ventricle_left.nii.gz",
                "heart_atrium_right.nii.gz",
                "heart_ventricle_right.nii.gz",
            ]
            for name in names:
                save_nifti(tmp / name, dtype=np.uint8)
            shifted = LPS_AFFINE.copy()
            shifted[1, 3] = 2.0
            save_nifti(tmp / names[-1], affine=shifted, dtype=np.uint8)
            with self.assertRaisesRegex(ValueError, "affine does not match"):
                labelmap.build_multiclass_labelmap(tmp, tmp / "merged.nii.gz")

    def test_labelmap_rejects_oblique_reference_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            oblique = LPS_AFFINE.copy()
            oblique[0, 1] = 0.1
            save_nifti(tmp / "pulmonary_artery.nii.gz", affine=oblique, dtype=np.uint8)
            with self.assertRaisesRegex(ValueError, "axis-aligned LPS"):
                labelmap.build_multiclass_labelmap(tmp, tmp / "merged.nii.gz")


if __name__ == "__main__":
    unittest.main()
