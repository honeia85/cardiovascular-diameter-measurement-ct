"""Verify that the installed TotalSegmentator weights match the ones used for the
published results.

Segmentation output depends on the weights, so reproducing the reported numbers
requires the same checkpoints. Run this after the first pipeline run (weights are
downloaded on demand).

    python verify_weights.py
"""
import hashlib
import sys
from pathlib import Path

WEIGHTS_ROOT = Path.home() / ".totalsegmentator" / "nnunet" / "results"

# SHA256 of the checkpoints used for the published results (downloaded 2026-02-19).
EXPECTED = {
    "Dataset301_heart_highres_1559subj/nnUNetTrainer__nnUNetPlans__3d_fullres":
        {"fold_0/checkpoint_final.pth":
            "ca616ae0e9e2f4d7e8b01bc8fca00e430c4392d6a449ccdc2f9dfc4bc03f62eb",
         "plans.json":
            "3359f3da44ee822bb49eb72b1248c531774ac6227a35d90055ab4533a18efb32"},
    "Dataset297_TotalSegmentator_total_3mm_1559subj/"
    "nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres":
        {"fold_0/checkpoint_final.pth":
            "1e38e40356adc2706a662e365405a97f862d62a7da65f6bf81d025aee1b979ac",
         "plans.json":
            "d1cb3c15f53dc36fdb618e0f9082573d1f50b74a9b5fa73094ac8680845839f1"},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print(f"weights root: {WEIGHTS_ROOT}\n")
    problems = []
    for model_dir, files in EXPECTED.items():
        print(model_dir.split("/")[0])
        for rel, want in files.items():
            p = WEIGHTS_ROOT / model_dir / rel
            if not p.exists():
                print(f"  MISSING  {rel}")
                problems.append(f"missing: {p}")
                continue
            got = sha256(p)
            if got == want:
                print(f"  OK       {rel}")
            else:
                print(f"  MISMATCH {rel}\n           expected {want}\n           got      {got}")
                problems.append(f"mismatch: {p}")
        print()

    if problems:
        print("FAILED - the installed weights differ from the published ones.")
        print("Reported measurements may not reproduce.")
        for p in problems:
            print("  -", p)
        return 1
    print("All weights match the published run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
