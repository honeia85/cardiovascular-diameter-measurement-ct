# Security policy

This repository is frozen research software for reproducing the accompanying study. Its
dependency versions are pinned to the evaluated environment and should not be interpreted
as a maintained production stack.

- Run the pipeline in an isolated Python 3.12 environment on trusted, local inputs.
- Do not expose the command-line pipeline as a network service.
- Obtain TotalSegmentator only from its official distribution, and run
  `python verify_weights.py` before relying on downloaded model weights.
- Keep TotalSegmentator license keys, credentials, medical images, and generated outputs
  outside version control. Never attach patient data to a GitHub issue.
- Before clinical, network-facing, or other production use, audit current dependencies,
  update them as appropriate, and revalidate the complete segmentation and measurement
  pipeline because dependency changes may change inference output.

Report a security concern through GitHub private vulnerability reporting. Do not include
protected health information or credentials in the report.
