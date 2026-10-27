# Independent RGB–IR reliability fusion experiment

This implementation is independent of the code under `drone/`. It uses the installed Ultralytics package as a dependency, but defines its own paired-input architecture here.

## Scope

- Two separate YOLO11-OBB feature streams, initialized from the existing RGB and IR checkpoints.
- Reliability gates are learned from the paired feature maps at backbone stages 4, 6, and 10.
- The fused features are passed through an OBB neck/head.
- No TASCNet, TaRFusion, S2RH, SARM, or imports from `drone/` are used.

## Important status

This branch is an implementation scaffold, **not yet validated for training**. Do not launch a long run until the paired dataset path, forward pass, OBB loss, and backward pass all pass the smoke test. The earlier attempt that imported `drone/scripts/tascnet_yolo11_obb.py` is not the implementation to use.

The dataset's RGB/IR labels are not identical. For the primary comparison, the intended convention is to use RGB labels as the training/evaluation reference and use IR images as a paired input. This limitation must be disclosed.

## Environment

The experiment targets the existing local environment:
- Python 3.13
- PyTorch with CUDA
- Ultralytics YOLO11-OBB
- paired dataset under `D:/dronevehicle_feasibility/yolo_obb_data`

A separate end-to-end trainer and smoke-test need to be completed before this scaffold can be considered runnable.
