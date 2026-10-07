# ODCM-SADAS

Research code for a multimodal vehicle and vulnerable-road-user detection and
classification pipeline using camera, LiDAR, and radar data. The pipeline
combines sensor preprocessing and calibration, a fused YOLO detector, ASSA
fusion-weight search, and a GOA-tuned LSTM classifier.

> This code is intended solely for research and study purposes and is not for
> commercial use.

This repository contains source code and documentation only. The nuScenes
dataset, generated preprocessing files, trained models, and experimental
results are not included. The training and evaluation stages must be run on
the target machine before reporting results.

## Pipeline

1. **Collect data (optional):** select a subset from an authorized local
   nuScenes installation using
   [`Project/nuscense_data_collecton/README.md`](Project/nuscense_data_collecton/README.md).
2. **Preprocess:** build splits and tracks, project/calibrate sensor data, and
   export images, labels, and sensor features.
3. **Train fusion model:** train the fused detector, then search the modality
   weights with ASSA.
4. **Classify tracks:** export detections, tune/train the LSTM, and collect
   evaluation metrics.

See [Project/EXECUTION.md](Project/EXECUTION.md) for end-to-end commands that
run from a regular terminal. The
[preprocessing guide](Project/ReadMe.md) describes the data products and
class mapping.

## Requirements

- Python 3.10 or newer
- PyTorch and torchvision, installed using the command appropriate for the
  machine and its CPU/GPU configuration
- For model training and search, a supported NVIDIA GPU and CUDA-enabled
  PyTorch installation are strongly recommended

Create an isolated environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell, activate it with:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the PyTorch and torchvision build suited to the host by following the
official [PyTorch installation selector](https://pytorch.org/get-started/locally/),
then install the remaining project dependencies:

```bash
python -m pip install -r requirements.txt
```

## Data and model files

The scripts expect the selected nuScenes sample directories at
`Samples/scenes/<sample-token>/`, with each sample containing `metadata.json`,
`camera/`, `lidar/`, and `radar/`. The optional collector writes this layout
under `Samples/scenes/`. nuScenes access and redistribution are subject to the
dataset's terms; obtain and prepare the data separately.

Place the compatible pretrained YOLO11s checkpoint at `yolo11s.pt` in the
repository root. Pretrained and trained model weights are not included in the
repository. Do not publish dataset files or model weights unless their
licenses permit redistribution.

## Reproducibility and publication

- The scripts use fixed defaults and expose command-line options for settings
  such as epochs, batch size, random seed, and output paths. Record any
  deviations when reporting results.
- The train/validation/test split used by model code is scene-grouped to avoid
  placing samples from the same scene in multiple splits.
- No trained metrics or publication citation are supplied here. Add verified
  experimental results and the final citation before release.

## Contact

For code clarification or to report a technical issue, contact the author at
[rmallava@gitam.in](mailto:rmallava@gitam.in).
