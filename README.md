
# ODCM-SADAS
##Title : "Deep Transfer Learning for Adaptive Sensor Fusion: YOLOv11 with ASSA-Optimized Multi-Sensor Integration and GOA-Tuned LSTM"
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

Results reported in the associated journal publication may not be reproduced
exactly by running this code in a different environment. Outcomes can depend
on factors including:

- **Data collection and selection:** nuScenes release/version, available
  sensor files, collection criteria, class quotas, visibility filters, and
  the exact samples collected. A different subset can change class balance
  and the number and length of object tracks.
- **Preprocessing and splits:** software versions, generated labels and sensor
  features, preprocessing options, and split assignments. The model pipeline
  uses scene-grouped splits to reduce leakage between train, validation, and
  test data; use the same processed data and split files when comparing runs.
- **Compute environment:** GPU model and memory, CPU, operating system,
  NVIDIA driver, CUDA, PyTorch/torchvision, and other dependency versions.
  Floating-point implementations, hardware kernels, data-loader behavior,
  and parallel execution can lead to small numerical differences and
  different training trajectories.
- **Run configuration:** random seeds, batch size, worker count, training
  duration, checkpoint initialization, and any command-line overrides.
  Search and training procedures may also be sensitive to initialization and
  stochastic operations.

For the closest reproduction of the journal results, use the same collected
sample set, pretrained checkpoint, split files, preprocessing outputs, and
training/search settings used for those results. This repository does not
include those data, weights, or published run artifacts, so exact replication
cannot be guaranteed from the source code alone.

### Recommended reporting practice

1. Record the nuScenes version and collection configuration, sample/annotation
   counts by class and split, and a checksum or archived copy of the split
   manifests.
2. Record the repository commit, Python and package versions, OS, GPU model,
   driver/CUDA versions, command lines, and all non-default settings.
3. Preserve the initial checkpoint and generated preprocessing outputs with
   the run, subject to dataset and weight licensing restrictions.
4. Run stochastic experiments with multiple seeds when feasible and report
   the mean and variability, alongside the individual settings.
5. Keep test data for final evaluation only; select checkpoints and tune
   hyperparameters using training and validation data.
6. Report failed or incomplete runs and deviations from the published setup
   rather than presenting them as exact replications.

No trained metrics or publication citation are supplied here. Add verified
results and the final citation before release.

## Project updates

Watch this repository for ongoing project updates. We will update the
repository as relevant code, documentation, and project information become
available.

## Contact

For code clarification or to report a technical issue, contact the author at
[rmallava@gitam.in](mailto:rmallava@gitam.in).
