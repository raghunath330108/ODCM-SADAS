# Running the pipeline

These instructions use a terminal or command prompt only. Run commands from
the repository root unless a command explicitly changes directory. On Windows,
replace `python3` with `py` if needed and use the activation command in the
root README.

## 1. Prepare data and model weights

The scripts expect:

- `Samples/scenes/<sample-token>/metadata.json`
- Camera images under each sample's `camera/` directory
- `lidar/LIDAR_TOP.pcd.bin` and five radar point-cloud files under `radar/`
- A compatible pretrained `yolo11s.pt` checkpoint in the repository root

The repository does not include nuScenes data or pretrained weights. If using
the included collector, first install the project requirements and run its
dry-run against a local nuScenes release:

```bash
python Project/nuscense_data_collecton/collect_nuscenes.py --dataset-root /path/to/nuscenes --version v1.0-trainval --output-dir Samples --dry-run
```

Review the reported candidate and split counts. If they are suitable, repeat
the command without `--dry-run` to collect the files. The output directory
must not already exist. The collector's output at `Samples/scenes` is the
sample root used by the downstream defaults.

## 2. Create splits and preprocess

Run from the repository root:

```bash
python Project/Preprocessing/make_index.py --metadata-dir Samples/scenes --out Project/Output --seed 42
python Project/Preprocessing/preprocess.py process --samples-root Samples/scenes --out Project/Output
python Project/Preprocessing/preprocess.py stats --out Project/Output --split train
python Project/Preprocessing/verify_all.py --out Project/Output --samples-root Samples/scenes
```

Do not continue if indexing, preprocessing, or verification reports missing or
invalid data. `verify_all.py` must finish with an overall pass before model
training.

## 3. Check the fusion setup and train the detector

Confirm that PyTorch can see the intended GPU:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

For full training, CUDA should be available. The fusion checks and training
commands are:

```bash
python Project/Fusion/check_fusion.py
python Project/Fusion/train_fusion.py
```

The default training run uses 100 epochs and batch size 16. It writes
`Project/Output/fusion/best.pt`, `last.pt`, and `train_log.csv`. If GPU memory
is insufficient, rerun with a smaller `--batch` and record the change. Inspect
the training log and validation metrics before continuing.

## 4. Search fusion weights with ASSA

After detector training completes:

```bash
python Project/Fusion/assa_search.py
```

The default search uses 30 iterations and a population that decreases from 20
to 5. It evaluates on the validation split and writes
`Project/Output/fusion/assa_result.json`. Check the selected weights and
compare the reported objective and metrics with the equal-weight result.

## 5. Export detections and train/evaluate the LSTM

Run these commands in order:

```bash
python Project/Classification/export_detections.py
python Project/Classification/goa_lstm.py
python Project/Classification/collect_metrics.py
```

Detection export writes observations and a detector report under
`Project/Output/classification/`. The LSTM script runs the GOA hyperparameter
search and then final training (300 epochs by default). To run the separate
3,000-epoch configuration referenced in the manuscript, use a separate output
directory:

```bash
python Project/Classification/goa_lstm.py --epochs 3000 --out Project/Output/classification_3000
python Project/Classification/collect_metrics.py --cls Project/Output/classification_3000
```

Only evaluate final performance on the test split after selecting settings
using the training and validation splits. Report the LSTM alongside the
detector-only baselines, per-class metrics, and the dataset limitations
(including class imbalance and short tracks).

## 6. Outputs and run records

Generated files are written under `Project/Output/`:

- `splits.json`, `tracks.json`, `norm_stats.json`
- `images/`, `labels/`, and `sensors/`
- `fusion/` detector checkpoints, training log, and ASSA results
- `classification/` detections, LSTM checkpoints/logs, and final metrics

Record the Python/PyTorch/CUDA versions, GPU model, command-line settings,
wall-clock time, and any errors for every reported run. Keep validation and
test results separate; do not replace failed runs with trial or synthetic
outputs.
