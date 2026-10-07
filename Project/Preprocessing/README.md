# Preprocessing

This stage prepares the collected nuScenes samples for the detector and
temporal classifier. It creates scene-grouped train/validation/test splits,
projects and rasterizes camera, LiDAR, and radar data, and computes
normalization statistics from the training split only.

Run commands below from the repository root. The expected sample layout is
`Samples/scenes/<sample-token>/`, containing `metadata.json` and the
`camera/`, `lidar/`, and `radar/` directories.

## Run

```bash
python Project/Preprocessing/make_index.py --metadata-dir Samples/scenes --out Project/Output --seed 42
python Project/Preprocessing/preprocess.py process --samples-root Samples/scenes --out Project/Output
python Project/Preprocessing/preprocess.py stats --out Project/Output --split train
python Project/Preprocessing/verify_all.py --out Project/Output --samples-root Samples/scenes
```

Run verification before continuing to model training. It checks the processed
samples, sensor maps, labels, calibration-related projections, and model input
construction. Resolve reported failures rather than proceeding with
incomplete or invalid data.

## Outputs

The stage writes to `Project/Output/`:

- `splits.json` and `tracks.json`
- `images/` and `labels/` in a shared 640x640 image canvas
- `sensors/` archives with projected maps, LiDAR BEV maps, calibration, and
  object features
- `norm_stats.json`, computed using training samples only

See [the project preprocessing guide](../ReadMe.md) for image dimensions,
camera channels, BEV views, and class IDs. See the
[execution guide](../EXECUTION.md) for full setup and pipeline instructions.
