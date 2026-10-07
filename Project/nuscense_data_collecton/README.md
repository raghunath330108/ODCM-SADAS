# nuScenes data collector

This directory contains a utility for selecting and collecting a
camera-visible nuScenes subset for the preprocessing and training pipeline.
The source dataset is not included and must be available locally under terms
that permit its use.

## Install

From the repository root, activate the project's Python environment and
install the dependencies as described in the [root README](../../README.md):

```bash
python -m pip install -r requirements.txt
```

## Collect

Run a dry-run first. Replace `/path/to/nuscenes` with the directory containing
the `v1.0-trainval/` folder:

```bash
python Project/nuscense_data_collecton/collect_nuscenes.py --dataset-root /path/to/nuscenes --version v1.0-trainval --output-dir Samples --dry-run
```

If the reported sample availability and split counts are suitable, run the
same command without `--dry-run` to copy the selected files. The destination
must not already exist; existing output is never overwritten. Use
`python Project/nuscense_data_collecton/collect_nuscenes.py --help` for all
options.

The collector uses a deterministic seed of 42 and targets 250 class
memberships per class by default. Multi-class samples can contribute to more
than one class quota. It prefers visibility levels 3 and 4, and uses level 2
only as needed. Samples require five camera channels, LiDAR, and all five
radar channels; `CAM_BACK` is not collected.

## Output layout

```text
Samples/
├── metadata/
│   └── collection_manifest.json
└── scenes/
    └── <sample-token>/
        ├── metadata.json
        ├── camera/
        ├── lidar/
        └── radar/
```

This `Samples/scenes/` directory is the default input for preprocessing and
fusion. The collector manifest records selection and split provenance;
downstream model splits are created separately and grouped by scene to avoid
temporal leakage.

## Tests

From the repository root:

```bash
python -m unittest discover -s Project/nuscense_data_collecton -p 'test_*.py'
```
