# nuScenes data collection

The optional collector prepares camera-visible samples for six classes:
bicycle, bus, car, motorcycle, person, and truck. It expects an authorized
local nuScenes installation and writes sample directories under
`Samples/scenes/`.

Run a dry-run first, replacing `/path/to/nuscenes` with the local dataset root:

```bash
python Project/nuscense_data_collecton/collect_nuscenes.py --dataset-root /path/to/nuscenes --version v1.0-trainval --output-dir Samples --dry-run
```

Review candidate availability and the class/split counts. To collect, repeat
the command without `--dry-run`. The output directory must not already exist;
the collector will not overwrite an existing destination.

Selection targets 250 sample memberships per class by default. A frame may
count for multiple classes. Visibility levels 3 and 4 are preferred, with
level 2 used as needed; samples must include the required five cameras,
LiDAR, and five radar channels. The collector records its selection and split
provenance in `Samples/metadata/collection_manifest.json`.

The collector's unit tests can be run with:

```bash
python -m unittest discover -s Project/nuscense_data_collecton -p 'test_*.py'
```

nuScenes data is not included in this repository. Follow the dataset terms
when accessing, processing, or redistributing it.
