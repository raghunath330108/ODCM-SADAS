# Multimodal fusion

This stage trains a fused YOLO detector that uses camera images, LiDAR BEV
maps, and radar maps, then tunes the global modality weights on the validation
split with the Artificial Search Swarm Algorithm (ASSA).

Run commands from the repository root after preprocessing. The scripts use
`Project/Output/` and expect the sample directories at `Samples/scenes/` and a
compatible `yolo11s.pt` checkpoint in the repository root.

## Check and train

```bash
python Project/Fusion/check_fusion.py
python Project/Fusion/train_fusion.py
```

Training defaults to 100 epochs and batch size 16, with equal initial fusion
weights. It writes `best.pt`, `last.pt`, and `train_log.csv` under
`Project/Output/fusion/`. CUDA is recommended. If GPU memory is insufficient,
reduce the batch size with `--batch` and record the setting used.

## Search modality weights

After training finishes, run:

```bash
python Project/Fusion/assa_search.py
```

The default search uses the validation split, 30 iterations, and a population
decreasing from 20 to 5. It writes `assa_result.json` alongside the detector
checkpoints. Review its selected weights and metrics against the
equal-weight baseline before proceeding.

Do not use `--trial` outputs as experimental results. See the
[execution guide](../EXECUTION.md) for the end-to-end workflow.
