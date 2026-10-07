# Detection export and temporal classification

This stage exports detections and object-level features from the trained
fusion detector, tunes and trains an LSTM on object tracks, and collects
classification and detector-baseline metrics.

Run commands from the repository root after preprocessing and fusion training.
Inputs include the scene-grouped splits, processed sensor data, the trained
checkpoint under `Project/Output/fusion/`, and the ASSA result when available.

## Run

Run these commands in order:

```bash
python Project/Classification/export_detections.py
python Project/Classification/goa_lstm.py
python Project/Classification/collect_metrics.py
```

Detection export writes observations and a detector report to
`Project/Output/classification/`. LSTM training runs the GOA hyperparameter
search and then trains the final model (300 epochs by default). Metrics
collection generates the final JSON and Markdown reports, per-class CSVs,
confusion matrices, and training curves.

For the separate 3,000-epoch setting, use a distinct output directory:

```bash
python Project/Classification/goa_lstm.py --epochs 3000 --out Project/Output/classification_3000
python Project/Classification/collect_metrics.py --cls Project/Output/classification_3000
```

Use the validation split for model selection and reserve the test split for
final evaluation. Report LSTM results alongside detector-only baselines and
include relevant dataset limitations. Dry-run outputs and synthetic-feature
fallbacks are for mechanics checks only and are not experimental results.

See the [execution guide](../EXECUTION.md) for the full run sequence and
output locations.
