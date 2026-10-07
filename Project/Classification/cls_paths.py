"""Paths shared by the Classification stage (3.4 export, 3.5 LSTM, 3.6 GOA); also exposes the Fusion and Preprocessing modules."""
import sys
from pathlib import Path

FUSION = Path(__file__).resolve().parents[1] / "Fusion"
if str(FUSION) not in sys.path:
    sys.path.insert(0, str(FUSION))

from fusion_paths import OUT, PROJECT, SAMPLES_ROOT, YOLO_WEIGHTS  # noqa: E402,F401
