"""Paths shared by the Fusion stage; importing this module also exposes the Preprocessing modules."""
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT.parent
PREPROCESSING = PROJECT / "Preprocessing"
OUT = PROJECT / "Output"
SAMPLES_ROOT = CODE_ROOT / "Samples" / "scenes"
YOLO_WEIGHTS = CODE_ROOT / "yolo11s.pt"

if str(PREPROCESSING) not in sys.path:
    sys.path.insert(0, str(PREPROCESSING))
