import sys
from pathlib import Path

# スクリプト群（common.py など）を import できるようにする
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
