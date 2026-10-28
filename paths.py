"""Central paths for the repository. Override with environment variables instead of editing scripts.

SMORFFI_FEATURES  path to the SMoRFFI per-frame feature file features_v1.npz (see data/README.md)
EMULATED_DIR      folder with the emulated impersonation corpus (Zenodo record 20186947), used by audit/ras.py
PAPER_TEX         path to the paper source main.tex, used only by rapid/verify_numbers.py"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_FILE = Path(os.environ.get('SMORFFI_FEATURES', ROOT / 'data' / 'features_v1.npz'))
EMULATED_DIR = Path(os.environ.get('EMULATED_DIR', ROOT / 'data' / 'RFFI_Emulated'))
FIG_DIR = ROOT / 'figures'
PAPER_TEX = Path(os.environ.get('PAPER_TEX', ROOT / 'paper' / 'main.tex'))
SMORFFI_SHA256 = '4d3efa580e733493add9a920a457517bb9c17a5b91dc0611d77c8217aef29ab8'
