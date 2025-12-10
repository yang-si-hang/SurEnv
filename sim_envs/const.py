import os
from pathlib import Path

ROOT_DIR_PATH = os.path.abspath(os.path.dirname(__file__))
ASSET_DIR_PATH = os.path.join(ROOT_DIR_PATH, 'assets')
DATA_DIR_PATH = os.path.join(ROOT_DIR_PATH, 'data')
DEBUG_DIR_PATH = Path(ROOT_DIR_PATH) / "debug"