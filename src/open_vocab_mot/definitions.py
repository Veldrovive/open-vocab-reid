from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv()

DUKEMTMC_VIDEO_REID_PATH = Path(os.getenv("DUKEMTMC_VIDEO_REID_PATH"))
DUKEMTMC_VIDEO_REID_SIDECAR_PATH = Path(os.getenv("DUKEMTMC_VIDEO_REID_SIDECAR_PATH"))
MARS_PATH = Path(os.getenv("MARS_PATH"))