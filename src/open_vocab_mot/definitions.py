from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv()

DUKEMTMC_VIDEO_REID_PATH = Path(os.getenv("DUKEMTMC_VIDEO_REID_PATH"))