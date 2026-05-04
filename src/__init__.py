import os
from pathlib import Path

MTRAG_DATA = Path(os.environ.get("MTRAG_DATA", str(Path.cwd() / "mt-rag-benchmark")))
MTRAG_SPLITS = Path(os.environ.get("MTRAG_SPLITS", str(Path.cwd() / "splits")))
