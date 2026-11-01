import os
from pathlib import Path

# Keep model/tokenizer downloads inside the project.
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / ".cache" / "hf"))
