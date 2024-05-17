import sys
from pathlib import Path

source_path = Path(__file__).parents[1].joinpath("src").resolve()
sys.path.append(str(source_path))
