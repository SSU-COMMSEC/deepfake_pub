"""Where everything lives for the deepfake (SimSwap) target.

UCF-101 is the same dataset the C3D target uses (dfbench.paths). PhantomSeal is an external
checkout with its own third-party models and checkpoints; point PHANTOMSEAL_ROOT at it
(default: repos/PhantomSeal, next to repos/u3d). Bulky caches (decoded face frames, ~3 GB) go to
DEEPFAKE_CACHE (default: cache/deepfake).
"""

import os
import sys
from pathlib import Path

_COMMON = Path(__file__).resolve().parents[1] / "common"
if str(_COMMON) not in sys.path:
    sys.path.insert(0, str(_COMMON))

from dfbench.paths import DF_ROOT, SPLIT_DIR as _SPLIT_DIR, VIDEO_DIR  # noqa: E402

ROOT = Path(DF_ROOT)
PHANTOMSEAL = Path(os.environ.get("PHANTOMSEAL_ROOT", ROOT / "repos" / "PhantomSeal"))
SIMSWAP = PHANTOMSEAL / "third_party" / "SimSwap"

UCF_ROOT = Path(VIDEO_DIR)
SPLIT_DIR = Path(_SPLIT_DIR)

CACHE = Path(os.environ.get("DEEPFAKE_CACHE", ROOT / "cache" / "deepfake"))
RESULTS = Path(os.environ.get("DEEPFAKE_RESULTS", ROOT / "results" / "deepfake"))


def setup_imports() -> None:
    """Makes PhantomSeal's `src.*` importable (its code is used unmodified)."""
    if not (PHANTOMSEAL / "src").is_dir():
        raise SystemExit(f"PhantomSeal not found at {PHANTOMSEAL}. Clone it there or set "
                         f"PHANTOMSEAL_ROOT (see docs/DEEPFAKE.md).")
    if str(PHANTOMSEAL) not in sys.path:
        sys.path.insert(0, str(PHANTOMSEAL))
