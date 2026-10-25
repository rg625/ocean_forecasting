"""Machine-independent roots for the audit scripts (the Python side of env.sh).

REPO_ROOT is this checkout / worktree; MAIN_ROOT is the main checkout on this machine, where
data, the LSiM weights and legacy runs live (env OCEAN_MAIN overrides it).
"""

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _main_root() -> Path:
    if os.environ.get("OCEAN_MAIN"):
        return Path(os.environ["OCEAN_MAIN"])
    common = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return (REPO_ROOT / common).resolve().parent


MAIN_ROOT = _main_root()
