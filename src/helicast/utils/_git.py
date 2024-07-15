import subprocess
from pathlib import Path
from typing import Union

__all__ = [
    "get_git_root",
]


def get_git_root() -> Union[Path, None]:
    """Returns the root directory of the Git repository, if available. If not,
    returns None."""
    try:
        # Run the 'git rev-parse --show-toplevel' command to get the root directory of the repository
        git_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"]
        ).strip()
        return Path(git_root.decode("utf-8"))
    except subprocess.CalledProcessError:
        return None
