"""Portable data paths: CLI paths are relative to the repository, not caller cwd."""
import os
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]


def repo_path(value, base=None):
    """Resolve a relative CLI path; base overrides the root for paper output."""
    text = str(value)
    if Path(text).is_absolute() or PureWindowsPath(text).anchor:
        raise ValueError("Use a relative data/output path")
    return ((ROOT if base is None else base) / text.replace("\\", "/")).resolve()


def relative_path(path, base=None):
    """Serialize data paths with portable separators for logs/subprocesses."""
    path = Path(path)
    if not path.is_absolute():
        return str(path).replace("\\", "/")
    return Path(os.path.relpath(path, ROOT if base is None else base)).as_posix()
