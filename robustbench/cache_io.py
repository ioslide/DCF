"""Read owned tensors without retaining a second copy in the file cache."""

import os
from pathlib import Path

import torch


def advise_file_pages(file, offset=0, length=0):
    """Best-effort release of clean pages; never flush dirty data globally."""
    if hasattr(os, "posix_fadvise"):
        try:
            os.posix_fadvise(file.fileno(), offset, length, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass


def drop_file_pages(path):
    try:
        with Path(path).open("rb") as file:
            advise_file_pages(file)
    except OSError:
        pass


def load_tensor_cache(path, **kwargs):
    """torch.load owns its storage, so its source pages can be released."""
    try:
        return torch.load(path, **kwargs)
    finally:
        drop_file_pages(path)
