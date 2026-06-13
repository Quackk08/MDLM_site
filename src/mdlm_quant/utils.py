"""Shared reproducibility and CSV helpers."""

from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    """Resolve an experiment device without silently claiming another device."""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def ensure_dir(path: str | Path) -> Path:
    """Create a directory and return it as a Path."""

    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write dictionaries to CSV."""

    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    """Read a CSV file as dictionaries."""

    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path: str | Path, data: Any) -> None:
    """Write JSON with UTF-8 encoding."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def parameter_count(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters."""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def package_versions() -> dict[str, str]:
    """Collect package versions used by the experiment."""

    versions = {"python": ".".join(map(str, os.sys.version_info[:3])), "torch": torch.__version__, "numpy": np.__version__}
    try:
        import matplotlib

        versions["matplotlib"] = matplotlib.__version__
    except Exception:
        versions["matplotlib"] = "unavailable"
    return versions


def wall_time() -> float:
    """Return a monotonic timestamp."""

    return time.perf_counter()


def finite_float(value: float, fallback: float = 0.0) -> float:
    """Convert NaN/Inf values to a finite fallback."""

    return value if math.isfinite(value) else fallback

