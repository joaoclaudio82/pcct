"""Optional MONAI inference adapter.

This module intentionally avoids bundling model weights. It defines a stable
inference contract for validated/pretrained models supplied by the research team.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def predict_with_monai(
    volume: np.ndarray,
    model_path: str | Path,
    device: str = "cpu",
    threshold: float = 0.5,
) -> np.ndarray:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("instale o extra 'ml' para usar inferência MONAI/PyTorch") from exc

    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    x = torch.from_numpy(volume.astype(np.float32))[None, None].to(device)
    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)
    return (prob[0, 0].cpu().numpy() >= threshold)


def monai_available() -> bool:
    try:
        import monai  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False
