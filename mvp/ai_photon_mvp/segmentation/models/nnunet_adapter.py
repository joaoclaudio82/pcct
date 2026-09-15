"""Thin adapter contract for nnU-Net v2 inference.

The adapter validates installation and documents the expected model-folder
contract without bundling external weights or datasets.
"""
from __future__ import annotations

from pathlib import Path


def nnunet_available() -> bool:
    try:
        import nnunetv2  # noqa: F401
        return True
    except ImportError:
        return False


def validate_nnunet_model_folder(path: str | Path) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if not p.is_dir():
        raise ValueError("o caminho do modelo nnU-Net deve ser um diretório")
    if not any(p.rglob("checkpoint_final.pth")):
        raise ValueError("checkpoint_final.pth não encontrado no diretório informado")
    return p
