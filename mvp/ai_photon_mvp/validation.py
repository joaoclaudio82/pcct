"""Shared numerical input contracts; array spacing is always z, y, x."""
import numpy as np


def spacing_zyx(spacing):
    values = np.asarray(spacing, dtype=float)
    if values.ndim == 0:
        values = np.repeat(values, 3)
    if values.shape != (3,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("spacing must contain three finite positive values")
    return tuple(float(v) for v in values)


def volume_array(data, name="volume", finite=True):
    array = np.asarray(data)
    if array.ndim != 3 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty 3D array")
    if finite and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def paired_arrays(first, second):
    a = volume_array(first)
    b = volume_array(second)
    if a.shape != b.shape:
        raise ValueError("arrays must have identical shapes")
    return a, b
