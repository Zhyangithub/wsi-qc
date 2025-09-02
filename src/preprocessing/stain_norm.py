# -*- coding: utf-8 -*-
"""
Macenko stain normalization utilities.

This module implements:
- Robust tissue masking
- Illumination (Io) estimation per tile
- Macenko stain basis extraction (2x stains)
- Concentration solving and reconstruction with reference basis
- A full tile normalization pipeline

All functions operate on uint8 RGB images (H, W, 3) in [0, 255].
"""
from __future__ import annotations

import json
import math
import numpy as np
from typing import Tuple, Dict, Any, Optional

# -----------------------------
# Utility helpers
# -----------------------------

_EPS = 1e-8

def _to_float(rgb: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB [0..255] to float64 array."""
    return rgb.astype(np.float64, copy=False)

def _unit(v: np.ndarray) -> np.ndarray:
    """Normalize vectors along last dimension to unit length (avoid 0/0)."""
    n = np.linalg.norm(v, axis=-1, keepdims=True) + _EPS
    return v / n

# -----------------------------
# Tissue mask and Io estimation
# -----------------------------

def rgb2hsv(rgb: np.ndarray) -> np.ndarray:
    """Vectorized RGB->HSV, outputs float64 in [0,1]."""
    # from colorsys but vectorized: we can normalize and rely on numpy operations
    rgb01 = _to_float(rgb) / 255.0
    r, g, b = rgb01[..., 0], rgb01[..., 1], rgb01[..., 2]
    cmax = np.max(rgb01, axis=-1)
    cmin = np.min(rgb01, axis=-1)
    delta = cmax - cmin
    # Hue
    h = np.zeros_like(cmax)
    mask = delta > _EPS
    # red is max
    idx = (cmax == r) & mask
    h[idx] = ( (g[idx] - b[idx]) / (delta[idx] + _EPS) ) % 6.0
    # green is max
    idx = (cmax == g) & mask
    h[idx] = ((b[idx] - r[idx]) / (delta[idx] + _EPS)) + 2.0
    # blue is max
    idx = (cmax == b) & mask
    h[idx] = ((r[idx] - g[idx]) / (delta[idx] + _EPS)) + 4.0
    h = (h / 6.0) % 1.0
    # Saturation
    s = np.zeros_like(cmax)
    s[mask] = delta[mask] / (cmax[mask] + _EPS)
    # Value
    v = cmax
    return np.stack([h, s, v], axis=-1)

def tissue_mask(rgb: np.ndarray, sat_thresh: float = 0.1, val_thresh: float = 0.98) -> np.ndarray:
    """
    Create a tissue mask using HSV saturation and value thresholds.
    - sat_thresh: minimum saturation to count as tissue (default 0.10).
    - val_thresh: exclude very bright background (value near 1.0).
    """
    hsv = rgb2hsv(rgb)
    s = hsv[..., 1]
    v = hsv[..., 2]
    mask = (s > sat_thresh) & (v < val_thresh)
    return mask

def estimate_io(rgb: np.ndarray, mask: Optional[np.ndarray] = None, percentile: float = 95.0) -> np.ndarray:
    """
    Estimate illumination (Io) per channel from tissue pixels using high percentile (default P95).
    Enforce a reasonable lower bound (>= 180) to avoid dark Io leading to purple-black output.
    """
    img = _to_float(rgb)
    if mask is None:
        mask = tissue_mask(rgb)
    if not np.any(mask):
        # fallback: use overall percentiles
        vals = np.percentile(img.reshape(-1, 3), percentile, axis=0)
    else:
        vals = np.percentile(img[mask], percentile, axis=0)
    vals = np.clip(vals, 180.0, 255.0)  # lower bound prevents under-illumination
    return vals

# -----------------------------
# Optical density transforms
# -----------------------------

def rgb2od(rgb: np.ndarray, io: np.ndarray) -> np.ndarray:
    """
    Convert RGB to optical density (OD). io is the illumination vector per channel.
    OD = -log( (I + 1) / Io ), adding +1 to avoid log(0).
    """
    img = _to_float(rgb)
    io = io.reshape(1, 1, 3)
    od = -np.log((img + 1.0) / (io + _EPS))
    return od

def od2rgb(od: np.ndarray, io: np.ndarray) -> np.ndarray:
    """Inverse OD to RGB: I = Io * exp(-OD)."""
    io = io.reshape(1, 1, 3)
    I = io * np.exp(-od)
    I = np.clip(I, 0.0, 255.0)
    return I.astype(np.uint8)

# -----------------------------
# Macenko basis extraction
# -----------------------------

def macenko_basis(od: np.ndarray, beta: float = 0.15, alpha: float = 0.1) -> np.ndarray:
    """
    Extract 2x stain basis (3x2) from OD using Macenko method.
    - Keep "strong" pixels whose OD norm > beta.
    - SVD to get top-2 PCs (U: 3x2), project OD to 2D PC plane.
    - Take angle extremes at alpha/1-alpha quantiles, map back to 3D, unit normalize.
    Returns H (3x2) with unit columns. Column order is auto-sorted (Hematoxylin first, Eosin second) using a simple heuristic (blue-heavy first).
    """
    X = od.reshape(-1, 3)
    norms = np.linalg.norm(X, axis=1)
    strong = X[norms > beta]
    if strong.shape[0] < 500:  # too few pixels; relax beta
        strong = X[np.argsort(norms)[-min(2000, X.shape[0]):]]
    # Center (mean subtraction)
    mu = strong.mean(axis=0, keepdims=True)
    Z = strong - mu
    # SVD on covariance
    U, S, Vt = np.linalg.svd(np.cov(Z.T), full_matrices=True)
    # top-2 PC in 3D
    P = U[:, :2]  # 3x2
    # project
    Y = Z @ P   # (N,2)
    # angle distribution
    ang = np.arctan2(Y[:, 1], Y[:, 0])
    a_min = np.percentile(ang, alpha * 100.0)
    a_max = np.percentile(ang, (1.0 - alpha) * 100.0)
    v1 = np.array([np.cos(a_min), np.sin(a_min)])
    v2 = np.array([np.cos(a_max), np.sin(a_max)])
    # back to 3D and unit
    h1 = _unit(P @ v1)
    h2 = _unit(P @ v2)
    H = np.stack([h1, h2], axis=1)  # 3x2
    # order: hematoxylin first (blue-heavy), eosin second (red-heavy)
    # Heuristic: compare blue vs red components
    def is_hema(col):
        return col[2] >= col[0]  # blue >= red -> hematoxylin-like
    if not is_hema(H[:, 0]) and is_hema(H[:, 1]):
        H = H[:, [1, 0]]
    elif not is_hema(H[:, 0]) and not is_hema(H[:, 1]):
        # both look eosin-like; fallback to ordering by which has larger blue component
        if H[2, 1] > H[2, 0]:
            H = H[:, [1, 0]]
    return H

def concentrations_nn(od: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Solve non-negative concentrations C from OD ≈ C @ H.T (shape: OD (N,3), H (3,2), C (N,2)).
    We use least squares then clip to non-negative; empirically stable and very fast.
    """
    Ht = H.T  # 2x3
    inv = np.linalg.pinv(Ht @ H) @ Ht   # (2x3)
    C = (od.reshape(-1, 3) @ inv.T)    # (N,2)
    C = np.clip(C, 0.0, None)
    return C

def reconstruct_from_C(C: np.ndarray, H_ref: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    """Reconstruct OD from concentrations C and reference basis H_ref (3x2)."""
    od_rec = C @ H_ref.T  # (N,3)
    H, W = shape_hw
    return od_rec.reshape(H, W, 3)

# -----------------------------
# Main pipeline
# -----------------------------

def normalize_macenko(
    rgb: np.ndarray,
    ref: Dict[str, Any],
    beta: float = 0.15,
    alpha: float = 0.1,
    clamp: float = 1.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """
    Full Macenko normalization for a single RGB tile.

    Parameters
    ----------
    rgb : np.ndarray
        uint8 RGB image.
    ref : dict
        Reference dict with keys:
        - 'H_ref' (3x2 list) unit stain basis for target
        - 'Io_ref' (length-3 list) per-channel illumination for target
        Optionally:
        - 'alpha', 'beta', 'io_percentile' metadata.
    beta, alpha : float
        Macenko thresholds (strong OD norm threshold; angle quantile).
    clamp : float
        Clamp factor on concentrations (1.0 disables; >1.0 shrinks extremes slightly).
    gamma : float
        Gamma correction on final RGB (1.0 disables).

    Returns
    -------
    np.ndarray
        uint8 normalized RGB image.
    """
    H_ref = np.asarray(ref["H_ref"], dtype=np.float64)
    Io_ref = np.asarray(ref["Io_ref"], dtype=np.float64)
    # Estimate Io from this tile for OD conversion
    mask = tissue_mask(rgb)
    Io_tile = estimate_io(rgb, mask=mask, percentile=float(ref.get("io_percentile", 95)))
    # Forward OD
    od = rgb2od(rgb, Io_tile)
    # Per-tile basis (to compute concentrations in native space)
    H_tile = macenko_basis(od, beta=beta, alpha=alpha)
    # Concentrations
    C = concentrations_nn(od, H_tile)
    if clamp > 1.0:
        # shrink extremes by dividing with clamp factor on high quantiles
        q_hi = np.percentile(C, 99.5, axis=0)
        C = np.minimum(C, (q_hi / clamp))
    # Reconstruct with reference basis and Io_ref
    od_rec = reconstruct_from_C(C, H_ref, rgb.shape[:2])
    rgb_norm = od2rgb(od_rec, Io_ref)
    if abs(gamma - 1.0) > 1e-3:
        rgb_norm = np.clip(255.0 * (rgb_norm.astype(np.float32) / 255.0) ** (1.0 / gamma), 0, 255).astype(np.uint8)
    return rgb_norm

# -----------------------------
# JSON IO
# -----------------------------

def save_ref_json(path: str, H_ref: np.ndarray, Io_ref: np.ndarray, meta: Optional[Dict[str, Any]] = None) -> None:
    d = {
        "H_ref": H_ref.tolist(),
        "Io_ref": Io_ref.tolist(),
    }
    if meta:
        d.update(meta)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)

def load_ref_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
