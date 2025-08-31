
import numpy as np
import cv2

# ----------------------------
# Macenko stain normalization
# ----------------------------
# Reference:
#  - Macenko et al., 2009. A Method for Normalizing Histology Slides for Quantitative Analysis.
# Implementation notes:
#  - Estimate stain matrix (H) via SVD on optical density (OD) of tissue pixels.
#  - Solve concentrations C by least squares (non-negative optional).
#  - Normalize by mapping C to a target stain matrix H_ref and target max concentrations.

def rgb2od(I, Io=255.0, eps=1e-6):
    I = I.astype(np.float32)
    return -np.log((I + eps) / Io)

def od2rgb(OD, Io=255.0):
    I = (np.exp(-OD) * Io).clip(0, 255).astype(np.uint8)
    return I

def _normalize_rows(A):
    n = np.linalg.norm(A, axis=1, keepdims=True) + 1e-8
    return A / n

def estimate_stain_matrix_macenko(I, Io=255.0, beta=0.15, alpha=1.0):
    """
    Estimate stain matrix H (2x3) using Macenko.
    I: RGB uint8 image
    beta: OD threshold for background filtering
    alpha: percentile of extreme angles (1 ~ 99), default 1%
    Returns H (2x3) with rows = [H, E] normalized to unit norm in OD space.
    """
    if I.ndim != 3 or I.shape[2] != 3:
        raise ValueError("Input must be RGB image.")

    OD = rgb2od(I, Io=Io).reshape((-1, 3))
    # Remove background pixels (low OD ~ white background)
    mask = (OD > beta).any(axis=1)
    OD_filt = OD[mask]
    if OD_filt.shape[0] < 100:
        OD_filt = OD

    # PCA on OD
    U, S, Vt = np.linalg.svd(OD_filt, full_matrices=False)
    V = Vt.T  # 3x3, columns are principal directions

    # Project OD onto first two PCs
    OD2 = OD_filt @ V[:, :2]  # N x 2

    # Angles
    phi = np.arctan2(OD2[:, 1], OD2[:, 0])

    # Extreme angles at alpha and 100-alpha percentiles
    min_phi = np.percentile(phi, alpha)
    max_phi = np.percentile(phi, 100 - alpha)

    v1 = np.array([np.cos(min_phi), np.sin(min_phi)])
    v2 = np.array([np.cos(max_phi), np.sin(max_phi)])

    # Map back to OD space
    HE = np.stack([v1, v2], axis=1)  # 2 x 2
    H = (V[:, :2] @ HE).T            # 2 x 3

    # Normalize rows
    H = _normalize_rows(H.astype(np.float32))
    return H  # rows: H, E

def compute_concentrations(I, H, Io=255.0, non_negative=True):
    """
    Solve for concentrations C given stain matrix H (2x3).
    Returns C: HxW x 2 (concentrations for H & E)
    """
    h, w, _ = I.shape
    OD = rgb2od(I, Io=Io).reshape((-1, 3))

    H_t = H.T  # 3x2
    pinv = np.linalg.pinv(H_t)  # 2x3

    C = (OD @ pinv.T)  # N x 2
    if non_negative:
        C = np.clip(C, 0, None)

    return C.reshape((h, w, 2)).astype(np.float32)

def normalize_macenko(I, H_ref=None, C_ref_max=None, Io=255.0, beta=0.15, alpha=1.0, n_stepsize=50_000):
    """
    Full Macenko normalization pipeline.
    - If H_ref or C_ref_max not provided, use canonical targets (or self).
    - Rescale concentrations by ratio of target 99th percentile (C_ref_max) to this image's 99th percentile.
    Returns normalized RGB image, and a dict with estimated matrices.
    """
    H_src = estimate_stain_matrix_macenko(I, Io=Io, beta=beta, alpha=alpha)
    C_src = compute_concentrations(I, H_src, Io=Io)

    # Compute 99th percentile concentrations (robust range)
    C_src_flat = C_src.reshape((-1, 2))
    if C_src_flat.shape[0] > n_stepsize:
        idx = np.random.choice(C_src_flat.shape[0], n_stepsize, replace=False)
        C_sub = C_src_flat[idx]
    else:
        C_sub = C_src_flat
    C99_src = np.percentile(C_sub, 99, axis=0) + 1e-8

    # Canonical target if no reference provided
    if H_ref is None:
        H_ref = np.array([[0.65, 0.70, 0.29],
                          [0.07, 0.99, 0.11]], dtype=np.float32)
        H_ref = _normalize_rows(H_ref)
    if C_ref_max is None:
        C_ref_max = np.array([1.0, 1.0], dtype=np.float32)

    # Rescale concentrations
    scale = (C_ref_max / C99_src).astype(np.float32)
    C_tgt = (C_src * scale.reshape((1,1,2))).reshape((-1, 2))

    # Recompose
    OD_tgt = C_tgt @ H_ref  # N x 3
    I_norm = od2rgb(OD_tgt.reshape(I.shape), Io=Io)

    meta = {
        "H_src": H_src.tolist(),
        "H_ref": H_ref.tolist(),
        "C99_src": C99_src.tolist(),
        "C_ref_max": C_ref_max.tolist(),
        "scale": scale.tolist(),
    }
    return I_norm, meta

def estimate_reference_from_tiles(tiles, sample_k=64, Io=255.0, beta=0.15, alpha=1.0):
    """
    tiles: list of BGR uint8 tiles (OpenCV convention)
    Returns (H_ref, C_ref_max) estimated from a set of tiles (robust 99th percentile).
    """
    Hs = []
    Cs = []
    for t in tiles[:sample_k]:
        rgb = cv2.cvtColor(t, cv2.COLOR_BGR2RGB)
        H = estimate_stain_matrix_macenko(rgb, Io=Io, beta=beta, alpha=alpha)
        C = compute_concentrations(rgb, H, Io=Io).reshape((-1,2))
        Hs.append(H)
        if C.shape[0] > 10000:
            idx = np.random.choice(C.shape[0], 10000, replace=False)
            C = C[idx]
        Cs.append(C)
    H_ref = _normalize_rows(np.mean(np.stack(Hs, axis=0), axis=0))
    C_all = np.vstack(Cs)
    C_ref_max = np.percentile(C_all, 99, axis=0) + 1e-8
    return H_ref.astype(np.float32), C_ref_max.astype(np.float32)
