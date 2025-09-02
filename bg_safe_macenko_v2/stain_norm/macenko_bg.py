
import numpy as np
import cv2
from skimage import morphology
from scipy.ndimage import distance_transform_edt, gaussian_filter

def _ensure_uint8(img):
    if img.dtype == np.uint8:
        return img
    if img.max() <= 1.0:
        img = (img * 255.0).round().astype(np.uint8)
    else:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def tissue_mask(img_rgb, sat_thr=0.05, val_thr=0.95, od_bg_thr=0.15, min_size=256):
    """
    Robust tissue mask combining HSV and optical density (OD) gating.
    Returns (mask, Io) where Io is the 99th percentile white estimate per channel.
    """
    img = _ensure_uint8(img_rgb)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    S = hsv[..., 1] / 255.0
    V = hsv[..., 2] / 255.0
    hsv_mask = (S > sat_thr) & (V < val_thr)

    Io = np.percentile(img.reshape(-1, 3), 99, axis=0).astype(np.float32)
    Io = np.clip(Io, 220, 255)  # safeguard against low white estimate
    OD = -np.log((img.astype(np.float32) + 1.0) / (Io + 1.0))
    od_sum = OD.sum(axis=-1)
    od_mask = od_sum > od_bg_thr

    m = hsv_mask & od_mask
    # clean up small speckles and holes
    m = morphology.remove_small_objects(m, min_size=min_size)
    m = morphology.remove_small_holes(m, area_threshold=min_size)
    m = morphology.binary_opening(m, morphology.disk(1))
    return m, Io

def background_layer(img_rgb, policy='keep', Io_ref=(255, 255, 255), sigma=15):
    """
    Produce a background layer according to the chosen policy:
      - 'keep': keep original RGB background
      - 'ref' : fill with Io_ref (reference white)
      - 'local': per-image 99th percentile with smoothing to keep illumination gradients
    """
    img = img_rgb.astype(np.float32)
    if policy == 'keep':
        return img
    if policy == 'ref':
        return np.ones_like(img) * np.array(Io_ref, dtype=np.float32)
    if policy == 'local':
        Io_local = np.percentile(img.reshape(-1, 3), 99, axis=0)
        bg = np.ones_like(img) * Io_local
        for c in range(3):
            bg[..., c] = gaussian_filter(bg[..., c], sigma=sigma)
        return bg
    raise ValueError(f'Unknown bg policy: {policy}')

def _parse_ref_json(ref_json):
    """
    Load reference parameters:
      {
        "W_ref": [[...],[...],[...]],  # 3x2 with columns as H/E vectors in OD space
        "conc_p": [1, 99],             # percentile stretch for concentrations
        "Io_ref": [255,255,255]        # reference white
      }
    """
    if ref_json is None:
        return None, (1, 99), (255, 255, 255)
    with open(ref_json, 'r') as f:
        cfg = json.load(f)
    W_ref = cfg.get('W_ref', None)
    if W_ref is not None:
        W_ref = np.asarray(W_ref, dtype=np.float32)
        if W_ref.shape == (2,3):
            W_ref = W_ref.T
    conc_p = tuple(cfg.get('conc_p', (1, 99)))
    Io_ref = tuple(cfg.get('Io_ref', (255, 255, 255)))
    return W_ref, conc_p, Io_ref

def macenko_normalize(img_rgb, W_ref, conc_p=(1, 99), Io_ref=(255, 255, 255),
                      mask=None, bg_policy='keep', edge_soft_px=4, od_bg_thr=0.15):
    """
    Background-safe Macenko normalization.
    Args:
        img_rgb: np.uint8 RGB image
        W_ref: 3x2 reference stain matrix (columns are H & E in OD space)
        conc_p: (p_low, p_high) percentiles for concentration stretch
        Io_ref: target white RGB
        mask: boolean tissue mask; if None, computed internally
        bg_policy: 'keep' | 'ref' | 'local'
        edge_soft_px: soft edge width in pixels
        od_bg_thr: OD-sum threshold to zero out background OD
    Returns:
        out_rgb (uint8), mask (bool)
    """
    img = _ensure_uint8(img_rgb)
    H, W, _ = img.shape

    # Background visual layer
    bg = background_layer(img, policy=bg_policy, Io_ref=Io_ref)

    # Mask + soft edge
    if mask is None:
        mask, Io_est = tissue_mask(img)
    else:
        Io_est = np.percentile(img.reshape(-1, 3), 99, axis=0).astype(np.float32)

    dist = distance_transform_edt(mask)
    alpha = np.clip(dist / float(edge_soft_px), 0, 1)  # inside tissue ~1, edges taper

    # Optical density with background clamped to 0
    OD = -np.log((img.astype(np.float32) + 1.0) / (Io_est + 1.0))
    od_sum = OD.sum(axis=-1)
    OD[od_sum < od_bg_thr] = 0.0

    # Flatten for linear least squares
    M = OD.reshape(-1, 3).T  # 3 x N
    # Solve concentrations with non-negativity by clamping
    # W_ref is 3x2; compute H (2 x N)
    Hc, _, _, _ = np.linalg.lstsq(W_ref, M, rcond=None)
    Hc = np.maximum(Hc, 0.0)  # 2 x N

    # Stretch concentrations using percentiles over tissue only
    H1 = Hc[0, :].reshape(H, W)
    H2 = Hc[1, :].reshape(H, W)
    # Avoid empty masks
    if mask.sum() > 0:
        p1, p99 = np.percentile(H1[mask], conc_p[0]), np.percentile(H1[mask], conc_p[1])
        e1, e99 = np.percentile(H2[mask], conc_p[0]), np.percentile(H2[mask], conc_p[1])
    else:
        p1, p99, e1, e99 = 0.0, 1.0, 0.0, 1.0

    H1n = np.clip((H1 - p1) / (p99 - p1 + 1e-6), 0, 1)
    H2n = np.clip((H2 - e1) / (e99 - e1 + 1e-6), 0, 1)

    # Reconstruct OD only inside tissue
    OD_hat = np.zeros_like(OD)
    Hn = np.stack([H1n, H2n], axis=-1)  # HxWx2
    OD_hat[mask] = (W_ref @ Hn[mask].T).T

    # Soft edge blending in OD space
    OD_hat = OD_hat * alpha[..., None]

    # Back to RGB using reference white
    Io_ref = np.array(Io_ref, dtype=np.float32)
    out = Io_ref * np.exp(-OD_hat)
    out = np.clip(out, 0, 255)

    # Compose with background layer (alpha already 0 outside mask)
    out_rgb = out + bg * (1 - alpha[..., None])
    return out_rgb.astype(np.uint8), mask
