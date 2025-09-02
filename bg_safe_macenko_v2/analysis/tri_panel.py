
import numpy as np
import matplotlib.pyplot as plt
from skimage import color

def deltaE_lab(img_a, img_b, mask=None):
    """Compute ΔE76 in Lab; if mask provided, set outside to NaN."""
    a = img_a.astype(np.float32) / 255.0
    b = img_b.astype(np.float32) / 255.0
    La = color.rgb2lab(a)
    Lb = color.rgb2lab(b)
    dE = np.sqrt(((La - Lb) ** 2).sum(axis=-1))
    if mask is not None:
        out = np.full_like(dE, np.nan, dtype=np.float32)
        out[mask] = dE[mask]
        return out
    return dE

def tri_panel(original, normalized, dE, save_path):
    """Save a tri-panel figure with Original / Normalized / ΔE."""
    fig = plt.figure(figsize=(15,5), dpi=200)
    ax1 = plt.subplot(1,3,1)
    ax2 = plt.subplot(1,3,2)
    ax3 = plt.subplot(1,3,3)

    ax1.imshow(original)
    ax1.set_title("Original")
    ax1.axis('off')

    ax2.imshow(normalized)
    ax2.set_title("Macenko normalized")
    ax2.axis('off')

    im = ax3.imshow(dE, cmap='magma')
    ax3.set_title("ΔE (Lab)")
    ax3.axis('off')
    cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label("ΔE")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def dE_stats(dE, mask=None):
    """Return a dict of summary stats for ΔE over tissue mask."""
    if mask is not None:
        vals = dE[mask]
    else:
        vals = dE[~np.isnan(dE)]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return dict(mean=np.nan, std=np.nan, p95=np.nan, p99=np.nan)
    return dict(
        mean=float(np.mean(vals)),
        std=float(np.std(vals)),
        p95=float(np.percentile(vals,95)),
        p99=float(np.percentile(vals,99)),
        n=int(vals.size),
    )
