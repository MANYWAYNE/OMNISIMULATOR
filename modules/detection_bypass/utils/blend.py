# Color quantization / region blending

import numpy as np
from PIL import Image

try:
    from sklearn.cluster import MiniBatchKMeans
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


def blend_colors(
    image: np.ndarray,
    tolerance: float = 10.0,
    min_region_size: int = 50,
    max_kmeans_samples: int = 100_000,
) -> np.ndarray:
    """
    Color-region quantization that breaks the smooth gradients in AI images.
    Falls back gracefully if sklearn is unavailable.
    """
    if not SKLEARN_OK:
        # Simplified fallback: PIL quantize
        pil = Image.fromarray(image).quantize(colors=max(8, int(256 / max(tolerance / 10, 1)))).convert('RGB')
        return np.array(pil)

    H, W, C = image.shape
    flat = image.reshape(-1, C).astype(np.float32)

    # Sample pixels for clustering
    n = min(max_kmeans_samples, flat.shape[0])
    idx = np.random.choice(flat.shape[0], n, replace=False)
    sample = flat[idx]

    # Determine K from tolerance (lower tolerance → more clusters)
    k = max(2, int(256 / max(tolerance, 1)))
    k = min(k, 128)

    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3)
    km.fit(sample)

    # Assign all pixels
    labels = km.predict(flat).reshape(H, W)
    centers = km.cluster_centers_

    # Paint regions (ignoring min_region_size for now to keep it fast)
    out = centers[labels].reshape(H, W, C)
    return np.clip(out, 0, 255).astype(np.uint8)