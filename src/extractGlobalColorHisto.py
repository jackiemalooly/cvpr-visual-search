import numpy as np

def rgb_histogram(img, Q: int) -> np.ndarray:
    # Quantize each color channel into Q divisions
    # Note: img is expected to be a normalized RGB image (colors range [0,1] not [0,255])
    qimg = np.floor(img * Q).astype(int)

    # Create a single integer value for each pixel that summarizes its RGB values
    bin = (qimg[:, :, 0] * Q**2 +
               qimg[:, :, 1] * Q +
               qimg[:, :, 2])

    # Flatten into a long vector of values and accumulate histogram with Q^3 bins
    vals = bin.ravel()
    H, _ = np.histogram(vals, bins=np.arange(Q**3 + 1), density=False)

    # Normalize to sum to 1
    return H / H.sum()