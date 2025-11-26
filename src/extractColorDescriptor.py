import numpy as np

def extract_color_descriptor(img):
    # Compute the average red, green, and blue values as a basic color descriptor
    R = np.mean(img[:, :, 2])  # Note: OpenCV uses BGR format
    G = np.mean(img[:, :, 1])
    B = np.mean(img[:, :, 0])
    return np.array([R, G, B])