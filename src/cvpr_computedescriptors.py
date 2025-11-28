import os
import numpy as np
import cv2
import scipy.io as sio
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

from extractRandom import extractRandom as extract_random
from extractColorDescriptor import extract_color_descriptor

DEFAULT_BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_PATH = os.getenv("BASE_PATH", DEFAULT_BASE_PATH)
print(f"BASE_PATH: {BASE_PATH}")

DATASET_FOLDER = os.path.join(BASE_PATH, 'MSRC_ObjCategImageDatabase_v2')
OUT_FOLDER = os.path.join(BASE_PATH, 'descriptors')
OUT_SUBFOLDER = 'globalRGBhisto'


def main() -> None:
    images_dir = os.path.join(DATASET_FOLDER, 'Images')
    os.makedirs(os.path.join(OUT_FOLDER, OUT_SUBFOLDER), exist_ok=True)

    # Iterate through all BMP files in the dataset folder
    for filename in os.listdir(images_dir):
        if not filename.endswith(".bmp"):
            continue

        print(f"Processing file {filename}")
        img_path = os.path.join(images_dir, filename)
        img = cv2.imread(img_path).astype(np.float64) / 255.0  # Normalize the image
        fout = os.path.join(OUT_FOLDER, OUT_SUBFOLDER, filename.replace('.bmp', '.mat'))

        # Call extractRandom (or another feature extraction function) to get the descriptor
        F = extract_color_descriptor(img)

        # Save the descriptor to a .mat file
        sio.savemat(fout, {'F': F})


if __name__ == "__main__":
    main()

