import os
import math
import numpy as np
import scipy.io as sio
import cv2
from random import randint
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

from cvpr_compare import cvpr_compare
from image_handling import descriptor_to_image_path, load_and_fit_image, add_label

def main():
    """Main function to perform visual search and evaluation"""
    # Set up paths and load descriptors
    DEFAULT_BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    BASE_PATH = os.getenv("BASE_PATH", DEFAULT_BASE_PATH)
    print(f"BASE_PATH: {BASE_PATH}")

    DESCRIPTOR_FOLDER = os.path.join(BASE_PATH, 'descriptors')
    DESCRIPTOR_SUBFOLDER = 'globalRGBhisto'
    
    ALLFEAT = []
    ALLFILES = []
    descriptor_dir = os.path.join(DESCRIPTOR_FOLDER, DESCRIPTOR_SUBFOLDER)
    for filename in os.listdir(descriptor_dir):
        if filename.endswith('.mat'):
            img_path = os.path.join(descriptor_dir, filename)
            img_data = sio.loadmat(img_path)
            ALLFILES.append(img_path)
            ALLFEAT.append(img_data['F'][0])  # Assuming F is a 1D array

    # Convert ALLFEAT to a numpy array
    ALLFEAT = np.array(ALLFEAT)

    # Pick a random image as the query
    NIMG = ALLFEAT.shape[0]
    queryimg = randint(0, NIMG - 1)

    # Compute the distance between the query and all other descriptors
    dst = []
    query = ALLFEAT[queryimg]
    for i in range(NIMG):
        candidate = ALLFEAT[i]
        distance = cvpr_compare(query, candidate)
        dst.append((distance, i))

    # Sort the distances
    dst.sort(key=lambda x: x[0])

    SHOW = 15

    # Prepare list of result paths, skipping the query itself
    result_img_paths = []
    for distance, idx in dst:
        if idx == queryimg:
            continue
        result_mat_path = ALLFILES[idx]
        result_img_paths.append(descriptor_to_image_path(result_mat_path))
        if len(result_img_paths) == SHOW:
            break

    # Load images and build labeled cells
    cells = []
    for rank, img_path in enumerate(result_img_paths, start=1):
        cell = load_and_fit_image(img_path)
        if cell is None:
            print(f"Warning: Could not load result image at {img_path}")
            continue
        cells.append(add_label(cell, f"Rank {rank}"))

    if not cells:
        raise RuntimeError("No images could be loaded to display.")

    # Place query image at the top of the cells and label as "Query"
    query_cell = load_and_fit_image(descriptor_to_image_path(ALLFILES[queryimg]))
    query_cell = add_label(query_cell, "Query")
    cells.insert(0, query_cell) 

    # Arrange cells into a grid for a single window display
    cols = 4
    rows = math.ceil(len(cells) / cols)
    cell_h, cell_w = cells[0].shape[0], cells[0].shape[1]
    padding = 10
    grid_h = rows * cell_h + (rows + 1) * padding
    grid_w = cols * cell_w + (cols + 1) * padding
    grid = np.full((grid_h, grid_w, 3), (20, 20, 20), dtype=np.uint8)

    for idx, cell in enumerate(cells):
        row = idx // cols
        col = idx % cols
        y = padding + row * (cell_h + padding)
        x = padding + col * (cell_w + padding)
        grid[y:y + cell_h, x:x + cell_w] = cell

    cv2.imshow("Query and Top Results", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
