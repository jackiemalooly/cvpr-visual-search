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
from image_handling import (
    descriptor_to_image_path,
    load_and_fit_image,
    add_label,
    load_ground_truth_labels,
)
from evals import compute_precision_recall_at_k

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
    ground_truth = load_ground_truth_labels(BASE_PATH)

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
    BASE_CELL_SIZE = (200, 150)
    LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
    LABEL_SCALE = 0.5
    LABEL_THICKNESS = 1
    LABEL_PADDING = 12

    # Prepare top matches, skipping the query itself
    top_matches = []
    for distance, idx in dst:
        if idx == queryimg:
            continue
        top_matches.append((distance, idx))
        if len(top_matches) == SHOW:
            break

    precision_recall_stats = compute_precision_recall_at_k(
        top_matches, queryimg, ALLFILES, ground_truth
    )

    if precision_recall_stats:
        print("\nPrecision/Recall over top results:")
        print(f"{'n':>3} {'Result':>12} {'Precision':>10} {'Recall':>10}")
        for stat in precision_recall_stats:
            print(f"{stat['n']:>3} {stat['candidate_id']:>12} "
                  f"{stat['precision']:>10.3f} {stat['recall']:>10.3f}")

    # Build label text ahead of time so cells wide enough
    labeled_matches = []
    label_texts = ["Query"]
    for rank, (_, idx) in enumerate(top_matches, start=1):
        label = f"Rank {rank}"
        if precision_recall_stats and len(precision_recall_stats) >= rank:
            pr = precision_recall_stats[rank - 1]
            label += f" | P={pr['precision']:.2f} R={pr['recall']:.2f}"
        label_texts.append(label)
        labeled_matches.append((idx, label))

    def label_pixel_width(text: str) -> int:
        (text_w, _), _ = cv2.getTextSize(
            text,
            LABEL_FONT,
            LABEL_SCALE,
            LABEL_THICKNESS
        )
        return text_w

    max_label_width = max(label_pixel_width(text) for text in label_texts)
    cell_width = max(BASE_CELL_SIZE[0], max_label_width + LABEL_PADDING)
    cell_size = (cell_width, BASE_CELL_SIZE[1])

    # Load images and build labeled cells
    cells = []
    for idx, label in labeled_matches:
        img_path = descriptor_to_image_path(ALLFILES[idx])
        cell = load_and_fit_image(img_path, cell_size=cell_size)
        if cell is None:
            print(f"Warning: Could not load result image at {img_path}")
            continue
        cells.append(add_label(cell, label))

    if not cells:
        raise RuntimeError("No images could be loaded to display.")

    # Place query image at the top of the cells and label as "Query"
    query_cell = load_and_fit_image(
        descriptor_to_image_path(ALLFILES[queryimg]),
        cell_size=cell_size
    )
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
