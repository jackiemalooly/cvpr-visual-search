import os
import math
import csv
import numpy as np
import scipy.io as sio
import cv2
import matplotlib.pyplot as plt
from random import sample
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
    mat_path_to_image_id,
)
from evals import compute_precision_recall_at_k

def main():
    """Main function to perform visual search and evaluation"""
    # Set up paths and load descriptors
    print("Starting visual search...")
    print("--------------------------------")
    print("Loading data...")
    DEFAULT_BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    BASE_PATH = os.getenv("BASE_PATH", DEFAULT_BASE_PATH)
    print(f"BASE_PATH: {BASE_PATH}")

    DESCRIPTOR_FOLDER = os.path.join(BASE_PATH, 'descriptors')
    DESCRIPTOR_SUBFOLDER = 'globalRGBhisto'

    # Constants
    SHOW = 15
    BASE_CELL_SIZE = (200, 150)
    DISPLAY_SCALE = 0.55

    # Load up the data
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

    # Pick several images as the queries
    print("--------------------------------")
    print("Picking queries...")
    NUM_QUERIES = 10
    NIMG = ALLFEAT.shape[0]
    QUERY_IMGS = sample(range(NIMG), NUM_QUERIES) # using random.sample to avoid duplicates
    print(f"Picked {NUM_QUERIES} queries: {QUERY_IMGS}")
    # For each query image, compute the distance between the query and all other descriptors and store the results in a list of tuples: (queryimg, dst)
    # dst is a list of tuples: (distance, idx)
    # [(query_id0, [(dist, idx), ...]), (query_id1, [...]), ...]
    print("--------------------------------")
    print("Processing queries...")
    dst_all = []
    top_matches_all = []
    precision_recall_stats_all = []
    for queryimg in QUERY_IMGS:
        print(f"--------------------------------")
        print(f"Computing distances for query {queryimg}")
        dst = []
        query = ALLFEAT[queryimg]
        for i in range(NIMG):
            candidate = ALLFEAT[i]
            distance = cvpr_compare(query, candidate)
            dst.append((distance, i))
        # Sort the distances
        dst.sort(key=lambda x: x[0])
        dst_all.append((queryimg, dst))

        # Prepare top matches and compute precision and recall, skipping the query itself
        print(f"Preparing {SHOW} top matches and precision and recall stats for query {queryimg}...")
        print("--------------------------------")
        top_matches = []
        for distance, idx in dst:
            if idx == queryimg:
                continue
            top_matches.append((distance, idx))
            if len(top_matches) == SHOW:
                break
        top_matches_all.append((queryimg, top_matches))

        precision_recall_stats = compute_precision_recall_at_k(
            top_matches, queryimg, ALLFILES, ground_truth
        )
        precision_recall_stats_all.append((queryimg, precision_recall_stats))


    # Build a grid where each row starts with the query image and is followed by its matches
    rows_of_cells = []
    max_cols = 0
    cell_size = tuple(max(40, int(dim * DISPLAY_SCALE)) for dim in BASE_CELL_SIZE)
    padding = max(4, int(10 * DISPLAY_SCALE))
    for queryimg, top_matches in top_matches_all:
        row_cells = []
        query_cell = load_and_fit_image(
            descriptor_to_image_path(ALLFILES[queryimg]),
            cell_size=cell_size
        )
        query_cell = add_label(query_cell, "Query")
        row_cells.append(query_cell)

        for rank, (_, match_idx) in enumerate(top_matches, start=1):
            match_cell = load_and_fit_image(
                descriptor_to_image_path(ALLFILES[match_idx]),
                cell_size=cell_size
            )
            row_cells.append(match_cell)

        max_cols = max(max_cols, len(row_cells))
        rows_of_cells.append(row_cells)

    if not rows_of_cells:
        print("No query results to display.")
        return

    # Arrange rows into a single window display
    rows = len(rows_of_cells)
    cols = max_cols
    cell_h, cell_w = rows_of_cells[0][0].shape[0], rows_of_cells[0][0].shape[1]
    grid_h = rows * cell_h + (rows + 1) * padding
    grid_w = cols * cell_w + (cols + 1) * padding
    grid = np.full((grid_h, grid_w, 3), (20, 20, 20), dtype=np.uint8)

    for row_idx, row_cells in enumerate(rows_of_cells):
        for col_idx, cell in enumerate(row_cells):
            y = padding + row_idx * (cell_h + padding)
            x = padding + col_idx * (cell_w + padding)
            grid[y:y + cell_h, x:x + cell_w] = cell

    cv2.imshow("Query and Top Results", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Persist and plot precision/recall information when available
    print("--------------------------------")
    print("Persisting precision/recall statistics...")
    stats_with_data = [
        (query_idx, stats)
        for query_idx, stats in precision_recall_stats_all
        if stats
    ]

    if not stats_with_data:
        print("No precision/recall statistics available. Skipping CSV export and PR curve.")
        return

    results_dir = os.path.join(BASE_PATH, "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "precision_recall_stats.csv")

    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ["query_index", "query_id", "rank", "candidate_id", "precision", "recall"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for query_idx, stats in stats_with_data:
            query_id = mat_path_to_image_id(ALLFILES[query_idx])
            for entry in stats:
                writer.writerow({
                    "query_index": query_idx,
                    "query_id": query_id,
                    "rank": entry["n"],
                    "candidate_id": entry["candidate_id"],
                    "precision": entry["precision"],
                    "recall": entry["recall"],
                })

    print(f"Precision/recall statistics saved to {csv_path}")

    # Build averaged precision-recall curve across all queries
    max_rank = max(len(stats) for _, stats in stats_with_data)
    avg_precisions = []
    avg_recalls = []
    for rank in range(1, max_rank + 1):
        precisions_at_rank = []
        recalls_at_rank = []
        for _, stats in stats_with_data:
            if len(stats) >= rank:
                precisions_at_rank.append(stats[rank - 1]["precision"])
                recalls_at_rank.append(stats[rank - 1]["recall"])
        if precisions_at_rank and recalls_at_rank:
            avg_precisions.append(sum(precisions_at_rank) / len(precisions_at_rank))
            avg_recalls.append(sum(recalls_at_rank) / len(recalls_at_rank))

    print("Plotting averaged precision-recall curve...")
    plt.figure(figsize=(8, 6))
    plt.plot(avg_recalls, avg_precisions, marker="o")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Average Precision-Recall Curve Across Queries')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
