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
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
load_dotenv()

from cvpr_compare import cvpr_compare
from image_handling import (
    descriptor_to_image_path,
    load_and_fit_image,
    add_label,
    load_ground_truth_labels,
    mat_path_to_image_id,
    extract_class_from_image_id,
)
from evals import (
    compute_precision_recall_at_k,
    build_confusion_matrix,
    render_confusion_matrix_image,
    show_confusion_matrix_image,
    _collect_class_labels,
)


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
    EXPERIMENT_NAME = 'baseline_RGBhisto'
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

    print("--------------------------------")
    print("Displaying queries and top results...")
    print("--------------------------------")
    cv2.imshow("Queries and Top Results", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Saving queries and top results to {os.path.join(BASE_PATH, f"queries_and_top_results_{EXPERIMENT_NAME}.png")}")
    cv2.imwrite(os.path.join(BASE_PATH, f"queries_and_top_results_{EXPERIMENT_NAME}.png"), grid)

    # Persist and plot precision/recall information when available
    print("--------------------------------")
    print("Persisting precision/recall statistics...")
    stats_with_data = []
    for query_idx, stats in precision_recall_stats_all:
        if not stats:
            continue
        query_id = mat_path_to_image_id(ALLFILES[query_idx])
        query_class = extract_class_from_image_id(query_id)
        enriched_stats = []
        for entry in stats:
            candidate_class = extract_class_from_image_id(entry["candidate_id"])
            enriched_stats.append({
                **entry,
                "candidate_class": candidate_class,
            })
        stats_with_data.append({
            "query_index": query_idx,
            "query_id": query_id,
            "query_class": query_class,
            "stats": enriched_stats,
        })

    if not stats_with_data:
        print("No precision/recall statistics available. Skipping CSV export and PR curve.")
        return

    class_labels = _collect_class_labels(stats_with_data)

    results_dir = os.path.join(BASE_PATH, "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"precision_recall_stats_{EXPERIMENT_NAME}.csv")

    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = [
            "query_index",
            "query_id",
            "query_class",
            "rank",
            "candidate_id",
            "candidate_class",
            "precision",
            "recall",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in stats_with_data:
            for stat in entry["stats"]:
                writer.writerow({
                    "query_index": entry["query_index"],
                    "query_id": entry["query_id"],
                    "query_class": entry["query_class"],
                    "rank": stat["n"],
                    "candidate_id": stat["candidate_id"],
                    "candidate_class": stat["candidate_class"],
                    "precision": stat["precision"],
                    "recall": stat["recall"],
                })

    print(f"Precision/recall statistics saved to {csv_path}")

    # Calculate Average Precision (AP) and Mean Average Precision (MAP)
    ap = 0
    map = 0
    for entry in stats_with_data:
        for stat in entry["stats"]:
            ap += stat["precision"] * stat["recall"]
    if stats_with_data:
        ap /= len(stats_with_data)
    map = ap / len(stats_with_data) if stats_with_data else 0
    print(f"Average Precision (AP): {ap}")
    print(f"Mean Average Precision (MAP): {map}")
    # Save AP and MAP to a new CSV file
    ap_map_csv_path = os.path.join(results_dir, f"ap_map_stats_{EXPERIMENT_NAME}.csv")
    with open(ap_map_csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["ap", "map"])
        writer.writeheader()
        writer.writerow({"ap": ap, "map": map})
    print(f"AP and MAP saved to {ap_map_csv_path}")

    # Build averaged precision-recall curve across all queries
    max_rank = max(len(entry["stats"]) for entry in stats_with_data)
    avg_precisions = []
    avg_recalls = []
    for rank in range(1, max_rank + 1):
        precisions_at_rank = []
        recalls_at_rank = []
        for entry in stats_with_data:
            stats = entry["stats"]
            if len(stats) >= rank:
                precisions_at_rank.append(stats[rank - 1]["precision"])
                recalls_at_rank.append(stats[rank - 1]["recall"])
        if precisions_at_rank and recalls_at_rank:
            avg_precisions.append(sum(precisions_at_rank) / len(precisions_at_rank))
            avg_recalls.append(sum(recalls_at_rank) / len(recalls_at_rank))

    print("Plotting averaged precision-recall curve...")
    plt.figure(figsize=(8, 6))
    plt.plot(avg_recalls, avg_precisions)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Average Precision-Recall Curve Across Queries')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"pr_curve_{EXPERIMENT_NAME}.png"))
    plt.show()

    print("--------------------------------")
    print("Building confusion matrices...")
    cm_specs = [
        ("Top-1", 1, f"confusion_matrix_top1_{EXPERIMENT_NAME}.png"),
        ("Top-5", 5, f"confusion_matrix_top5_{EXPERIMENT_NAME}.png"),
        (f"Top-{SHOW}", SHOW, f"confusion_matrix_top{SHOW}_{EXPERIMENT_NAME}.png"),
    ]

    for label, top_k, filename in cm_specs:
        cm, labels = build_confusion_matrix(
            stats_with_data,
            top_k=top_k,
            class_labels=class_labels
        )
        cm_image = render_confusion_matrix_image(cm, labels)
        save_path = os.path.join(results_dir, filename)
        cv2.imwrite(save_path, cm_image)
        print(f"{label} confusion matrix saved to {save_path}")
        show_confusion_matrix_image(cm_image, f"{label} Confusion Matrix")

if __name__ == "__main__":
    main()
