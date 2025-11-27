import os
from random import randint
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

from cvpr_compare import (
    rank_nearest_neighbors, 
    distance_to_confidence,
)
from image_handling import (
    visualize_query_results,
    mat_path_to_image_id,
    load_descriptor_bank,
)
from evals import init_evaluator

DEFAULT_DESCRIPTOR_SUBFOLDER = 'globalRGBhisto'
DEFAULT_TOP_K_RESULTS = 15
DISPLAY_COLS = 4
GRID_PADDING = 10

def main():
    """Main function to perform visual search and evaluation"""
    # Set up paths and load descriptors
    DEFAULT_BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    BASE_PATH = os.getenv("BASE_PATH", DEFAULT_BASE_PATH)
    print(f"BASE_PATH: {BASE_PATH}")

    descriptor_folder = os.path.join(BASE_PATH, 'descriptors')
    descriptor_subfolder = os.getenv(
        "DESCRIPTOR_SUBFOLDER",
        DEFAULT_DESCRIPTOR_SUBFOLDER
    )
    top_k_results = int(os.getenv("TOP_K_RESULTS", str(DEFAULT_TOP_K_RESULTS)))

    allfeat, allfiles = load_descriptor_bank(descriptor_folder, descriptor_subfolder)

    evaluator, reporter = init_evaluator(BASE_PATH)

    if evaluator:
        print(f"[Eval] Loaded {len(evaluator.ground_truth)} ground-truth labels "
              f"across {len(evaluator.class_names)} classes.")

    nimg = allfeat.shape[0]
    query_to_visualize = randint(0, nimg - 1)
    visualization_matches = None
    evaluated_queries = 0
    recorded_predictions = 0

    for query_idx in range(nimg):
        ranked_matches = rank_nearest_neighbors(allfeat, query_idx)
        top_matches = ranked_matches[:top_k_results]

        if query_idx == query_to_visualize:
            visualization_matches = top_matches

        if not evaluator:
            continue

        query_image_id = mat_path_to_image_id(allfiles[query_idx])
        query_true_class = evaluator.ground_truth.get(query_image_id)
        if not query_true_class:
            continue

        predictions_for_eval = []
        for distance, candidate_idx in top_matches:
            image_id = mat_path_to_image_id(allfiles[candidate_idx])
            confidence = distance_to_confidence(distance)
            predictions_for_eval.append((image_id, query_true_class, confidence))

        if not predictions_for_eval:
            continue

        evaluator.add_batch_results(predictions_for_eval)
        evaluated_queries += 1
        recorded_predictions += len(predictions_for_eval)

        if evaluated_queries % 50 == 0:
            print(f"[Eval] Processed {evaluated_queries}/{nimg} queries with ground truth.")

    if visualization_matches:
        visualize_query_results(
            query_to_visualize,
            visualization_matches,
            allfiles,
            cols=DISPLAY_COLS,
            padding=GRID_PADDING
        )
    else:
        print("Warning: Unable to visualize sample query; no images loaded.")

    if evaluator and evaluator.results:
        print(f"[Eval] Recorded {recorded_predictions} predictions from "
              f"{evaluated_queries} queries.")
        try:
            metrics = evaluator.evaluate()
        except ValueError as err:
            print(f"[Eval] Unable to compute metrics: {err}")
        else:
            print(reporter.generate_report(metrics))
            try:
                reporter.plot_confusion_matrix(metrics)
            except Exception as err:
                print(f"[Eval] Could not render confusion matrix: {err}")

            results_path = os.path.join(BASE_PATH, "evaluation_results.csv")
            try:
                reporter.save_detailed_results(results_path)
                print(f"[Eval] Detailed results saved to {results_path}")
            except Exception as err:
                print(f"[Eval] Could not save detailed results: {err}")
    elif evaluator:
        print("[Eval] No overlapping ground-truth entries found; nothing to evaluate.")

if __name__ == "__main__":
    main()
