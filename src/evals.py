import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Dict
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from image_handling import mat_path_to_image_id

def _sorted_class_labels(labels):
    """Return labels sorted numerically when possible, otherwise lexicographically."""
    def sort_key(label):
        try:
            return (0, int(label))
        except ValueError:
            return (1, label)
    return sorted(labels, key=sort_key)

def _collect_class_labels(stats_with_data: list[dict]) -> list[str]:
    """Collect unique class labels appearing in queries or retrieved candidates."""
    labels = set()
    for entry in stats_with_data:
        query_class = entry.get("query_class")
        if query_class:
            labels.add(query_class)
        for stat in entry.get("stats", []):
            candidate_class = stat.get("candidate_class")
            if candidate_class:
                labels.add(candidate_class)
    return _sorted_class_labels(labels)

def compute_precision_recall_at_k(top_matches: List[tuple],
                                  query_idx: int,
                                  all_files: List[str],
                                  ground_truth: Dict[str, str]):
    """Return per-rank precision/recall stats for the query if GT is available."""
    if not ground_truth:
        return None

    query_id = mat_path_to_image_id(all_files[query_idx])
    query_label = ground_truth.get(query_id)
    if not query_label:
        print(f"[Metrics] Missing ground-truth label for query {query_id}.")
        return None

    relevant_total = sum(1 for label in ground_truth.values() if label == query_label) - 1
    if relevant_total <= 0:
        print(f"[Metrics] Not enough relevant samples to compute recall for class {query_label}.")
        return None

    stats = []
    relevant_found = 0
    for rank, (_, candidate_idx) in enumerate(top_matches, start=1):
        candidate_id = mat_path_to_image_id(all_files[candidate_idx])
        candidate_label = ground_truth.get(candidate_id)
        if candidate_label == query_label:
            relevant_found += 1
        precision = relevant_found / rank
        recall = relevant_found / relevant_total
        stats.append({
            "n": rank,
            "candidate_id": candidate_id,
            "precision": precision,
            "recall": recall,
        })
    return stats

def build_confusion_matrix(stats_with_data: list[dict], top_k: int = 1, class_labels: list[str] = None):
    """
    Build a confusion matrix using sklearn.metrics.confusion_matrix.
    If top_k is provided, only the first top_k results per query contribute.
    """
    y_true = []
    y_pred = []
    labels = list(class_labels) if class_labels else []

    for entry in stats_with_data:
        query_class = entry.get("query_class")
        stats = entry.get("stats", [])
        if not query_class or not stats:
            continue

        selected_stats = stats if top_k is None else stats[:top_k]
        for stat in selected_stats:
            candidate_class = stat.get("candidate_class")
            if not candidate_class:
                continue
            y_true.append(query_class)
            y_pred.append(candidate_class)

    if not y_true:
        return np.zeros((0, 0), dtype=np.float32), labels

    if not labels:
        labels = _collect_class_labels(stats_with_data)
    matrix = sk_confusion_matrix(y_true, y_pred, labels=labels)
    return matrix.astype(np.float32), labels

def render_confusion_matrix_image(confusion_matrix, class_labels,
                                  cell_size=80, axis_padding=140):
    """Turn the numeric confusion matrix into a labeled heatmap image."""
    label_count = len(class_labels)
    if label_count == 0:
        # Return a placeholder image indicating missing data
        placeholder = np.full((200, 400, 3), (30, 30, 30), dtype=np.uint8)
        cv2.putText(
            placeholder,
            "No data for confusion matrix",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        return placeholder

    max_value = float(confusion_matrix.max()) if confusion_matrix.size else 1.0
    if max_value == 0:
        max_value = 1.0
    normalized = (confusion_matrix / max_value).astype(np.float32)
    heatmap_gray = (normalized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_VIRIDIS)
    heatmap_color = cv2.resize(
        heatmap_color,
        (label_count * cell_size, label_count * cell_size),
        interpolation=cv2.INTER_NEAREST
    )

    canvas_h = label_count * cell_size + axis_padding
    canvas_w = label_count * cell_size + axis_padding
    canvas = np.full((canvas_h, canvas_w, 3), (30, 30, 30), dtype=np.uint8)
    canvas[axis_padding:, axis_padding:] = heatmap_color

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    # Axis labels
    cv2.putText(
        canvas,
        "Predicted",
        (axis_padding + (label_count * cell_size) // 2 - 80, axis_padding - 60),
        font,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    cv2.putText(
        canvas,
        "True",
        (20, axis_padding + (label_count * cell_size) // 2),
        font,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Tick labels and counts
    for idx, label in enumerate(class_labels):
        text_size, _ = cv2.getTextSize(label, font, 0.6, 2)
        text_x = axis_padding + idx * cell_size + (cell_size - text_size[0]) // 2
        cv2.putText(
            canvas,
            label,
            (text_x, axis_padding - 20),
            font,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        text_y = axis_padding + idx * cell_size + (cell_size + text_size[1]) // 2
        cv2.putText(
            canvas,
            label,
            (20, text_y),
            font,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    for row in range(label_count):
        for col in range(label_count):
            value = int(confusion_matrix[row, col])
            cell_x = axis_padding + col * cell_size
            cell_y = axis_padding + row * cell_size
            norm_value = normalized[row, col]
            text_color = (0, 0, 0) if norm_value > 0.6 else (255, 255, 255)
            text = str(value)
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = cell_x + (cell_size - text_size[0]) // 2
            text_y = cell_y + (cell_size + text_size[1]) // 2
            cv2.putText(
                canvas,
                text,
                (text_x, text_y),
                font,
                font_scale,
                text_color,
                thickness,
                cv2.LINE_AA
            )

    return canvas

def show_confusion_matrix_image(confusion_matrix_image, title):
    """Display the rendered confusion matrix using matplotlib."""
    rgb_image = cv2.cvtColor(confusion_matrix_image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.imshow(rgb_image)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()