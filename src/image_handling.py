import os
import cv2
import sys
import math
import scipy.io as sio
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

DEFAULT_BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_PATH = os.getenv("BASE_PATH", DEFAULT_BASE_PATH)
print(f"BASE_PATH: {BASE_PATH}")
IMAGE_FOLDER = os.path.join(BASE_PATH, 'MSRC_ObjCategImageDatabase_v2', 'Images')

def descriptor_to_image_path(mat_path: str) -> str:
    """Convert a descriptor path to its corresponding image path."""
    img_name = os.path.splitext(os.path.basename(mat_path))[0] + '.bmp'
    return os.path.join(IMAGE_FOLDER, img_name)


def load_and_fit_image(img_path: str, cell_size=(200, 150), bg_color=(30, 30, 30)):
    """Load an image, keep aspect ratio within cell, and return cell with label area."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    target_w, target_h = cell_size
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    resized = cv2.resize(img, new_size)
    cell = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)
    y_offset = (target_h - resized.shape[0]) // 2
    x_offset = (target_w - resized.shape[1]) // 2
    cell[y_offset:y_offset + resized.shape[0], x_offset:x_offset + resized.shape[1]] = resized
    return cell


def add_label(image: np.ndarray, text: str):
    """Overlay a label at the bottom-left corner of an image cell."""
    return cv2.putText(
        image.copy(),
        text,
        (5, image.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

def visualize_query_results(query_idx: int,
                             matches,
                             all_files,
                             cols: int,
                             padding: int):
    """Render a grid with the sample query and its nearest neighbors."""
    if not matches:
        print("Warning: Unable to visualize sample query; no matches provided.")
        return

    result_img_paths = [
        descriptor_to_image_path(all_files[idx]) for _, idx in matches
    ]

    cells = []
    for rank, img_path in enumerate(result_img_paths, start=1):
        cell = load_and_fit_image(img_path)
        if cell is None:
            print(f"Warning: Could not load result image at {img_path}")
            continue
        cells.append(add_label(cell, f"Rank {rank}"))

    query_cell = load_and_fit_image(descriptor_to_image_path(all_files[query_idx]))
    if query_cell is not None:
        cells.insert(0, add_label(query_cell, "Query"))

    if not cells:
        print("Warning: Unable to visualize sample query; no images loaded.")
        return

    cols = max(1, min(cols, len(cells)))
    rows = math.ceil(len(cells) / cols)
    cell_h, cell_w = cells[0].shape[0], cells[0].shape[1]
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

def mat_path_to_image_id(mat_path: str) -> str:
    """Return the dataset image identifier from a descriptor path."""
    return os.path.splitext(os.path.basename(mat_path))[0]

def load_descriptor_bank(descriptor_folder: str, descriptor_subfolder: str):
    """Load all descriptors in the requested subfolder."""
    descriptor_dir = os.path.join(descriptor_folder, descriptor_subfolder)

    if not os.path.isdir(descriptor_dir):
        raise RuntimeError(f"Descriptor directory not found: {descriptor_dir}")

    all_feat = []
    all_files = []
    for filename in sorted(os.listdir(descriptor_dir)):
        if not filename.endswith('.mat'):
            continue
        img_path = os.path.join(descriptor_dir, filename)
        img_data = sio.loadmat(img_path)
        all_files.append(img_path)
        all_feat.append(img_data['F'][0])  # Assuming F is a 1D array

    if not all_files:
        raise RuntimeError(f"No descriptor files found in {descriptor_dir}")

    return np.array(all_feat), all_files