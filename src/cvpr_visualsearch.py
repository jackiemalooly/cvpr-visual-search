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

DEFAULT_BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_PATH = os.getenv("BASE_PATH", DEFAULT_BASE_PATH)
print(f"BASE_PATH: {BASE_PATH}")

DESCRIPTOR_FOLDER = os.path.join(BASE_PATH, 'descriptors')
DESCRIPTOR_SUBFOLDER = 'globalRGBhisto'
IMAGE_FOLDER = os.path.join(BASE_PATH, 'MSRC_ObjCategImageDatabase_v2', 'Images')

# Load all descriptors
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


# Sort the distances
dst.sort(key=lambda x: x[0])

SHOW = 15

# Prepare list of result paths, skipping the query itself
result_img_paths = []
for distance, idx in dst:
    if idx == queryimg:
        continue
    result_img_paths.append(descriptor_to_image_path(ALLFILES[idx]))
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

