import os
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

# Sort the distances
dst.sort(key=lambda x: x[0])

# Show the top 15 results
SHOW = 15
for i in range(SHOW):
    mat_path = ALLFILES[dst[i][1]]
    img_name = os.path.splitext(os.path.basename(mat_path))[0] + '.bmp'
    img_path = os.path.join(IMAGE_FOLDER, img_name)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))  # Make image quarter size
    cv2.imshow(f"Result {i+1}", img)
    cv2.waitKey(0)
cv2.destroyAllWindows()

