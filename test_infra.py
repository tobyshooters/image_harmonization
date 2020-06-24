import os
import cv2
import numpy as np
from shutil import copyfile

D = "HFlickr"
for directory in ["comps", "masks", "truth", "grids"]:
    if not os.path.exists("test_data/" + directory):
        os.mkdir("test_data/" + directory)

with open(D + "/HFlickr_test.txt", "r") as f:
    for i, l in enumerate(f.readlines()):
        name = l.strip().split(".")[0]
        simple_name = "_".join(name.split("_")[:-1])
        simplest_name = "_".join(simple_name.split("_")[:-1])

        ###################################################################
        # STEP 1: ISOLATE TEST
        # comp_path = os.path.join(D, "composite_images", name + ".jpg")
        # mask_path = os.path.join(D, "masks", simple_name + ".png")
        # real_path = os.path.join(D, "real_images", simplest_name + ".jpg")

        # copyfile(comp_path, "test_data/comps/" + name + ".jpg")
        # copyfile(mask_path, "test_data/masks/" + simple_name + ".png")
        # copyfile(real_path, "test_data/truth/" + simplest_name + ".jpg")
        ###################################################################

        ###################################################################
        # STEP 2: GRID RESULTS
        c = cv2.imread("test_data/comps/" + name + ".jpg")
        m = cv2.imread("test_data/masks/" + simple_name + ".png")
        r = cv2.imread("test_data/results/" + name + ".jpg")
        t = cv2.imread("test_data/truth/" + simplest_name + ".jpg")

        grid = np.hstack([c, m, r, t])
        h, w, _ = grid.shape
        cv2.putText(grid, "composite",    (0      + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(grid, "mask",         (w//4   + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(grid, "result",       (2*w//4 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(grid, "ground truth", (3*w//4 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite("test_data/grids/" + name + ".jpg", grid)
        ###################################################################

        if i == 199: break
