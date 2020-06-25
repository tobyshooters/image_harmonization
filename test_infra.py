import os
import cv2
import numpy as np
from shutil import copyfile

D = "HAdobe5k"
if not os.path.exists(D + "/grids"):
    os.mkdir(D + "/grids")

i = 0
done = []

for l in os.listdir(D + "/results"):
    print(i)

    name = l.strip().split("_")[0]
    name_with_id = "_".join(l.strip().split("_")[:3])

    if name_with_id in done:
        continue

    ###################################################################
    # STEP 1: ISOLATE TEST
    c = cv2.imread(os.path.join(D, "composite_images", name_with_id + ".jpg"))
    m = cv2.imread(os.path.join(D, "masks", name + "_1.png"))
    o = cv2.imread(os.path.join(D, "results", name_with_id + "_model_output.jpg"))
    l = cv2.imread(os.path.join(D, "results", name_with_id + "_transfered_Lab.jpg"))
    h = cv2.imread(os.path.join(D, "results", name_with_id + "_transfered_hist.jpg"))
    r = cv2.imread(os.path.join(D, "real_images", name + ".jpg"))
    ###################################################################

    ###################################################################
    # STEP 2: GRID RESULTS
    grid = np.hstack([c, m, o, l, h, r])
    h, w, _ = c.shape

    cv2.putText(grid, "composite",    (0*w + 10, h-100), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 0, 255), 12)
    cv2.putText(grid, "mask",         (1*w + 10, h-100), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 0, 255), 12)
    cv2.putText(grid, "model output", (2*w + 10, h-100), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 0, 255), 12)
    cv2.putText(grid, "Lab transfer", (3*w + 10, h-100), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 0, 255), 12)
    cv2.putText(grid, "RGb transfer", (4*w + 10, h-100), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 0, 255), 12)
    cv2.putText(grid, "ground truth", (5*w + 10, h-100), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 0, 255), 12)

    cv2.imwrite(D + "/grids/" + name_with_id + ".jpg", grid)
    ###################################################################

    done.append(name_with_id)
    i += 1
