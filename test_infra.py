import os
import cv2
import numpy as np
from shutil import copyfile

def gridImages(imgs, labels):
    h, w, _ = imgs[0].shape
    grid = np.hstack(imgs)
    for i, l in enumerate(labels):
        cv2.putText(grid, l, (i*w+10, h-50), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 255), 5)
    return grid

D = "HAdobe5k"
if not os.path.exists(D + "/grids"):
    os.mkdir(D + "/grids")

done = []
for i, l in enumerate(os.listdir(D + "/results")):
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
    grid = gridImages([c, m, o, l, h, r], ["composite", "mask", "model output", "Lab", "RGB", "ground truth"])
    cv2.imwrite(D + "/grids/" + name_with_id + ".jpg", grid)
    ###################################################################

    done.append(name_with_id)
