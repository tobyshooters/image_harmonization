import cv2
import numpy as np

# Faster than numpy operations with axis
def mean(img):
    return np.array([img[:, i].mean() for i in range(3)])

def std(img):
    return np.array([img[:, i].std() for i in range(3)])

# Reinhart transfer method, done in Lab space
# https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.pdf

def transfer_Lab_statistics(img, style, mask):
    # Preprocess inputs, images to Lab space
    h, w, _ = img.shape
    img_lab = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float64)
    style_lab = cv2.cvtColor(style.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float64)

    # Relevant region
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    alpha = (mask > 127).astype(np.float64)
    img_roi = img_lab[alpha == 1]
    style_roi = style_lab[alpha == 1]

    # Accumulate statistics
    img_avg = mean(img_roi)
    img_std = std(img_roi)
    style_avg = mean(style_roi)
    style_std = std(style_roi)

    # Transfer statistics
    canvas = img_lab.copy()
    canvas -= img_avg
    canvas *= style_std / img_std
    canvas += style_avg

    # Paste masked region
    img_lab[alpha == 1] = canvas[alpha == 1]

    # Convert back to image
    img_lab = np.clip(img_lab, 0, 255)
    img_lab = cv2.cvtColor(img_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return img_lab
