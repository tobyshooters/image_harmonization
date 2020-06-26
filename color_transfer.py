import cv2
import numpy as np

# Faster than numpy operations with axis
def mean(img):
    return np.array([img[:, i].mean() for i in range(3)])

def std(img):
    return np.array([img[:, i].std() for i in range(3)])


# This is an implementation of Algorithms for Rendering in Artistic Styles (2001)
# https://cs.nyu.edu/media/publications/hertzmann_aaron.pdf

def transfer_color_histogram(img, style, mask):
    w, h, _ = style.shape
    sflat = img.reshape(-1, 3).T
    cflat = style.reshape(-1, 3).T

    alpha = (mask[:, :, 0] > 127).astype(np.float64)
    cflat_roi = style[alpha == 1].T
    sflat_roi = img[alpha == 1].T

    # Square root of covariance via diagonalization
    mean_c = np.mean(cflat_roi, axis=1, keepdims=True)
    cov_c = np.cov(cflat_roi)
    r, v = np.linalg.eig(cov_c)
    cov_sqrt_c = v.dot(np.diag(r**.5)).dot(v.T)

    # Square root of covariance via diagonalization
    mean_s = np.mean(sflat_roi, axis=1, keepdims=True)
    cov_s = np.cov(sflat_roi)
    r, v = np.linalg.eig(cov_s)
    cov_sqrt_s = v.dot(np.diag(r**-.5)).dot(v.T)

    # Shift sflat according to distribution
    A = cov_sqrt_c.dot(cov_sqrt_s)
    canvas = A.dot(sflat - mean_s) + mean_c

    # Paste masked region and re-format
    canvas = np.clip(canvas, 0, 255)
    canvas = canvas.reshape((3, w, h)).transpose((1,2,0))

    output = img.copy()
    output[alpha == 1] = canvas[alpha == 1]

    return output.astype('uint8')


# Reinhart transfer method, done in Lab space
# https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.pdf

def transfer_Lab_statistics(img, style, mask, soften=False):
    # Preprocess inputs, images to Lab space
    h, w, _ = img.shape
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)
    style_lab = cv2.cvtColor(style, cv2.COLOR_BGR2LAB).astype(np.float64)

    # Relevant region
    alpha = (mask[:, :, 0] > 127).astype(np.float64)
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
    output = img_lab.copy()
    output[alpha == 1] = canvas[alpha == 1]

    # Blur paste to avoid rough edges
    if soften:
        m = 2 * int(min(h, w) / 200) + 1
        n = 2 * int(min(h, w) / 100) + 1
        blurred_output = cv2.GaussianBlur(output, (n, n), 0)
        gradient = cv2.morphologyEx(alpha, cv2.MORPH_GRADIENT, np.ones((m, m), np.uint8))
        blurred_gradient = cv2.GaussianBlur(gradient, (n, n), 0)[:, :, None]
        output = (1 - blurred_gradient) * output + blurred_gradient * blurred_output

    # Convert back to image
    output = np.clip(output, 0, 255)
    output = cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return output
