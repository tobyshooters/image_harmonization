import argparse
import os
import os.path as osp
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, '.')
from iharm.inference.predictor import Predictor
from iharm.inference.utils import load_model, find_checkpoint
from iharm.mconfigs import ALL_MCONFIGS
from iharm.utils.log import logger
from iharm.utils.exp import load_config_file

from color_transfer import transfer_Lab_statistics


def main():
    args, cfg = parse_args()

    if args.gpu == "-1":
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.gpu}')
    checkpoint_path = find_checkpoint(cfg.MODELS_PATH, args.checkpoint)
    net = load_model(args.model_type, checkpoint_path, verbose=True)
    predictor = Predictor(net, device)

    image_names = os.listdir(args.images)

    def _save_image(image_name, bgr_image, flag=""):
        file_name = image_name.split(".")[0]
        path = os.path.join(cfg.RESULTS_PATH, file_name + flag + ".jpg")
        cv2.imwrite(path, bgr_image, [cv2.IMWRITE_JPEG_QUALITY, 85])

    logger.info(f'Save images to {cfg.RESULTS_PATH}')

    already_processed = os.listdir(cfg.RESULTS_PATH)

    with open(args.test_path, "r") as f:
        test_files = [l.strip() for l in f.readlines()]

    num_processed = 0

    for image_name in tqdm(image_names):

        if args.num_inputs > 0 and num_processed == args.num_inputs:
            break

        if image_name.split(".")[-1] != "jpg":
            continue

        if image_name not in test_files:
            continue

        file_name = image_name.split(".")[0] + "_model_output.jpg"
        if file_name in already_processed:
            print(image_name + " already done")
            continue

        print(image_name + " running")

        image_path = osp.join(args.images, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        og_image = image.copy()
        image_size = image.shape

        divisor = 128
        if args.resize > 0:

            def f(h, w, r, d=128):
                if h > w:
                    f = round(h / r)
                    new_h = d * (r // d)
                    new_w = d * ((w // f) // d)
                else:
                    f = round(w / r)
                    new_h = d * ((h // f) // d)
                    new_w = d * (r // d)
                return max(new_h, 256), max(new_w, 256)

            h, w = f(image_size[0], image_size[1], args.resize)
            print("Image shape:", image_size)
            print("New shape:", (h, w))
            resize_shape = (w, h)

        if args.resize > 0:
            image = cv2.resize(image, resize_shape, cv2.INTER_LINEAR)
        else:
            h, w, _ = image.shape
            shape = (divisor * (w // divisor), divisor * (h // divisor))
            image = cv2.resize(image, shape, cv2.INTER_LINEAR)

        mask_path = osp.join(args.masks, '_'.join(image_name.split('_')[:-1]) + '.png')
        mask_image = cv2.imread(mask_path)
        og_mask = mask_image.copy()

        if args.resize > 0:
            mask_image = cv2.resize(mask_image, resize_shape, cv2.INTER_LINEAR)
        else:
            h, w, _ = mask_image.shape
            shape = (divisor * (w // divisor), divisor * (h // divisor))
            mask_image = cv2.resize(mask_image, shape, cv2.INTER_LINEAR)

        mask = mask_image[:, :, 0]
        mask[mask <= 100] = 0
        mask[mask > 100] = 1
        mask = mask.astype(np.float32)

        # Expects RGB!
        pred = predictor.predict(image, mask)

        if args.original_size:
            pred = cv2.resize(pred, image_size[:-1][::-1])

        bgr_pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR).astype(np.uint8)
        _save_image(image_name, bgr_pred, flag="_model_output")

        if args.color_transfer:
            assert(args.original_size)
            # Expects RGB!
            output = transfer_Lab_statistics(og_image, pred, og_mask)
            output = cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_RGB2BGR)
            _save_image(image_name, output, flag="_transfered_Lab")

        num_processed += 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_type', choices=ALL_MCONFIGS.keys())
    parser.add_argument('checkpoint', type=str,
                        help='The path to the checkpoint. '
                             'This can be a relative path (relative to cfg.MODELS_PATH) '
                             'or an absolute path. The file extension can be omitted.')
    parser.add_argument(
        '--images', type=str,
        help='Path to directory with .jpg images to get predictions for.'
    )
    parser.add_argument(
        '--masks', type=str,
        help='Path to directory with .png binary masks for images, named exactly like images without last _postfix.'
    )
    parser.add_argument(
        '--resize', type=int, default=256,
        help='Resize image to a given size before feeding it into the network. If -1 the network input is not resized.'
    )
    parser.add_argument(
        '--original-size', action='store_true', default=False,
        help='Resize predicted image back to the original size.'
    )
    parser.add_argument('--gpu', type=str, default=0, help='ID of used GPU.')
    parser.add_argument('--config-path', type=str, default='./config.yml', help='The path to the config file.')
    parser.add_argument(
        '--results-path', type=str, default='',
        help='The path to the harmonized images. Default path: cfg.EXPS_PATH/predictions.'
    )

    parser.add_argument(
        '--test-path', type=str,
        help='Path to text file identifying tests '
    )

    parser.add_argument(
        '--color-transfer', action='store_true', default=False,
        help='Use color transfer to get a full-resolution image'
    )

    parser.add_argument(
        '--num-inputs', type=int, default=-1,
        help='Number of inputs to run'
    )

    args = parser.parse_args()
    cfg = load_config_file(args.config_path, return_edict=True)
    cfg.EXPS_PATH = Path(cfg.EXPS_PATH)
    cfg.RESULTS_PATH = Path(args.results_path) if len(args.results_path) else cfg.EXPS_PATH / 'predictions'
    cfg.RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    logger.info(cfg)
    return args, cfg


if __name__ == '__main__':
    main()
