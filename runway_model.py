# --------------------------------------
# Run locally:
# --------------------------------------
# python3 scripts/predict_for_dir.py   \
#     hrnet18_idih256                  \
#     checkpoints/hrnet18_idih256.pth  \
#     --images test_data/images/       \
#     --masks test_data/masks/         \
#     --results-path test_data/results \
#     --gpu -1                         \
#     --resize 256                     \
#     --original-size                  \
# --------------------------------------

import os
import torch
import cv2
import numpy as np
import runway

from iharm.inference.predictor import Predictor
from iharm.mconfigs import ALL_MCONFIGS

MODEL = "hrnet18_idih256"
INPUT_SIZE = 256
MASK_TYPE = None

def _load_model(model_type, checkpoint):
    if type(checkpoint) is str:
        checkpoint = torch.load(checkpoint, map_location='cpu')

    model = ALL_MCONFIGS[model_type]['model'](**ALL_MCONFIGS[model_type]['params'])
    state = model.state_dict()
    state.update(checkpoint)
    model.load_state_dict(state)
    return model

options = {
    'checkpoint': runway.file(extension='.pth'),
    'mask': runway.category(choices=["binary", "alpha"], default="alpha", description="Type of mask"),
}

@runway.setup(options=options)
def setup(opts):
    global MASK_TYPE
    MASK_TYPE = opts["mask"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = _load_model(MODEL, opts['checkpoint'])
    return Predictor(net, device)


inputs = {
    'composite_image': runway.image,
    'foreground_mask': runway.image(channels=4) 
}

outputs = {
    'harmonized_image': runway.image
}

@runway.command('harmonize', inputs=inputs, outputs=outputs)
def harmonize(model, inputs):
    image = np.array(inputs["composite_image"])
    image_size = image.shape[:2]
    image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE), cv2.INTER_LINEAR)

    mask = np.array(inputs["foreground_mask"])
    mask = cv2.resize(mask, (INPUT_SIZE, INPUT_SIZE), cv2.INTER_LINEAR)
    if MASK_TYPE == "binary":
        mask = mask[:, :, 0]
        mask[mask <= 100] = 0
        mask[mask > 100] = 1
    else:
        mask = mask[:, :, -1]
        mask[mask > 0] = 1
    mask = mask.astype(np.float32)

    output = model.predict(image, mask)
    return cv2.resize(output, image_size[::-1], cv2.INTER_LINEAR)


if __name__ == '__main__':
    runway.run(model_options={'checkpoint': "checkpoints/hrnet18_idih256.pth"})
