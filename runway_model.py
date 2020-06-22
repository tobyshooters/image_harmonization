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
import runway
import numpy as np
from iharm.inference.predictor import Predictor
from iharm.mconfigs import ALL_MCONFIGS

MODEL = "hrnet18_idih256"
INPUT_SIZE = 256


def _load_model(model_type, checkpoint):
    net = ALL_MCONFIGS[model_type]['model'](**ALL_MCONFIGS[model_type]['params'])
    if checkpoint is None:
        checkpoint = torch.load("checkpoints/hrnet18_idih256.pth", map_location='cpu')
    state = net.state_dict()
    state.update(checkpoint)
    net.load_state_dict(state)
    return net

@runway.setup(options={'checkpoint': runway.file(extension='.pth')})
def setup(opts):
    net = _load_model(MODEL, opts['checkpoint'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return Predictor(net, device)


inputs = {
    'composite_image': runway.image,
    'foreground_mask': runway.image 
}

outputs = {
    'harmonized_image': runway.image
}

@runway.command('harmonize', inputs=inputs, outputs=outputs)
def harmonize(model, inputs):
    image = np.array(inputs["composite_image"])
    og_size = image.shape[:2]
    image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE), cv2.INTER_LINEAR)

    mask = np.array(inputs["foreground_mask"])
    mask = cv2.resize(mask, (INPUT_SIZE, INPUT_SIZE), cv2.INTER_LINEAR)
    mask =  mask[:, :, 0]
    mask[mask <= 100] = 0
    mask[mask > 100] = 1
    mask = mask.astype(np.float32)

    output = model.predict(image, mask)
    return cv2.resize(output, og_size[::-1], cv2.INTER_LINEAR)


if __name__ == '__main__':
    runway.run()
