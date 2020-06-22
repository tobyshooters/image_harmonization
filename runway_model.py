
# python3 scripts/predict_for_dir.py   \
#     hrnet18_idih256                  \
#     checkpoints/hrnet18_idih256.pth  \
#     --images test_data/images/       \
#     --masks test_data/masks/         \
#     --results-path test_data/results \
#     --gpu -1                         \
#     --resize 256                     \
#     --original-size                  \

import os
import torch
import cv2
import runway
from iharm.inference.predictor import Predictor

MODEL = "hrnet18_idih256"
INPUT_SIZE = 256
PORT = 4231

def _load_model(model_type, checkpoint):
    net = ALL_MCONFIGS[model_type]['model'](**ALL_MCONFIGS[model_type]['params'])
    state = net.state_dict()
    state.update(checkpoint)
    net.load_state_dict(state)
    return net

def _fmt_input(img, size=INPUT_SIZE):
    return cv2.resize(np.array(img), (size, size), cv2.INTER_LINEAR)


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
    img = _fmt_input(inputs["composite_image"])
    mask = _fmt_input(inputs["foreground_mask"])
    _, mask = cv2.threshold(mask[:, :, 0], 127, 255, cv2.THRESH_BINARY)
    return model.predict(image, mask)


if __name__ == '__main__':
    runway.run(port=PORT)
