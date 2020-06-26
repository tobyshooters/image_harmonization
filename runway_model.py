import os
import torch
import cv2
import runway
import numpy as np

from iharm.inference.predictor import Predictor
from iharm.mconfigs import ALL_MCONFIGS
from color_transfer import transfer_Lab_statistics

MODEL = "hrnet18_idih256"
INPUT_SIZE = 256

def _load_model(model_type, checkpoint):
    if type(checkpoint) is str:
        checkpoint = torch.load(checkpoint, map_location='cpu')

    model = ALL_MCONFIGS[model_type]['model'](**ALL_MCONFIGS[model_type]['params'])
    state = model.state_dict()
    state.update(checkpoint)
    model.load_state_dict(state)
    return model


@runway.setup(options={'checkpoint': runway.file(extension='.pth')})
def setup(opts):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = _load_model(MODEL, opts['checkpoint'])
    return Predictor(net, device)


inputs = {
    'composite_image': runway.image,
    'foreground_mask': runway.image,
    'transfer_color': runway.boolean(default=True, description="Transfer colors back to source image for high-resolution output"),
}

outputs = {
    'harmonized_image': runway.image
}

@runway.command('harmonize', inputs=inputs, outputs=outputs)
def harmonize(model, inputs):
    og_image = np.array(inputs["composite_image"])
    image_size = og_image.shape[:2]
    image = cv2.resize(og_image, (INPUT_SIZE, INPUT_SIZE), cv2.INTER_LINEAR)

    og_mask = np.array(inputs["foreground_mask"])
    mask = cv2.resize(og_mask, (INPUT_SIZE, INPUT_SIZE), cv2.INTER_LINEAR)
    mask = mask[:, :, 0]
    mask[mask <= 100] = 0
    mask[mask > 100] = 1

    output = model.predict(image, mask.astype(np.float32))
    output = cv2.resize(output, image_size[::-1], cv2.INTER_LINEAR)

    if inputs["transfer_color"]:
        output = transfer_Lab_statistics(og_image, output, og_mask)

    return output


if __name__ == '__main__':
    runway.run(debug=True, model_options={'checkpoint': "models/checkpoints/hrnet18_idih256.pth"})
