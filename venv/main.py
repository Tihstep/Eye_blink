from model import Eye_CNN
import torch
import Foo
from PIL import Image

class OpenEyesClassificator():
    def __init__(self):
        self.model = Eye_CNN
        model.load_state_dict(torch.load('weights.pt',map_location=torch.device('cpu')))
    def predict(self, inpIm):
        img = Image.open(inpIm)
        tens_img = transform(torch.tensor(image))
        with torch.no_grad():
            output = model(tens_img).unsqueeze(0)
            output.detach().numpy()
            confidence = points_to_conf(output)
        return confidence