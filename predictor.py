import numpy as np
import torch

import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms


def process_image(image):
    img = Image.open(image)
    convert_to_tensor = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    tensor_img = convert_to_tensor(img).float()
    np_image = np.array(tensor_img)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean) / std
    return np.transpose(np_image, (2, 0, 1))


def predict(image_path, model, top_k=5, gpu=True):
    if torch.cuda.is_available() and gpu:
        model.to('cuda:0')

    image = Variable(torch.FloatTensor(process_image(image_path)), requires_grad=True)
    image = image.unsqueeze(0)

    torch_image = image
    if torch.cuda.is_available() and gpu:
        torch_image = image.cuda()

    with torch.no_grad():
        output = model.forward(torch_image)

    predictions = F.softmax(output.data, dim=1)
    probs, classes = predictions.topk(top_k)

    if torch.cuda.is_available() and gpu:
        return probs.cpu().numpy()[0], classes.cpu().numpy()[0]
    else:
        return probs.numpy()[0], classes.numpy()[0]
