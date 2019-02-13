import torch
from torch import nn
import torchvision.models as models

ARCH_INPUTS = {
    'alexnet': 9216,
    'densenet121': 1024,
    'vgg16': 25088,
}


def setup(arch='vgg16', hidden_units=120, gpu=True):
    if arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        print("Im sorry but {} is not a valid model.Did you mean vgg16, densenet121,or alexnet?".format(structure))

    # freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(ARCH_INPUTS[arch], hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier

    if torch.cuda.is_available() and gpu:
        model.cuda()

    return model
