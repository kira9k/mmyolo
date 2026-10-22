import torch.nn as nn
from torch.nn.modules.utils import _pair


def replace_maxpool5x5(model):
    """Decompose SPPF pooling independently of the installed surgery version."""
    for name, child in list(model.named_children()):
        if (isinstance(child, nn.MaxPool2d)
                and _pair(child.kernel_size) == (5, 5)
                and _pair(child.stride) == (1, 1)
                and _pair(child.padding) == (2, 2)
                and _pair(child.dilation) == (1, 1) and not child.ceil_mode
                and not child.return_indices):
            replacement = nn.Sequential(
                nn.MaxPool2d(3, stride=1, padding=1),
                nn.MaxPool2d(3, stride=1, padding=1))
            replacement.train(child.training)
            setattr(model, name, replacement)
        else:
            replace_maxpool5x5(child)
