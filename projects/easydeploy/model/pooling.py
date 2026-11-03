import torch.nn as nn
from torch.nn.modules.utils import _pair


def get_surgery_kwargs(maxpool_kernel_size, version=2):
    """Disable implicit MaxPool decomposition without changing other surgery."""
    if maxpool_kernel_size is None:
        return {}
    if maxpool_kernel_size not in (3, 5):
        raise ValueError('maxpool_kernel_size must be 3 or 5')
    if version not in (1, 2):
        raise ValueError('Surgery version must be 1 or 2')
    from edgeai_torchmodelopt import xmodelopt
    surgery = getattr(xmodelopt.surgery, f'v{version}')
    if hasattr(surgery, 'get_replacement_flag_dict_default'):
        replacements = dict(surgery.get_replacement_flag_dict_default())
        for key in replacements:
            if 'maxpool' in str(key).lower():
                replacements[key] = False
    elif hasattr(surgery, 'get_replacement_dict_default'):
        replacements = {
            key: value
            for key, value in surgery.get_replacement_dict_default().items()
            if 'maxpool' not in str(key).lower()
        }
        if not replacements:
            raise RuntimeError(
                'Cannot disable MaxPool surgery with an empty replacement map')
    else:
        raise RuntimeError(
            'Cannot control MaxPool replacement in this surgery version')
    return dict(replacement_dict=replacements)


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
