# Copyright (c) OpenMMLab. All rights reserved.
"""Physical channel pruning and portable topology metadata for MMYOLO."""
from contextlib import contextmanager
from copy import deepcopy
from types import MethodType

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmyolo.models.dense_heads import YOLOv8HeadModule
from mmyolo.models.layers import CSPLayerWithTwoConv


class RawYOLOv8(nn.Module):
    """Trace compute layers without decoding, NMS or changing the trained head."""

    def __init__(self, detector):
        super().__init__()
        self.detector = detector

    def forward(self, inputs):
        cls, bbox = self.detector.bbox_head.head_module(
            self.detector.extract_feat(inputs))
        return tuple(torch.cat((c, b), dim=1) for c, b in zip(cls, bbox))


def refresh_channel_attributes(model):
    for module in model.modules():
        if isinstance(module, ConvModule):
            module.in_channels = module.conv.in_channels
            module.out_channels = module.conv.out_channels
        elif isinstance(module, CSPLayerWithTwoConv):
            channels = module.main_conv.conv.out_channels
            if channels % 2:
                raise ValueError('C2f must retain two equal channel groups')
            module.mid_channels = channels // 2


def _separate_csp_forward(self, x):
    first = self.pruning_first(x)
    last = self.pruning_second(x)
    if self.fusion_mode == 'add_last':
        for block in self.blocks:
            last = block(last)
        return self.final_conv(first + last)
    if self.fusion_mode == 'add':
        merged = first + last
        for block in self.blocks:
            last = block(last)
            merged = merged + last
        return self.final_conv(merged)
    parts = [first, last]
    for block in self.blocks:
        last = block(last)
        parts.append(last)
    return self.final_conv(torch.cat(parts, dim=1))


@contextmanager
def traceable_csp_splits(model):
    import torch_pruning as tp

    originals = []
    replacements = {}
    for module in model.modules():
        if isinstance(module, CSPLayerWithTwoConv):
            main = module.main_conv
            originals.append((module, main, module.__dict__.get('forward')))
    # Two independent convolutions avoid reused Split outputs in DepGraph.
    # They are merged back into the original Conv+Split after pruning.
    for module, main, _ in originals:
        first, second = deepcopy(main), deepcopy(main)
        count = main.conv.out_channels // 2
        for branch, indices in ((first, list(range(count, 2 * count))),
                                (second, list(range(count)))):
            tp.prune_conv_out_channels(branch.conv, indices)
            if branch.with_norm:
                tp.prune_batchnorm_out_channels(branch.norm, indices)
        module.main_conv = nn.Identity()
        module.pruning_first = first
        module.pruning_second = second
        module.forward = MethodType(_separate_csp_forward, module)
        replacements[main.conv] = [first.conv, second.conv]
    try:
        yield replacements
        for module, main, _ in originals:
            first, second = module.pruning_first, module.pruning_second
            if (first.conv.in_channels != second.conv.in_channels
                    or first.conv.out_channels != second.conv.out_channels):
                raise RuntimeError('C2f branch widths diverged during pruning')
            main.conv.in_channels = first.conv.in_channels
            main.conv.out_channels = 2 * first.conv.out_channels
            main.conv.weight = nn.Parameter(
                torch.cat(
                    (first.conv.weight.detach(), second.conv.weight.detach()),
                    0),
                requires_grad=main.conv.weight.requires_grad)
            if main.conv.bias is not None:
                main.conv.bias = nn.Parameter(
                    torch.cat(
                        (first.conv.bias.detach(), second.conv.bias.detach()),
                        0),
                    requires_grad=main.conv.bias.requires_grad)
            if main.with_norm:
                main.norm.num_features = main.conv.out_channels
                for attr in ('weight', 'bias', 'running_mean', 'running_var'):
                    values = [
                        getattr(branch.norm, attr)
                        for branch in (first, second)
                    ]
                    if values[0] is not None:
                        value = torch.cat([v.detach() for v in values], 0)
                        old = getattr(main.norm, attr)
                        if isinstance(old, nn.Parameter):
                            value = nn.Parameter(
                                value, requires_grad=old.requires_grad)
                        setattr(main.norm, attr, value)
    finally:
        for module, main, forward in originals:
            module.main_conv = main
            del module.pruning_first
            del module.pruning_second
            if forward is None:
                del module.forward
            else:
                module.forward = forward


def prune_yolov8(model,
                 example,
                 ratio=0.2,
                 round_to=16,
                 scope='backbone-neck'):
    import torch_pruning as tp

    if not 0 < ratio < 1 or round_to < 1:
        raise ValueError('Expected 0 < ratio < 1 and round_to >= 1')
    if scope not in ('backbone', 'backbone-neck', 'all'):
        raise ValueError('Unknown pruning scope')
    head = model.bbox_head.head_module
    if not isinstance(head, YOLOv8HeadModule):
        raise ValueError('Structural pruning currently supports YOLOv8 only')
    model.eval()
    wrapper = RawYOLOv8(model).eval()
    with torch.no_grad():
        shapes_before = [list(o.shape) for o in wrapper(example)]
        macs_before, _ = tp.utils.count_ops_and_params(wrapper, example)
    params_before = sum(p.numel() for p in model.parameters())
    channels_before = {
        name: [m.in_channels, m.out_channels]
        for name, m in model.named_modules() if isinstance(m, nn.Conv2d)
    }
    ignored = [
        next(m for m in model.backbone.modules() if isinstance(m, nn.Conv2d))
    ]
    if scope == 'backbone':
        ignored.append(model.neck)
    if scope != 'all':
        ignored.append(head)
    else:
        ignored.extend(branch[-1] for branch in head.cls_preds)
        ignored.extend(branch[-1] for branch in head.reg_preds)
    # Autograd is required by DepGraph even though no optimizer is used here.
    with torch.enable_grad(), traceable_csp_splits(model) as replacements:
        ignored = [
            replacement for layer in ignored
            for replacement in replacements.get(layer, [layer])
        ]
        pruner = tp.pruner.MagnitudePruner(
            wrapper,
            example.detach().requires_grad_(True),
            importance=tp.importance.GroupMagnitudeImportance(p=2),
            pruning_ratio=ratio,
            round_to=round_to,
            global_pruning=False,
            ignored_layers=ignored)
        pruner.step()
    refresh_channel_attributes(model)
    with torch.no_grad():
        outputs = wrapper(example)
        shapes_after = [list(o.shape) for o in outputs]
        if not all(torch.isfinite(output).all() for output in outputs):
            raise RuntimeError('Pruning produced non-finite detection outputs')
        macs_after, _ = tp.utils.count_ops_and_params(wrapper, example)
    if shapes_before != shapes_after:
        raise RuntimeError('Pruning changed the detection output contract')
    params_after = sum(p.numel() for p in model.parameters())
    if params_after >= params_before:
        raise RuntimeError(
            'No channels removed; lower round_to or raise ratio')
    changes = {
        name:
        dict(
            before=channels_before[name],
            after=[m.in_channels, m.out_channels])
        for name, m in model.named_modules() if isinstance(m, nn.Conv2d)
        and channels_before[name] != [m.in_channels, m.out_channels]
    }
    return dict(
        method='torch-pruning/group-l2',
        requested_ratio=ratio,
        round_to=round_to,
        scope=scope,
        input_shape=list(example.shape),
        output_shapes=shapes_after,
        parameters_before=params_before,
        parameters_after=params_after,
        macs_before=int(macs_before),
        macs_after=int(macs_after),
        parameter_reduction=1 - params_after / params_before,
        changed_convolutions=changes)


def describe_structure(model):
    structure = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            structure[name] = dict(
                type='Conv2d',
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                groups=module.groups)
        elif isinstance(module, nn.modules.batchnorm._BatchNorm):
            structure[name] = dict(
                type='BatchNorm', num_features=module.num_features)
    return structure


def restore_structure(model, structure):
    """Resize parameters before loading a structurally pruned state_dict."""
    modules = dict(model.named_modules())
    for name, spec in structure.items():
        module = modules[name]
        if spec['type'] == 'Conv2d' and isinstance(module, nn.Conv2d):
            in_channels, out_channels, groups = (spec['in_channels'],
                                                 spec['out_channels'],
                                                 spec['groups'])
            if min(in_channels, out_channels,
                   groups) < 1 or (in_channels % groups
                                   or out_channels % groups):
                raise ValueError(f'Invalid convolution structure: {name}')
            module.in_channels = in_channels
            module.out_channels = out_channels
            module.groups = groups
            module.weight = nn.Parameter(
                module.weight.new_empty(out_channels, in_channels // groups,
                                        *module.kernel_size),
                requires_grad=module.weight.requires_grad)
            if module.bias is not None:
                module.bias = nn.Parameter(
                    module.bias.new_empty(out_channels),
                    requires_grad=module.bias.requires_grad)
        elif spec['type'] == 'BatchNorm' and isinstance(
                module, nn.modules.batchnorm._BatchNorm):
            size = spec['num_features']
            if size < 1:
                raise ValueError(f'Invalid BatchNorm structure: {name}')
            module.num_features = size
            if module.affine:
                for attr in ('weight', 'bias'):
                    old = getattr(module, attr)
                    setattr(
                        module, attr,
                        nn.Parameter(
                            old.new_empty(size),
                            requires_grad=old.requires_grad))
            if module.track_running_stats:
                module.running_mean = module.running_mean.new_empty(size)
                module.running_var = module.running_var.new_empty(size)
        else:
            raise ValueError(f'Unsupported structure entry: {name}')
    refresh_channel_attributes(model)
