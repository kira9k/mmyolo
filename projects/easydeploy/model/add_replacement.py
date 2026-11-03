"""Inference-only replacement of feature sums by Concat + fixed dense Conv."""
import copy
import math

import torch
from torch import nn


def sum_conv_weight(channels, alpha=1, dtype=torch.float32, device='cpu'):
    identity = torch.eye(channels, dtype=dtype, device=device)
    return torch.cat((identity, alpha * identity), dim=1)[:, :, None, None]


def replace_exported_adds(model):
    """Replace equal-shape floating NCHW Add tensors in an exported FX graph.

    Broadcasting, scalar/shape additions and non-4D tensors are left intact.
    Alpha is preserved. In-place additions are not changed.
    """
    graph = model.graph
    replacements = []
    for node in list(graph.nodes):
        if node.op != 'call_function' or str(node.target) != 'aten.add.Tensor':
            continue
        a, b = node.args[:2]
        if not all(isinstance(value, torch.fx.Node) for value in (a, b)):
            continue
        av, bv = a.meta.get('val'), b.meta.get('val')
        if not (isinstance(av, torch.Tensor) and isinstance(bv, torch.Tensor)
                and av.ndim == bv.ndim == 4 and av.shape == bv.shape
                and av.is_floating_point() and av.dtype == bv.dtype):
            continue
        alpha = node.kwargs.get('alpha', 1)
        if not isinstance(alpha, (int, float)) or not math.isfinite(alpha):
            raise ValueError('Feature Add alpha must be a finite scalar')
        channels = int(av.shape[1])
        name = f'add_concat_conv_{node.name}'
        while hasattr(model, name):
            name += '_'
        conv = nn.Conv2d(
            2 * channels,
            channels,
            1,
            bias=False,
            device=av.device,
            dtype=av.dtype)
        conv.weight = nn.Parameter(
            sum_conv_weight(channels, alpha, av.dtype, av.device),
            requires_grad=False)
        model.add_module(name, conv.eval())
        with graph.inserting_before(node):
            weight = graph.get_attr(name + '.weight')
            cat = graph.call_function(torch.ops.aten.cat.default, ([a, b], 1))
            result = graph.call_function(torch.ops.aten.conv2d.default,
                                         (cat, weight))
        cat.meta['val'] = torch.cat((av, bv), dim=1)
        result.meta = copy.copy(node.meta)
        result.meta['nn_module_stack'] = dict(
            node.meta.get('nn_module_stack', {}))
        result.meta['nn_module_stack'][name] = (name, nn.Conv2d)
        result.meta['mmyolo_module_name'] = name
        node.replace_all_uses_with(result)
        graph.erase_node(node)
        replacements.append(
            dict(node=node.name, module=name, channels=channels, alpha=alpha))
    graph.lint()
    model.recompile()
    return replacements


def replace_onnx_adds(model):
    """Rewrite only equal-shape floating NCHW feature Add, not bbox decoding."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.shape_inference.infer_shapes(model)
    values = {
        value.name: value.type.tensor_type
        for value in (*model.graph.input, *model.graph.value_info,
                      *model.graph.output)
    }
    used = {
        name
        for node in model.graph.node
        for name in (*node.input, *node.output)
    }
    used.update(value.name for value in model.graph.initializer)
    nodes, replacements = [], []
    for index, node in enumerate(model.graph.node):
        if node.op_type != 'Add' or node.domain or len(node.input) != 2:
            nodes.append(node)
            continue
        types = [values.get(name) for name in node.input]
        shapes = [
            tuple(dim.dim_value
                  for dim in value.shape.dim) if value is not None else ()
            for value in types
        ]
        if (len(shapes[0]) != 4 or shapes[0] != shapes[1] or not all(shapes[0])
                or types[0].elem_type != types[1].elem_type
                or types[0].elem_type
                not in (TensorProto.FLOAT, TensorProto.FLOAT16,
                        TensorProto.DOUBLE)):
            nodes.append(node)
            continue
        channels = shapes[0][1]
        name = f'add_concat_conv_{index}'
        while any(name + suffix in used for suffix in ('', '_cat', '_weight')):
            name += '_'
        used.update((name, name + '_cat', name + '_weight'))
        dtype = helper.tensor_dtype_to_np_dtype(types[0].elem_type)
        weight = sum_conv_weight(channels).numpy().astype(dtype)
        model.graph.initializer.append(
            numpy_helper.from_array(weight, name + '_weight'))
        nodes.extend([
            helper.make_node(
                'Concat',
                list(node.input), [name + '_cat'],
                name=name + '_cat',
                axis=1),
            helper.make_node(
                'Conv', [name + '_cat', name + '_weight'],
                list(node.output),
                name=name,
                kernel_shape=[1, 1],
                strides=[1, 1],
                pads=[0, 0, 0, 0],
                group=1),
        ])
        replacements.append(
            dict(node=node.name, module=name, channels=channels))
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    onnx.checker.check_model(model)
    return model, replacements
