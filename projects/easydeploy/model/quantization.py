from fnmatch import fnmatchcase

import torch
import torch.nn as nn


def normalize_module_name(name):
    if name.startswith("L['self']."):
        name = name[len("L['self']."):]
    if name == 'baseModel':
        return ''
    if name.startswith('baseModel.'):
        return name[len('baseModel.'):]
    if name == 'baseHead' or name.startswith('baseHead.'):
        return 'bbox_head' + name[len('baseHead'):]
    return name


def _within(name, roots):
    return any(name == root or name.startswith(root + '.') for root in roots)


def select_fp32_modules(model, patterns=(), keep_last=False):
    modules = dict(model.named_modules())
    convs = {
        name: module
        for name, module in modules.items() if isinstance(module, nn.Conv2d)
    }
    selected = set()
    for pattern in patterns:
        pattern = normalize_module_name(pattern)
        matches = {
            name
            for name in modules if name and fnmatchcase(name, pattern) and any(
                _within(conv, (name, )) for conv in convs)
        }
        if not matches:
            raise ValueError(
                f'No convolution or convolution block matches: {pattern}')
        selected.update(matches)
    if keep_last:
        names_by_id = {id(module): name for name, module in modules.items()}
        head = model.bbox_head.head_module
        for branches in (head.cls_preds, head.reg_preds):
            for branch in branches:
                if not isinstance(branch[-1], nn.Conv2d):
                    raise ValueError(
                        'Expected a final Conv2d in each cls/reg branch')
                selected.add(names_by_id[id(branch[-1])])
    roots = []
    for name in sorted(selected, key=lambda name: (name.count('.'), name)):
        if not _within(name, roots):
            roots.append(name)
    return sorted(roots), sorted(
        name for name in convs if _within(name, roots))


def _module_scopes(node):
    # FX surgery retains relative paths inside backbone/neck GraphModules.
    context = ''
    for path, _ in node.meta.get('nn_module_stack', {}).values():
        raw = path[len("L['self']."):] if path.startswith(
            "L['self'].") else path
        canonical = normalize_module_name(raw)
        if raw.startswith(('baseModel.', 'baseHead.')):
            if canonical in ('backbone', 'neck'):
                context = canonical
        elif canonical and context and not canonical.startswith(context + '.'):
            canonical = context + '.' + canonical
        yield raw, canonical


def convolution_module_name(node):
    if 'mmyolo_module_name' in node.meta:
        return node.meta['mmyolo_module_name']
    scopes = [scope for scope in _module_scopes(node) if scope[1]]
    if scopes:
        return scopes[-1][1]
    weight = node.args[1] if len(node.args) > 1 else None
    if isinstance(weight, torch.fx.Node) and weight.op == 'get_attr':
        return normalize_module_name(
            str(weight.target).removesuffix('.weight'))
    return None


def _float_output_cats(graph, roots):
    output = next(node for node in graph.nodes if node.op == 'output')
    leaves, _ = torch.utils._pytree.tree_flatten(output.args[0])
    result = []
    for node in leaves:
        if not isinstance(node, torch.fx.Node) or str(
                node.target) != 'aten.cat.default':
            continue
        if any(
                _within(convolution_module_name(value) or '', roots) or any(
                    _within(canonical, roots)
                    for _, canonical in _module_scopes(value))
                for value in node.args[0]):
            result.append(node)
    return result


def tensor_add_nodes(graph):
    return [
        node for node in graph.nodes
        if node.op == 'call_function' and str(node.target) in (
            'aten.add.Tensor', 'aten.add_.Tensor')
    ]


def add_adjacent_convolutions(exported):
    """Find the first Conv on every tensor path into/out of each Add."""
    selected = set()
    for add in tensor_add_nodes(exported.graph):
        for forward in (False, True):
            pending = list(add.users if forward else add.all_input_nodes)
            visited = set()
            while pending:
                node = pending.pop()
                if node in visited:
                    continue
                visited.add(node)
                if node.op == 'call_function' and str(
                        node.target) == 'aten.conv2d.default':
                    weight = node.args[1]
                    if not isinstance(
                            weight, torch.fx.Node) or weight.op != 'get_attr':
                        raise RuntimeError(
                            'Cannot resolve an Add-adjacent convolution weight'
                        )
                    selected.add(
                        normalize_module_name(
                            str(weight.target).removesuffix('.weight')))
                    continue
                # Ignore parameter/shape arithmetic, but follow tensor tuples
                # from Split/BatchNorm as well as ordinary activation tensors.
                values, _ = torch.utils._pytree.tree_flatten(
                    node.meta.get('val'))
                if node.op in ('get_attr', 'output') or not any(
                        isinstance(value, torch.Tensor) for value in values):
                    continue
                pending.extend(node.users if forward else node.all_input_nodes)
    return sorted(selected)


def validate_fp32_adds(model):
    nodes = tensor_add_nodes(model.graph)
    annotated = [
        node.name for node in nodes if getattr(
            node.meta.get('quantization_annotation'), '_annotated', False)
    ]
    if annotated:
        raise RuntimeError(
            f'Add operators still annotated for INT8: {annotated}')
    return [node.name for node in nodes]


def make_quantizer(quantizer_cls,
                   qconfig,
                   exported,
                   fp32_modules,
                   keep_add_fp32=False):
    conv_scopes = set()
    for node in exported.graph.nodes:
        if node.op == 'call_function' and str(
                node.target) == 'aten.conv2d.default':
            weight = node.args[1]
            if weight.op == 'get_attr':
                name = normalize_module_name(
                    str(weight.target).removesuffix('.weight'))
                node.meta['mmyolo_module_name'] = name
                if _within(name, fp32_modules):
                    scopes = list(_module_scopes(node))
                    if scopes:
                        conv_scopes.add(scopes[-1][0])
    if not fp32_modules and not keep_add_fp32:
        return quantizer_cls().set_global(qconfig)
    scopes = {
        raw
        for node in exported.graph.nodes
        for raw, canonical in _module_scopes(node)
        if _within(canonical, fp32_modules)
    }
    scopes.update(conv_scopes)
    if fp32_modules and not scopes:
        raise RuntimeError(
            'No exported operators match the requested FP32 modules')

    class SelectiveQuantizer(quantizer_cls):

        def annotate(self, model):
            model = super().annotate(model)
            # Output-format Concat must not requantize protected cls/DFL values.
            for node in _float_output_cats(model.graph, fp32_modules):
                node.meta.pop('quantization_annotation', None)
            return model

    if keep_add_fp32:
        patterns = getattr(quantizer_cls, 'STATIC_OPS', None)
        if not isinstance(patterns, (list, tuple)) or 'add' not in patterns:
            raise RuntimeError(
                'This XNNPACK quantizer cannot exclude Add patterns')
        # Disable both patterns before annotation, including fused Add + ReLU.
        SelectiveQuantizer.STATIC_OPS = [
            pattern for pattern in patterns
            if pattern not in ('add', 'add_relu')
        ]

    quantizer = SelectiveQuantizer().set_global(qconfig)
    for scope in sorted(scopes):
        try:
            quantizer.set_module_name(scope, None)
        except AssertionError:
            # PyTorch 2.7 supports None in its annotation map but rejects the setter.
            mapping = getattr(quantizer, 'module_name_config', None)
            if not isinstance(mapping, dict):
                raise RuntimeError('This quantizer cannot exclude modules')
            mapping[scope] = None
    return quantizer


def convolution_precisions(model):
    result = []
    for node in model.graph.nodes:
        if node.op != 'call_function' or str(
                node.target) != 'aten.conv2d.default':
            continue
        weight = node.args[1]
        quantized = 'quantized_decomposed.dequantize_' in str(weight.target)
        source = weight.args[0] if quantized else weight
        dtype = None
        if isinstance(source, torch.fx.Node) and source.op == 'get_attr':
            value = model
            for field in str(source.target).split('.'):
                value = getattr(value, field)
            dtype = str(value.dtype).removeprefix('torch.')
        result.append(
            dict(
                node=node.name,
                module=convolution_module_name(node),
                quantized_weight=quantized,
                weight_dtype=dtype))
    return result
