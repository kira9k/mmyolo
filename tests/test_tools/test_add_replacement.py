from io import BytesIO
from unittest import TestCase

import onnx
import onnxruntime as ort
import torch
from torch import nn

from projects.easydeploy.model.add_replacement import (replace_exported_adds,
                                                       replace_onnx_adds,
                                                       sum_conv_weight)


class TestAddReplacement(TestCase):

    def test_weight_is_channelwise_sum(self):
        for channels in (1, 4, 17):
            for alpha in (1, 2, -0.5):
                a, b = torch.randn(1, channels, 3,
                                   5), torch.randn(1, channels, 3, 5)
                w = sum_conv_weight(channels, alpha)
                actual = torch.nn.functional.conv2d(torch.cat((a, b), 1), w)
                torch.testing.assert_close(actual, a + alpha * b)

    def test_exported_chain_preserves_alpha_and_users(self):

        class Model(nn.Module):

            def forward(self, a, b):
                z = torch.add(a, b, alpha=2)
                return z + a, z

        a, b = torch.randn(1, 4, 3, 5), torch.randn(1, 4, 3, 5)
        reference = Model().eval()
        exported = torch.export.export(reference, (a, b)).module()
        rows = replace_exported_adds(exported)
        self.assertEqual(len(rows), 2)
        torch.testing.assert_close(exported(a, b), reference(a, b))
        self.assertTrue(
            all(not p.requires_grad for p in exported.parameters()))
        self.assertFalse(
            any(
                str(n.target) == 'aten.add.Tensor'
                for n in exported.graph.nodes))
        self.assertEqual(replace_exported_adds(exported), [])
        reexported = torch.export.export(exported, (a, b)).module()
        torch.testing.assert_close(reexported(a, b), reference(a, b))

    def test_non_feature_additions_are_not_replaced(self):

        class Model(nn.Module):

            def forward(self, a, b, c):
                return a + b, a + 1, c + c

        inputs = (torch.randn(1, 4, 3, 5), torch.randn(1, 4, 1,
                                                       1), torch.randn(2, 4))
        model = Model().eval()
        exported = torch.export.export(model, inputs).module()
        self.assertEqual(replace_exported_adds(exported), [])
        torch.testing.assert_close(exported(*inputs), model(*inputs))

    def test_onnx_replacement_preserves_names_and_outputs(self):

        class Model(nn.Module):

            def forward(self, a, b):
                return a + b, a.flatten(1) + b.flatten(1)

        a, b = torch.randn(1, 4, 3, 5), torch.randn(1, 4, 3, 5)
        stream = BytesIO()
        torch.onnx.export(
            Model().eval(), (a, b),
            stream,
            opset_version=11,
            input_names=['a', 'b'],
            output_names=['sum', 'flat'])
        original = onnx.load_from_string(stream.getvalue())
        graph, rows = replace_onnx_adds(original)
        self.assertEqual(len(rows), 1)
        self.assertEqual([v.name for v in graph.graph.output], ['sum', 'flat'])
        self.assertEqual(sum(n.op_type == 'Add' for n in graph.graph.node), 1)
        self.assertEqual(sum(n.op_type == 'Conv' for n in graph.graph.node), 1)
        onnx.checker.check_model(graph)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(
            graph.SerializeToString(),
            options,
            providers=['CPUExecutionProvider'])
        actual = session.run(None, {'a': a.numpy(), 'b': b.numpy()})
        for value, expected in zip(actual, Model()(a, b)):
            torch.testing.assert_close(torch.from_numpy(value), expected)
        _, again = replace_onnx_adds(graph)
        self.assertEqual(again, [])
