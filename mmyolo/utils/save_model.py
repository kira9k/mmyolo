# Copyright (c) 2018-2021, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import numpy as np
import torch
import onnx
from .proto import mmyolo_meta_arch_pb2
from google.protobuf import text_format
from mmyolo.models.dense_heads import (RTMDetHead, YOLOv5Head, YOLOv7Head, YOLOv8Head, YOLOv6Head,
                                       YOLOXHead)


__all__ = ['save_model_proto']


def save_model_proto(model, onnx_model, input, output_filename, input_names=None, feature_names=None, output_names=None,
                     export_type=None):
    export_type = export_type or 'MMYOLO' # default is 'MMYOLO'
    output_filename = os.path.splitext(output_filename)[0] + '.prototxt'
    input = input[0] if isinstance(input, (list,tuple)) and \
                                  isinstance(input[0], (torch.Tensor, np.ndarray)) else input
    input_size = input.size() if isinstance(input, torch.Tensor) else input
    output_names = output_names or ['num_dets', 'boxes', 'scores', 'labels']

    is_yolov5 = hasattr(model, 'bbox_head') and isinstance(model.bbox_head, YOLOv5Head)
    is_yolox = hasattr(model, 'bbox_head') and isinstance(model.bbox_head, YOLOXHead)
    is_yolov6 = hasattr(model, 'bbox_head') and isinstance(model.bbox_head, YOLOv6Head)
    is_yolov7 = hasattr(model, 'bbox_head') and isinstance(model.bbox_head, YOLOv7Head)
    is_yolov8 = hasattr(model, 'bbox_head') and isinstance(model.bbox_head, YOLOv8Head)
    is_yolov5 = False if is_yolov8 or is_yolox or is_yolov6 or is_yolov7 else is_yolov5
    input_names = input_names or ('input',)
    if is_yolov5:
        feature_names = prepare_model_for_layer_outputs(onnx_model, export_layer_types='Conv', match_layer = 'Reshape',
                                                        return_layer='Conv')
        _save_mmyolo_proto_yolov5(model, input_size, output_filename, feature_names, output_names, export_type=export_type)
    elif is_yolox:
        feature_names = prepare_model_for_layer_outputs(onnx_model, export_layer_types='Conv', match_layer = 'Concat',
                                                        return_layer='Concat')
        _save_mmyolo_proto_yolox(model, input_size, output_filename, feature_names, output_names, export_type=export_type)
    elif is_yolov6:
        feature_names = prepare_model_for_layer_outputs(onnx_model, export_layer_types='Conv', match_layer = 'Concat',
                                                        return_layer='Concat')
        _save_mmyolo_proto_yolov6(model, input_size, output_filename, feature_names, output_names, export_type=export_type)
    elif is_yolov7:
        feature_names = prepare_model_for_layer_outputs(onnx_model, export_layer_types='Conv', match_layer = 'Mul',
                                                        return_layer='Mul')
        _save_mmyolo_proto_yolov7(model, input_size, output_filename, feature_names, output_names, export_type=export_type)
    elif is_yolov8:
        feature_names = prepare_model_for_layer_outputs(onnx_model, export_layer_types='Conv', match_layer = 'Concat',
                                                        return_layer = 'Conv')
        _save_mmyolo_proto_yolov8_caffessd_format(model, input_size, output_filename, feature_names, output_names, export_type=export_type)

    return output_filename
    #


###########################################################
def _create_rand_inputs(input_size, is_cuda=False):
    x = torch.rand(input_size)
    x = x.cuda() if is_cuda else x
    return x

###########################################################
def _save_mmyolo_onnx(model, input_list, output_filename, input_names=None, proto_names=None, output_names=None,
                     opset_version=11):
    #https://github.com/open-mmlab/mmyoloection/pull/1082
    assert hasattr(model, 'forward_dummy'), 'wrting onnx is not supported by this model'
    model.eval()
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    forward_backup = model.forward #backup forward
    model.forward = model.forward_dummy #set dummy forward
    torch.onnx.export(
        model,
        input_list,
        output_filename,
        input_names=input_names,
        output_names=proto_names,
        export_params=True,
        keep_initializers_as_inputs=True,
        do_constant_folding=True,
        verbose=False,
        opset_version=opset_version)
    model.forward = forward_backup #restore forward


###########################################################
def prepare_model_for_layer_outputs(onnx_model, export_layer_types=None, match_layer=None, return_layer=None):
    layer_output_names = []
    for i in range(len(onnx_model.graph.node)):
        output_0 = onnx_model.graph.node[i].output[0]
        if onnx_model.graph.node[i].op_type in export_layer_types:
            for j in range(len(onnx_model.graph.node)):
                if output_0 in onnx_model.graph.node[j].input:
                    if onnx_model.graph.node[j].op_type == match_layer:
                        if return_layer not in export_layer_types:
                            if onnx_model.graph.node[j].output[0] not in layer_output_names:
                                layer_output_names.append(onnx_model.graph.node[j].output[0])
                        else:
                            if onnx_model.graph.node[i].output[0] not in layer_output_names:
                                layer_output_names.append(onnx_model.graph.node[i].output[0])
    return layer_output_names

###########################################################
def _save_mmyolo_proto_yolov5(model, input_size, output_filename, proto_names=None, output_names=None, export_type='MMYOLO'):
    bbox_head = model.bbox_head
    base_sizes = model.bbox_head.prior_generator.base_sizes
    background_label_id = -1
    num_classes = bbox_head.num_classes

    yolo_params = []
    for base_size_id, base_size in enumerate(base_sizes):
        yolo_param = mmyolo_meta_arch_pb2.TIDLYoloParams(input=proto_names[base_size_id],
                                                        anchor_width=[base_size[idx][0] for idx in range(len(base_size))],
                                                        anchor_height=[base_size[idx][1] for idx in range(len(base_size))],
                                                         )
        yolo_params.append(yolo_param)

    nms_param = mmyolo_meta_arch_pb2.TIDLNmsParam(nms_threshold=0.45, top_k=200)
    detection_output_param = mmyolo_meta_arch_pb2.TIDLOdPostProc(num_classes=num_classes, share_location=True,
                                            background_label_id=background_label_id, nms_param=nms_param,
                                            code_type=mmyolo_meta_arch_pb2.CODE_TYPE_YOLO_V5, keep_top_k=200,
                                            confidence_threshold=0.3)

    yolov5 = mmyolo_meta_arch_pb2.TidlYoloOd(name='yolov5', output=output_names,
                                            in_width=input_size[3], in_height=input_size[2],
                                            yolo_param=yolo_params,
                                            detection_output_param=detection_output_param,
                                            framework=export_type)

    arch = mmyolo_meta_arch_pb2.TIDLMetaArch(name='yolov5',  tidl_yolo=[yolov5])

    with open(output_filename, 'wt') as pfile:
        txt_message = text_format.MessageToString(arch)
        pfile.write(txt_message)


###########################################################
def _save_mmyolo_proto_yolox(model, input_size, output_filename, proto_names=None, output_names=None, export_type='MMYOLO'):
    bbox_head = model.bbox_head
    base_sizes = model.bbox_head.featmap_strides

    background_label_id = -1
    num_classes = bbox_head.num_classes

    yolo_params = []
    for base_size_id, base_size in enumerate(base_sizes):
        yolo_param = mmyolo_meta_arch_pb2.TIDLYoloParams(input=proto_names[base_size_id],
                                                        anchor_width=[base_size],
                                                        anchor_height=[base_size])
        yolo_params.append(yolo_param)

    nms_param = mmyolo_meta_arch_pb2.TIDLNmsParam(nms_threshold=0.45, top_k=200)
    detection_output_param = mmyolo_meta_arch_pb2.TIDLOdPostProc(num_classes=num_classes, share_location=True,
                                            background_label_id=background_label_id, nms_param=nms_param,
                                            code_type=mmyolo_meta_arch_pb2.CODE_TYPE_YOLO_X, keep_top_k=200,
                                            confidence_threshold=0.3)

    yolox = mmyolo_meta_arch_pb2.TidlYoloOd(name='yolox', output=output_names,
                                            in_width=input_size[3], in_height=input_size[2],
                                            yolo_param=yolo_params,
                                            detection_output_param=detection_output_param,
                                            framework=export_type)

    arch = mmyolo_meta_arch_pb2.TIDLMetaArch(name='yolox',  tidl_yolo=[yolox])

    with open(output_filename, 'wt') as pfile:
        txt_message = text_format.MessageToString(arch)
        pfile.write(txt_message)


###########################################################
def _save_mmyolo_proto_yolov6(model, input_size, output_filename, proto_names=None, output_names=None, export_type='MMYOLO'):
    bbox_head = model.bbox_head
    base_sizes = model.bbox_head.featmap_strides

    background_label_id = -1
    num_classes = bbox_head.num_classes

    yolo_params = []
    for base_size_id, base_size in enumerate(base_sizes):
        yolo_param = mmyolo_meta_arch_pb2.TIDLYoloParams(input=proto_names[base_size_id],
                                                        anchor_width=[base_size],
                                                        anchor_height=[base_size])
        yolo_params.append(yolo_param)

    nms_param = mmyolo_meta_arch_pb2.TIDLNmsParam(nms_threshold=0.45, top_k=200)
    detection_output_param = mmyolo_meta_arch_pb2.TIDLOdPostProc(num_classes=num_classes, share_location=True,
                                            background_label_id=background_label_id, nms_param=nms_param,
                                            code_type=mmyolo_meta_arch_pb2.CODE_TYPE_YOLO_X, keep_top_k=200,
                                            confidence_threshold=0.3)

    yolov6 = mmyolo_meta_arch_pb2.TidlYoloOd(name='yolov6', output=output_names,
                                            in_width=input_size[3], in_height=input_size[2],
                                            yolo_param=yolo_params,
                                            detection_output_param=detection_output_param,
                                            framework=export_type)

    arch = mmyolo_meta_arch_pb2.TIDLMetaArch(name='yolov6',  tidl_yolo=[yolov6])
    #YOLOv6 architecture is currently not supported in TIDL. We need to pass the information that the objectness score is not
    #present in this model. O/W funationality is same as YOLOX.
    with open(output_filename, 'wt') as pfile:
        txt_message = text_format.MessageToString(arch)
        pfile.write(txt_message)


###########################################################
def _save_mmyolo_proto_yolov7(model, input_size, output_filename, proto_names=None, output_names=None, export_type='MMYOLO'):
    bbox_head = model.bbox_head
    base_sizes = model.bbox_head.prior_generator.base_sizes

    background_label_id = -1
    num_classes = bbox_head.num_classes

    yolo_params = []
    for base_size_id, base_size in enumerate(base_sizes):
        yolo_param = mmyolo_meta_arch_pb2.TIDLYoloParams(input=proto_names[base_size_id],
                                                        anchor_width=[base_size[idx][0] for idx in range(len(base_size))],
                                                        anchor_height=[base_size[idx][1] for idx in range(len(base_size))],
                                                         )
        yolo_params.append(yolo_param)

    nms_param = mmyolo_meta_arch_pb2.TIDLNmsParam(nms_threshold=0.45, top_k=200)
    detection_output_param = mmyolo_meta_arch_pb2.TIDLOdPostProc(num_classes=num_classes, share_location=True,
                                            background_label_id=background_label_id, nms_param=nms_param,
                                            code_type=mmyolo_meta_arch_pb2.CODE_TYPE_YOLO_V5, keep_top_k=200,
                                            confidence_threshold=0.3)

    yolov7 = mmyolo_meta_arch_pb2.TidlYoloOd(name='yolov7', output=output_names,
                                            in_width=input_size[3], in_height=input_size[2],
                                            yolo_param=yolo_params,
                                            detection_output_param=detection_output_param,
                                            framework=export_type)

    arch = mmyolo_meta_arch_pb2.TIDLMetaArch(name='yolov7',  tidl_yolo=[yolov7])

    with open(output_filename, 'wt') as pfile:
        txt_message = text_format.MessageToString(arch)
        pfile.write(txt_message)


###########################################################
def _save_mmyolo_proto_yolov8(model, input_size, output_filename, proto_names=None, output_names=None, export_type='MMYOLO'):
    bbox_head = model.bbox_head
    base_sizes = model.bbox_head.featmap_strides

    background_label_id = -1
    num_classes = bbox_head.num_classes

    yolo_params = []
    for base_size_id, base_size in enumerate(base_sizes):
        yolo_param = mmyolo_meta_arch_pb2.TIDLYoloParams(input=proto_names[base_size_id],
                                                        anchor_width=[base_size],
                                                        anchor_height=[base_size])
        yolo_params.append(yolo_param)

    nms_param = mmyolo_meta_arch_pb2.TIDLNmsParam(nms_threshold=0.45, top_k=200)
    detection_output_param = mmyolo_meta_arch_pb2.TIDLOdPostProc(num_classes=num_classes, share_location=True,
                                            background_label_id=background_label_id, nms_param=nms_param,
                                            code_type=mmyolo_meta_arch_pb2.CODE_TYPE_YOLO_V5, keep_top_k=200,
                                            confidence_threshold=0.3)

    yolov8 = mmyolo_meta_arch_pb2.TidlYoloOd(name='yolov8', output=output_names,
                                            in_width=input_size[3], in_height=input_size[2],
                                            yolo_param=yolo_params,
                                            detection_output_param=detection_output_param,
                                            framework=export_type)

    arch = mmyolo_meta_arch_pb2.TIDLMetaArch(name='yolov8',  tidl_yolo=[yolov8])
    #YOLOv8 architecture is currently not supported in TIDL. We need to pass the information that the objectness score is not
    #present in this model. Apart from that, the bbox decoding has o be taken care of.
    with open(output_filename, 'wt') as pfile:
        txt_message = text_format.MessageToString(arch)
        pfile.write(txt_message)


###########################################################
def _save_mmyolo_proto_yolov8_caffessd_format(model, input_size, output_filename, proto_names=None, output_names=None, export_type='MMYOLO'):
    bbox_head = model.bbox_head
    base_sizes = model.bbox_head.featmap_strides
    background_label_id = -1
    num_classes = bbox_head.num_classes
    class_input =[]
    box_input = []

    #separating class_input and box_input layers from  proto_names list
    for i in range(len(base_sizes)):
        class_input.append(proto_names[i*2])
        box_input.append(proto_names[i*2+1])

    nms_param = mmyolo_meta_arch_pb2.TIDLNmsParam(nms_threshold=0.45, top_k=200)
    detection_output_param = mmyolo_meta_arch_pb2.TIDLOdPostProc(num_classes=num_classes, share_location=True,
                                            background_label_id=background_label_id, nms_param=nms_param,
                                            code_type=mmyolo_meta_arch_pb2.CODE_TYPE_DIST2BBOX, keep_top_k=200,
                                            confidence_threshold=0.3)

    caffe_ssd_param = mmyolo_meta_arch_pb2.TidlMaCaffeSsd(name='yolov8',class_input=class_input,box_input=box_input,
                                                          in_width=input_size[3], in_height=input_size[2],
                                                          detection_output_param=detection_output_param,
                                                          output=output_names,score_converter='SIGMOID')

    arch = mmyolo_meta_arch_pb2.TIDLMetaArch(name='yolov8',  caffe_ssd=[caffe_ssd_param])
    #YOLOv8 architecture is currently not supported in TIDL. We need to pass the information that the objectness score is not
    #present in this model. Apart from that, the bbox decoding has o be taken care of.
    with open(output_filename, 'wt') as pfile:
        txt_message = text_format.MessageToString(arch)
        pfile.write(txt_message)