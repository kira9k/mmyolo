# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp

from mmdet.utils import setup_cache_size_limit_of_dynamo
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner
from mmengine.model import is_model_wrapper
from mmyolo.models.dense_heads import YOLOv5HeadModule, YOLOv7HeadModule, YOLOv8HeadModule, YOLOv6HeadModule

from mmyolo.registry import RUNNERS
from mmyolo.utils import is_metainfo_lower

from edgeai_torchmodelopt import xmodelopt
# from edgeai_torchtoolkit import xao
# from edgeai_torchtoolkit import xnn

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--lite',
        action='store_true',
        default=False,
        help='create lite version of a model by applying a set of fx based transformations on the model')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--model-surgery', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    # replace the ${key} with the value of cfg.key
    # cfg = replace_cfg_vals(cfg)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # Determine whether the custom metainfo fields are all lowercase
    is_metainfo_lower(cfg)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    if args.model_surgery:
        surgery_fn = xmodelopt.surgery.v1.convert_to_lite_model if args.model_surgery == 1 \
                     else (xmodelopt.surgery.v2.convert_to_lite_fx if args.model_surgery == 2 else None)
        
        runner._init_model_weights()
        if is_model_wrapper(runner.model):
            runner.model = runner.model.module
        runner.model.backbone = surgery_fn(runner.model.backbone)
        runner.model.neck = surgery_fn(runner.model.neck)
        # Only head_module of head goes through model_surgery as it contains all compute layers
        if not isinstance(runner.model.bbox_head.head_module, (YOLOv5HeadModule, YOLOv7HeadModule, YOLOv8HeadModule, YOLOv6HeadModule)):
            if hasattr(runner.model.bbox_head.head_module, 'reg_max'):
                reg_max = runner.model.bbox_head.head_module.reg_max
            else:
                reg_max = None
            runner.model.bbox_head.head_module = \
                surgery_fn(runner.model.bbox_head.head_module)
            if reg_max is not None:
                runner.model.bbox_head.head_module.reg_max = reg_max
        elif isinstance(runner.model.bbox_head.head_module, (YOLOv8HeadModule, YOLOv6HeadModule)):
            runner.model.bbox_head.head_module = xmodelopt.surgery.v1.convert_to_lite_model(runner.model.bbox_head.head_module)
        runner.model = runner.wrap_model(runner.cfg.get('model_wrapper_cfg'), runner.model)
    print("\n\n model summary : \n",runner.model)
    runner.train()


if __name__ == '__main__':
    main()
