# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.utils import setup_cache_size_limit_of_dynamo
from mmengine.config import Config, ConfigDict, DictAction
from mmengine.evaluator import DumpResults
from mmengine.runner import Runner
from mmengine.model import is_model_wrapper
from mmyolo.models.dense_heads import YOLOv5HeadModule, YOLOv7HeadModule, YOLOv6HeadModule, YOLOv8HeadModule

from mmyolo.registry import RUNNERS
from mmyolo.utils import is_metainfo_lower

from edgeai_torchmodelopt import xmodelopt

# TODO: support fuse_conv_bn
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMYOLO test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--model-surgery',
        type=int,
        default=0,
        help='create lite version of a model by applying a set of fx based transformations on the model')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--out',
        type=str,
        help='output result file (must be a .pkl file) in pickle format')
    parser.add_argument(
        '--json-prefix',
        type=str,
        help='the prefix of the output json file without perform evaluation, '
        'which is useful when you want to format the result to a specific '
        'format and submit it to the test server')
    parser.add_argument(
        '--tta',
        action='store_true',
        help='Whether to use test time augmentation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--deploy',
        action='store_true',
        help='Switch model to deployment mode')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
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

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.deploy:
        cfg.custom_hooks.append(dict(type='SwitchToDeployHook'))

    # add `format_only` and `outfile_prefix` into cfg
    if args.json_prefix is not None:
        cfg_json = {
            'test_evaluator.format_only': True,
            'test_evaluator.outfile_prefix': args.json_prefix
        }
        cfg.merge_from_dict(cfg_json)

    # Determine whether the custom metainfo fields are all lowercase
    is_metainfo_lower(cfg)

    if args.tta:
        assert 'tta_model' in cfg, 'Cannot find ``tta_model`` in config.' \
                                   " Can't use tta !"
        assert 'tta_pipeline' in cfg, 'Cannot find ``tta_pipeline`` ' \
                                      "in config. Can't use tta !"

        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        test_data_cfg = cfg.test_dataloader.dataset
        while 'dataset' in test_data_cfg:
            test_data_cfg = test_data_cfg['dataset']

        # batch_shapes_cfg will force control the size of the output image,
        # it is not compatible with tta.
        if 'batch_shapes_cfg' in test_data_cfg:
            test_data_cfg.batch_shapes_cfg = None
        test_data_cfg.pipeline = cfg.tta_pipeline

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # add `DumpResults` dummy metric
    if args.out is not None:
        assert args.out.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        runner.test_evaluator.metrics.append(
            DumpResults(out_file_path=args.out))
    if args.model_surgery:
        surgery_fn = xmodelopt.surgery.v1.convert_to_lite_model if args.model_surgery == 1 \
                     else (xmodelopt.surgery.v2.convert_to_lite_fx if args.model_surgery == 2 else None)
        runner.model.eval()
        if is_model_wrapper(runner.model):
            runner.model = runner.model.module
        # start testing
        runner.model.backbone = surgery_fn(runner.model.backbone)
        runner.model.neck = surgery_fn(runner.model.neck)

        # Only head_module of head goes through model_surgery as it contains all compute layers
        if not isinstance(runner.model.bbox_head.head_module, (YOLOv5HeadModule, YOLOv7HeadModule, YOLOv8HeadModule, YOLOv6HeadModule)):
            runner.model.bbox_head.head_module = \
                surgery_fn(runner.model.bbox_head.head_module)
            if hasattr(runner.model.bbox_head.head_module, 'reg_max') and hasattr(runner.model.bbox_head.head_module, 'proj'):
                    reg_max = runner.model.bbox_head.head_module.reg_max
                    proj = runner.model.bbox_head.head_module.proj
            else:
                    reg_max = None
                    proj = None
            if reg_max is not None and proj is not None:
                runner.model.bbox_head.head_module.reg_max = reg_max
                runner.model.bbox_head.head_module.proj = proj
        elif isinstance(runner.model.bbox_head.head_module, (YOLOv8HeadModule, YOLOv6HeadModule)):
            runner.model.bbox_head.head_module = xmodelopt.surgery.v1.convert_to_lite_model(runner.model.bbox_head.head_module)
        runner.model = runner.wrap_model(runner.cfg.get('model_wrapper_cfg'), runner.model)
    print("\n\n model summary : \n",runner.model)   
    runner.test()


if __name__ == '__main__':
    main()
