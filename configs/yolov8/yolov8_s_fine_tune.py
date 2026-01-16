_base_ = ['../_base_/default_runtime.py', '../_base_/det_p5_tta.py']

# ======================== Основные параметры ========================
data_root = 'data/combined_30percent_2025_v2/' 
data_root_test = 'data/combined_30percent_2025_v2/' #'data/test_data/'  

train_ann_file = 'train/ann.json'
train_data_prefix = 'train/images' 

val_ann_file = 'val/ann.json'
val_data_prefix = 'val/images' 

test_ann_file = 'val/ann.json'
test_data_prefix = 'val/images'
num_classes = 1  

train_batch_size_per_gpu = 32
train_num_workers = 8

val_batch_size_per_gpu = 1
val_num_workers = 4

batch_shapes_cfg = None
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)  # Normalization config

dataset_type = 'YOLOv5CocoDataset'

# persistent_workers must be False if num_workers is 0
persistent_workers = True

load_from = '/home/kira9k/ieos/mmyolo/work_dirs/yolov8_s_big_dataset_2/best_coco_bbox_mAP_epoch_210.pth'

# Обучение
max_epochs = 100
close_mosaic_epochs = 10       # последние 10 эпох без Mosaic
base_lr = 0.001               
lr_factor = 0.1

#loss_cls_weight = 0.5

loss_cls_weight = 1.0
loss_bbox_weight = 6.0
loss_dfl_weight = 0.5


#loss_bbox_weight = 0.05#7.5
#loss_dfl_weight = 0.15#1.5 / 4

model_test_cfg = dict(
    # The config of multi-label for multi-class prediction.
    multi_label=False,
    # The number of boxes before NMS
    nms_pre=30000,
    score_thr=0.35, #0.001 # Threshold to filter out boxes.
    nms=dict(type='nms', iou_threshold=0.6),  # NMS type and threshold #0.7
    max_per_img=100) #300 

# ======================== Модель (YOLOv8-s) ========================
deepen_factor = 0.33
widen_factor = 0.50
last_stage_out_channels = 1024
strides = [8, 16, 32]

model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[1., 1., 1.], 
        bgr_to_rgb=True),
    backbone=dict(
        type='YOLOv8CSPDarknet',
        arch='P5',
        last_stage_out_channels=last_stage_out_channels,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True),
        frozen_stages=3,    
        norm_eval=True),          
    neck=dict(
        type='YOLOv8PAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, 1024],
        out_channels=[256, 512, 1024],
        num_csp_blocks=3,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True),
        ),
    bbox_head=dict(
        type='YOLOv8Head',
        head_module=dict(
            type='YOLOv8HeadModule',
            num_classes=num_classes,
            in_channels=[256, 512, last_stage_out_channels],
            widen_factor=widen_factor,
            reg_max=16,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='SiLU', inplace=True),
            featmap_strides=strides),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0.5, strides=strides),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        # scaled based on number of detection layers
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='none',
            loss_weight=loss_cls_weight),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xyxy',
            reduction='sum',
            loss_weight=loss_bbox_weight,
            return_iou=False),
        loss_dfl=dict(
            type='mmdet.DistributionFocalLoss',
            reduction='mean',
            loss_weight=loss_dfl_weight)),
    train_cfg=dict(assigner=dict(
        type='BatchTaskAlignedAssigner',
        num_classes=num_classes,
        topk=10,
        alpha=0.5,
        beta=6.0)),
    test_cfg=model_test_cfg)

# ======================== Аугментации ========================
img_scale = (640, 640)
affine_scale = 0.9
max_aspect_ratio = 100

albu_train_transforms = [
    dict(type='Blur', p=0.05),
    ##dict(type='MedianBlur', p=0.05),
    #dict(type='GaussNoise', p=0.1),
    dict(type='RandomBrightnessContrast', brightness_limit=0.15, contrast_limit=0.2, p=0.5),
    dict(type='HueSaturationValue', hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
]

pre_transform = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True)
]

last_transform = [
    dict(type='mmdet.Albu',
         transforms=albu_train_transforms,
         bbox_params=dict(type='BboxParams', format='pascal_voc', label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
         keymap={'img': 'image', 'gt_bboxes': 'bboxes'}),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip', 'flip_direction'))
]

train_pipeline = [
    *pre_transform,
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0, pre_transform=pre_transform),
    dict(type='YOLOv5RandomAffine',
         max_rotate_degree=10,
         max_shear_degree=2,
         scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
         border=(-img_scale[0]//2, -img_scale[1]//2),
         border_val=(114, 114, 114)),
    *last_transform
]

train_pipeline_stage2 = [
    *pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(type='LetterResize', scale=img_scale, allow_scale_up=True, pad_val=dict(img=114.0)),
    dict(type='YOLOv5RandomAffine',
         max_rotate_degree=10,
         max_shear_degree=2,
         scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
         border_val=(114, 114, 114)),
    *last_transform
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]
# ======================== Даталоадеры ========================
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='yolov5_collate'),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=('human',)),
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline))


val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=('human',)),
        data_root=data_root,
        test_mode=True,
        data_prefix=dict(img=val_data_prefix),
        ann_file=val_ann_file,
        pipeline=test_pipeline,
        batch_shapes_cfg=batch_shapes_cfg))

test_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True, 
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=('human',)),
        data_root=data_root_test,
        test_mode=True,
        data_prefix=dict(img=test_data_prefix),
        ann_file=test_ann_file,              # ← именно тестовая разметка
        pipeline=test_pipeline,
        batch_shapes_cfg=batch_shapes_cfg)
    )
# ======================== Оптимизатор и шедулер ========================
optim_wrapper = dict(
    type='OptimWrapper',
    clip_grad=dict(max_norm=10.0),
    optimizer=dict(
        type='SGD',
        lr=base_lr,               # 0.001
        momentum=0.937,
        weight_decay=0.0005,
        nesterov=True),
    constructor='YOLOv5OptimizerConstructor',
    paramwise_cfg=dict(
        # Размораживаем только head + neck на старте
        base_total_batch_size=train_batch_size_per_gpu,
        custom_keys={
            '.backbone': dict(lr_mult=0.005),   
            '.neck': dict(lr_mult=0.2),       
            '.bbox_head': dict(lr_mult=1.0),  
        }))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(type='ForceBNToEvalHook', priority='VERY_HIGH'),]

# Шедулер
param_scheduler = [
    dict(type='LinearLR', start_factor=0.05, by_epoch=False, begin=0, end=1000),
    dict(type='CosineAnnealingLR', T_max=max_epochs, eta_min=base_lr*0.05, begin=5, end=max_epochs)
]

# ======================== Остальное ========================
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=5,
    dynamic_intervals=[(max_epochs - close_mosaic_epochs, 1)]
)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        max_keep_ckpts=3,
        save_best='auto'),
    param_scheduler=dict(type='YOLOv5ParamSchedulerHook'),
    logger=dict(type='LoggerHook', interval=50),
)

val_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=data_root + val_ann_file,
    metric='bbox'
)

test_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=data_root_test + test_ann_file,   # ← важно указать именно тестовый ann.json
    metric='bbox'
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')