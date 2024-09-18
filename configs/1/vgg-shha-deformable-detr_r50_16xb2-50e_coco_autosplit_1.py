dataset_type = 'CocoDataset'
data_root = 'data/SHTechA/'
work_dir = './work_dirs/vgg-shha-deformable-detr_r50_16xb2-50e_coco_split_1'
test_out_dir=f'{work_dir}/vis_test'
backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotationsCrowd', with_bbox=True),
    dict(type='RandomFlipCrowd', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[[{
            'type':
            'RandomChoiceResizeCrowd',
            'scales': [(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                       (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                       (736, 1333), (768, 1333), (800, 1333)
            ],
            'keep_ratio':
            True
        }, {
            'type': 'RandomCropCrowd',
            'crop_type': 'absolute_range',
            'crop_size':  (256, 256),# (384, 600),
            'allow_negative_crop': True
        }]]),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    # dict(type='Resize', scale=(1333, 704), keep_ratio=True), # scale=(1333, 800),
    dict(type='LoadAnnotationsCrowd', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='CocoCrowdDataset',
        data_root=data_root,
        ann_file='coco_train_data.json',
        data_prefix=dict(img='train_data/images_split/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline,
        backend_args=None))
val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='coco_test_data_split.json',
        data_prefix=dict(img='test_data/images_split/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            # dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ],
        backend_args=None))
test_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='coco_test_data_split.json',
        data_prefix=dict(img='test_data/images_split/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=None))
val_evaluator = dict(
    type='Crowd_Split_CocoMetric',
    ann_file=f'{data_root}coco_test_data_split.json',
    metric='mae',
    test_out_dir=test_out_dir,
    format_only=False,
    backend_args=None)
test_evaluator = dict(
    type='Crowd_Split_CocoMetric',
    ann_file=f'{data_root}coco_test_data_split.json',
    metric='mae',
    test_out_dir=test_out_dir,
    format_only=False,
    backend_args=None)
default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=2),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Crowd_DetVisualizationHook', draw=False, test_out_dir=test_out_dir))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
classes = 'person'
model = dict(
    type='Crowd_DeformableDETR',
    num_queries=300,
    num_feature_levels=4,
    query_feature_level=[0],
    with_box_refine=False,
    as_two_stage=False,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='VGG',
        depth=16,
        with_bn=True,
        num_stages=5,
        out_indices=(2, 3, 4)),
    neck=dict(
        type='ChannelMapper',
        in_channels=[256, 512, 512],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    encoder=dict(
        num_layers=2,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, batch_first=True),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.1)),
    ),
    decoder=dict(
        num_layers=2,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256, num_heads=8, dropout=0.1, batch_first=True),
            cross_attn_cfg=dict(embed_dims=256, batch_first=True),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.1)),
        post_norm_cfg=None),
    positional_encoding=dict(num_feats=128, normalize=True, offset=-0.5),
    bbox_head=dict(
        type='Crowd_DeformableDETRHead',
        num_classes=1,
        num_points=1,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='CrossEntropyLoss',
            ignore_index=-1,
            class_weight=[0.5, 1.0],
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    train_cfg=dict(
        assigner=dict(
            type='Crowd_HungarianAssigner',
            match_costs=[
                dict(type='ScoreLossCost', weight=1.0),
                dict(type='PointLossCost', weight=0.05)
            ])),
    test_cfg=dict(max_per_img=4000))
optim_wrapper = dict(
    type='OptimWrapper',
    constructor='CrowdOptimWrapperConstructor',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.1),
            sampling_offsets=dict(lr_mult=0.1),
            reference_points=dict(lr_mult=0.1))))
max_epochs = 701
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_begin=300, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[4000],
        gamma=0.1)
]
auto_scale_lr = dict(base_batch_size=32)
launcher = 'none'
