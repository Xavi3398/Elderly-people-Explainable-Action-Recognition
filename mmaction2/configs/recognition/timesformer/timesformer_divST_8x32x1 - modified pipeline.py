_base_ = ['../../_base_/default_runtime.py']

models = dict(timesformer_divST = 'timesformer_divST_8x32x1_15e_kinetics400_rgb-3f8e5d03.pth',
              timesformer_spaceOnly = 'timesformer_spaceOnly_8x32x1_15e_kinetics400_rgb-0cf829cd.pth')
configs = dict(timesformer_divST = '/configs/recognition/timesformer/timesformer_divST_8x32x1 - modified pipeline.py',
               timesformer_spaceOnly = '/configs/recognition/timesformer/timesformer_spaceOnly_8x32x1 - modified pipeline.py')
mmaction_dir = 'E:/OneDrive - Universitat de les Illes Balears/2021-2022/TFM/mmaction2'
dataset_dir = 'F:/ETRI-Activity3D/RGB Videos/EtriActivity3D'
dataset_name = 'EtriActivity3D'
pre_trained_model = models['timesformer_divST']
config_file = configs['timesformer_divST']
logs_dir = 'E:/OneDrive - Universitat de les Illes Balears/2021-2022/TFM/training_logs'
n_classes = 55

# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='TimeSformer',
        pretrained=  # noqa: E251
        'https://download.openmmlab.com/mmaction/recognition/timesformer/vit_base_patch16_224.pth',  # noqa: E501
        num_frames=8,
        img_size=224,
        patch_size=16,
        embed_dims=768,
        in_channels=3,
        dropout_ratio=0.,
        transformer_layers=None,
        attention_type='divided_space_time',
        norm_cfg=dict(type='LN', eps=1e-6)),
    cls_head=dict(type='TimeSformerHead', num_classes=n_classes, in_channels=768),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))

# dataset settings
dataset_type = 'VideoDataset'
data_root = dataset_dir + '/train/'
data_root_val = dataset_dir + '/val/'
data_root_test = dataset_dir + '/test/'
ann_file_train = dataset_dir + '/' + dataset_name + '_train_video.txt'
ann_file_val = dataset_dir + '/' + dataset_name + '_val_video.txt'
ann_file_test = dataset_dir + '/' + dataset_name + '_test_video.txt'

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_bgr=False)

train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=8, frame_interval=32, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=224),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=32,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=32,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=2,
    workers_per_gpu=5,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_test,
        pipeline=test_pipeline))

evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.005/8,
    momentum=0.9,
    paramwise_cfg=dict(
        custom_keys={
            '.backbone.cls_token': dict(decay_mult=0.0),
            '.backbone.pos_embed': dict(decay_mult=0.0),
            '.backbone.time_embed': dict(decay_mult=0.0)
        }),
    weight_decay=1e-4,
    nesterov=True)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

# learning policy
lr_config = dict(policy='step', step=[5, 10])
total_epochs = 15

# runtime settings
checkpoint_config = dict(interval=1)
work_dir = logs_dir

# Other
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
load_from = mmaction_dir + '/checkpoints/' + pre_trained_model
gpu_ids = range(1)