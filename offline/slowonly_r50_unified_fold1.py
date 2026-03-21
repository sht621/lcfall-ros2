# =============================================================================
# 統一最適化設定 カメラモデル - Fold 1
# =============================================================================
# 全手法統一設定:
# - Optimizer: AdamW
# - Learning Rate: 1e-4
# - Weight Decay: 1e-4
# - Betas: (0.9, 0.999)
# - Class Weight: [5.0, 1.0]
#
# Fold 1: Test=P01, Val=P02, Train=UPFall全+LCFall(P03,P04,P05)
# =============================================================================

_base_ = '/LCFall/external_libs/mmaction2/configs/_base_/default_runtime.py'

# カスタムモジュールのインポート
custom_imports = dict(
    imports=[
        'src.camera.fair_comparison.lowlight_dataset',
        'src.camera.fair_comparison.f1_evaluator',
    ],
    allow_failed_imports=False)

# パス設定
PROJECT_ROOT = '/LCFall'
FOLD = 1

# ログ設定
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=20)

# 環境設定
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

# =============================================================================
# モデル設定
# =============================================================================
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowOnly',
        depth=50,
        pretrained=None,
        in_channels=17,
        base_channels=32,
        num_stages=3,
        out_indices=(2, ),
        stage_blocks=(3, 4, 6),
        conv1_stride_s=1,
        pool1_stride_s=1,
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 2),
        dilations=(1, 1, 1)),
    cls_head=dict(
        type='I3DHead',
        in_channels=512,
        num_classes=2,
        spatial_type='avg',
        dropout_ratio=0.5,
        average_clips='prob',
        loss_cls=dict(
            type='CrossEntropyLoss',
            class_weight=[5.0, 1.0]
        )))

# =============================================================================
# データセット設定
# =============================================================================
dataset_type = 'PoseDataset'

# 既存のアノテーションファイルを使用
ann_file_train = f'/LCFall/src/camera/fair_comparison/annotations/lcfall_fair_fold{FOLD}_train.pkl'
ann_file_val = f'/LCFall/src/camera/fair_comparison/annotations/lcfall_fair_fold{FOLD}_val.pkl'
ann_file_test = f'/LCFall/src/camera/fair_comparison/annotations/lcfall_fair_fold{FOLD}_test.pkl'

left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
box_thr = 0.5

# パイプライン設定
train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(
        type='GeneratePoseTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(64, 64), keep_ratio=False),
    dict(
        type='SafeGeneratePoseTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False,
        handle_all_zeros=True),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='PackActionInputs')
]

test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48, num_clips=10, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(64, 64), keep_ratio=False),
    dict(
        type='SafeGeneratePoseTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False,
        double=True,
        left_kp=left_kp,
        right_kp=right_kp,
        handle_all_zeros=True),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='PackActionInputs')
]

# =============================================================================
# データローダー設定
# =============================================================================
train_dataloader = dict(
    batch_size=16,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='FilteredPoseDataset',
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        min_valid_frames=1,
        valid_ratio=None,
        filter_mode='train',
        test_mode=False,
        box_thr=box_thr))

val_dataloader = dict(
    batch_size=16,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        box_thr=box_thr,
        test_mode=True))

test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        pipeline=test_pipeline,
        box_thr=box_thr,
        test_mode=True))

# =============================================================================
# 評価設定
# =============================================================================
val_evaluator = dict(
    type='F1Metric',
    pos_label=0
)

test_evaluator = dict(
    type='F1MetricWithJSONDump',
    pos_label=0,
    out_file_path=f'/LCFall/src/unified_comparison/predictions/preds_camera_fold{FOLD}.json',
    ann_file_path=ann_file_test
)

# =============================================================================
# トレーニング設定
# =============================================================================
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=100,
    val_begin=1,
    val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# =============================================================================
# 統一最適化設定: AdamW, LR=1e-4, WD=1e-4
# =============================================================================
param_scheduler = []  # スケジューラーなし（学習率固定）

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.0001,
        betas=(0.9, 0.999)
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2))  # 全モデル統一: max_norm=1.0

# =============================================================================
# フック設定
# =============================================================================
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=100,
        max_keep_ckpts=1,
        save_best='f1/f1_score',
        rule='greater'
    ),
    logger=dict(
        type='LoggerHook',
        interval=75,
        ignore_last=False),
    early_stopping=dict(
        type='EarlyStoppingHook',
        monitor='f1/f1_score',
        patience=15,
        rule='greater',
        min_delta=0.001))

# =============================================================================
# その他の設定
# =============================================================================
load_from = '/LCFall/weights/camera/checkpoints/slowonly_r50_8xb32-u48-240e_k400-keypoint_20230731-7f498b55.pth'
work_dir = f'/LCFall/src/unified_comparison/camera/checkpoints/fold{FOLD}'
auto_scale_lr = dict(enable=False, base_batch_size=256)

# シード固定（全モデル統一: seed=42）
# 注意: 3D CNNの一部操作（max_pool3d等）は決定論的実装がないため、
# deterministic=Falseに設定。シード固定のみで再現性を確保。
randomness = dict(seed=42, deterministic=False)

