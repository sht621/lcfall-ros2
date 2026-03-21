# =============================================================================
# PoseC3D 推論用設定（最小構成）
# =============================================================================
# 学習パイプライン・データローダー・評価設定等は推論時不要なため省略。
# モデル定義のみを含む。
#
# 使用方法:
#   cfg = Config.fromfile('slowonly_r50_inference.py')
#   model = MODELS.build(cfg.model)
# =============================================================================

default_scope = 'mmaction'

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
        average_clips='prob'))
