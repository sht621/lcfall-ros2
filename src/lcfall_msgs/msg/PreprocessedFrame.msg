# 前処理済み 1 フレームデータ
# sync_preprocess_node → inference_node 間のインタフェース

std_msgs/Header header

# 1人分の 2D skeleton: 17 keypoints × (x_norm, y_norm, score) = 51 要素
# x_norm, y_norm は [0, 1] 正規化座標
float32[] skeleton_2d

# 背景差分後の前景生点群: 256 points × 3 (x, y, z) = 768 要素
# グローバル正規化はまだ適用しない
float32[] pointcloud_frame
