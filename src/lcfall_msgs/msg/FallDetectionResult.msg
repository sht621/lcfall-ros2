# 転倒検知推論結果
# inference_node → alert_node / visualization_node 間のインタフェース

std_msgs/Header header

# 0: non-falling, 1: falling
uint8 prediction

# 推論の確信度 [0.0, 1.0]
float32 confidence
