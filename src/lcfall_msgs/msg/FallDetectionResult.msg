# 転倒検知推論結果
# inference_node → alert_node / visualization_node 間のインタフェース

std_msgs/Header header

# false: non-falling, true: falling
bool prediction

# 推論の確信度 [0.0, 1.0]
float32 confidence
