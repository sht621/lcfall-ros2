# =============================================================
# 1. ベースイメージ (NVIDIA CUDA 12.1)
# =============================================================
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# 環境変数の設定
# DEBIAN_FRONTEND: インストール中の対話入力を無効化
# PYTHONUNBUFFERED: Pythonのログをリアルタイムで表示（ROS 2のデバッグに必須）
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# =============================================================
# 2. システム依存関係 & ROS 2 Humble (Desktop版)
# =============================================================
RUN apt-get update && apt-get install -y \
    curl gnupg2 lsb-release git cmake build-essential \
    libgl1-mesa-glx libglib2.0-0 \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt-get update && apt-get install -y \
    ros-humble-desktop \
    ros-humble-realsense2-camera \
    python3-colcon-common-extensions python3-pip \
    && rm -rf /var/lib/apt/lists/*

# =============================================================
# 3. Livox-SDK2 (ソースビルド)
# =============================================================
WORKDIR /
RUN git clone --depth 1 https://github.com/Livox-SDK/Livox-SDK2.git && \
    cd Livox-SDK2 && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && make install && \
    cd / && rm -rf /Livox-SDK2

# =============================================================
# 4. Python ML ライブラリ (PyTorch & OpenMMLab)
# =============================================================
RUN pip3 install --upgrade pip
RUN pip3 install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install openmim && \
    mim install mmengine==0.10.7 "mmcv==2.1.0" && \
    pip3 install mmpose==1.3.1 mmdet==3.3.0 "mmaction>=1.0.0"

# =============================================================
# 5. ワークスペース設定
# =============================================================
WORKDIR /root/ros2_ws
ENV ROS_DISTRO=humble
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

CMD ["/bin/bash"]