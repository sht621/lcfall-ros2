# =============================================================
# 1. ベースイメージ
# =============================================================
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV ROS_DISTRO=humble

# Blackwell/RTX 5070 Ti 対応のため、PTX を含めて前方互換を確保
ENV TORCH_CUDA_ARCH_LIST="9.0+PTX"
ENV FORCE_CUDA=1
ENV MMCV_WITH_OPS=1
ENV MAX_JOBS=8

# PyTorch 2.6+ の weights_only 既定値変更への保険
ENV TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# =============================================================
# 2. システム依存関係 & ROS 2 Humble
# =============================================================
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    gnupg2 \
    lsb-release \
    git \
    cmake \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-pip \
    && add-apt-repository universe \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2.list \
    && apt-get update \
    && apt-get install -y \
    python3-colcon-common-extensions \
    ros-humble-desktop \
    ros-humble-realsense2-camera \
    ros-humble-cv-bridge \
    ros-humble-message-filters \
    && rm -rf /var/lib/apt/lists/*

# =============================================================
# 3. Livox-SDK2
# =============================================================
WORKDIR /
RUN git clone --depth 1 https://github.com/Livox-SDK/Livox-SDK2.git && \
    cd Livox-SDK2 && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j"$(nproc)" && \
    make install && \
    cd / && rm -rf /Livox-SDK2

# =============================================================
# 4. livox_ros_driver2
# =============================================================
WORKDIR /root/ros2_ws
RUN mkdir -p src && \
    git clone --depth 1 https://github.com/Livox-SDK/livox_ros_driver2.git src/livox_ros_driver2 && \
    cp src/livox_ros_driver2/package_ROS2.xml src/livox_ros_driver2/package.xml && \
    . /opt/ros/humble/setup.sh && \
    colcon build --packages-up-to livox_ros_driver2 --cmake-args -DROS_EDITION="ROS2"  && \
    echo "source /root/ros2_ws/install/setup.bash" >> /root/.bashrc && \
    rm -rf build log

# =============================================================
# 5. system Python 競合パッケージを除去
# =============================================================
RUN apt-get update && \
    apt-get purge -y \
    python3-sympy \
    python3-mpmath \
    python3-scipy \
    python3-pythran || true && \
    rm -rf /var/lib/apt/lists/*

# =============================================================
# 6. Python ビルド基盤
# =============================================================
RUN pip3 install --upgrade pip && \
    pip3 uninstall -y opencv-python opencv-python-headless opencv-contrib-python || true && \
    pip3 install --no-cache-dir \
    "setuptools==60.2.0" \
    wheel \
    ninja \
    psutil \
    cython \
    "numpy<2"

# =============================================================
# 7. PyTorch
# =============================================================
RUN pip3 install --pre torch torchvision torchaudio "numpy<2" \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# =============================================================
# 8. OpenMMLab
# =============================================================
# mmcv インストール直前で setuptools / numpy を再固定
RUN pip3 install --no-cache-dir --force-reinstall \
    "setuptools==60.2.0" \
    "numpy<2" \
    wheel \
    ninja \
    psutil \
    cython && \
    pip3 install --no-cache-dir mmengine==0.10.7 && \
    pip3 install --no-build-isolation --no-cache-dir "mmcv==2.1.0" && \
    pip3 install chumpy --no-build-isolation && \
    pip3 install mmpose==1.3.1 mmdet==3.3.0 mmaction2==1.2.0 && \
    mkdir -p /usr/local/lib/python3.10/dist-packages/mmaction/models/localizers/drn && \
    echo "class DRN: pass" > /usr/local/lib/python3.10/dist-packages/mmaction/models/localizers/drn/__init__.py && \
    echo "class DRN: pass" > /usr/local/lib/python3.10/dist-packages/mmaction/models/localizers/drn/drn.py

# =============================================================
# 9. Python 追加依存
# =============================================================
RUN pip3 install --no-cache-dir \
    "numpy<2" \
    "opencv-python-headless<4.10.0.84" \
    "opencv-python<4.10.0.84" \
    "opencv-contrib-python<4.10.0.84"

# =============================================================
# 10. ワークスペース設定
# =============================================================
WORKDIR /root/ros2_ws
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

CMD ["/bin/bash"]
