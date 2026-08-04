# =============================================================
# RTX 3050 Laptop GPU host environment
# Ubuntu 22.04 / ROS 2 Humble / Python 3.10 / CUDA 12.1
# =============================================================
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV ROS_DISTRO=humble

# GeForce RTX 3050 Laptop GPU (Ampere, compute capability 8.6)
ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV FORCE_CUDA=1
ENV MMCV_WITH_OPS=1
ENV MAX_JOBS=8

# =============================================================
# System dependencies and ROS 2 Humble
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
    fonts-noto-cjk \
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
# Livox-SDK2
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
# livox_ros_driver2
# =============================================================
WORKDIR /root/ros2_ws
RUN mkdir -p src && \
    git clone --depth 1 https://github.com/Livox-SDK/livox_ros_driver2.git src/livox_ros_driver2 && \
    cp src/livox_ros_driver2/package_ROS2.xml src/livox_ros_driver2/package.xml && \
    . /opt/ros/humble/setup.sh && \
    colcon build --packages-up-to livox_ros_driver2 --cmake-args -DROS_EDITION="ROS2" -DDISTRO_ROS="humble" && \
    echo "source /root/ros2_ws/install/setup.bash" >> /root/.bashrc && \
    rm -rf build log

# =============================================================
# Remove conflicting Ubuntu Python packages
# =============================================================
RUN apt-get update && \
    apt-get purge -y \
    python3-sympy \
    python3-mpmath \
    python3-scipy \
    python3-pythran || true && \
    rm -rf /var/lib/apt/lists/*

# =============================================================
# Python build prerequisites
# =============================================================
RUN pip3 install --upgrade pip && \
    pip3 uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless || true && \
    pip3 install --no-cache-dir \
    "setuptools==60.2.0" \
    wheel \
    ninja \
    psutil \
    cython \
    "numpy==1.26.4"

# =============================================================
# Stable PyTorch CUDA 12.1 wheels
# =============================================================
RUN pip3 install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# =============================================================
# OpenMMLab versions used by the existing demo
# =============================================================
RUN pip3 install --no-cache-dir --force-reinstall \
    "setuptools==60.2.0" \
    "numpy==1.26.4" \
    wheel \
    ninja \
    psutil \
    cython && \
    pip3 install --no-cache-dir mmengine==0.10.7 && \
    pip3 install --no-build-isolation --no-cache-dir mmcv==2.1.0 && \
    pip3 install --no-build-isolation --no-cache-dir chumpy && \
    pip3 install --no-cache-dir \
    mmpose==1.3.1 \
    mmdet==3.3.0 \
    mmaction2==1.2.0 \
    mmpretrain==1.2.0 && \
    MMACTION_SITE="$(python3 -c 'import site; print(site.getsitepackages()[0])')" && \
    mkdir -p "${MMACTION_SITE}/mmaction/models/localizers/drn" && \
    echo "class DRN: pass" > "${MMACTION_SITE}/mmaction/models/localizers/drn/__init__.py" && \
    echo "class DRN: pass" > "${MMACTION_SITE}/mmaction/models/localizers/drn/drn.py"

# Keep one OpenCV wheel. The non-headless build is required by the existing
# X11 visualization node and ROI tuner.
RUN pip3 install --no-cache-dir \
    "numpy==1.26.4" \
    "Pillow==10.2.0" \
    "opencv-contrib-python==4.9.0.80"

# Keep the optional ROS middleware available without changing the Fast DDS
# production selection configured by Compose.
RUN --network=host \
    apt-get update && apt-get install -y \
    ros-humble-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

# =============================================================
# Workspace setup
# =============================================================
WORKDIR /root/ros2_ws
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

CMD ["/bin/bash"]
