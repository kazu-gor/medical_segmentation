FROM nvcr.io/nvidia/pytorch:20.12-py3

MAINTAINER Soh Yoshida <sohy@kansai-u.ac.jp>

ENV https_proxy http://proxy.itc.kansai-u.ac.jp:8080
ENV http_proxy http://proxy.itc.kansai-u.ac.jp:8080

# Specify number of CPUs can be used while building Tensorflow and OpenCV.
ARG NUM_CPUS_FOR_BUILD=4

RUN pip install --no-cache nvidia-tensorrt --index-url https://pypi.ngc.nvidia.com
# Downloads to user config dir
# ADD https://ultralytics.com/assets/Arial.ttf https://ultralytics.com/assets/Arial.Unicode.ttf /root/.config/Ultralytics/

# Install linux packages
# g++ required to build 'tflite_support' and 'lap' packages, libusb-1.0-0 required for 'tflite_support' package
RUN apt update \
    && apt install --no-install-recommends -y gcc git zip curl htop libgl1 libglib2.0-0 libpython3-dev gnupg g++ libusb-1.0-0

# Security updates
# https://security.snyk.io/vuln/SNYK-UBUNTU1804-OPENSSL-3314796
RUN apt upgrade --no-install-recommends -y openssl tar

# Create working directory
# WORKDIR /usr/src/ultralytics

# # Copy contents
# # COPY . /usr/src/ultralytics  # git permission issues inside container
# RUN git clone https://github.com/ultralytics/ultralytics -b main /usr/src/ultralytics
# ADD https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt /usr/src/ultralytics/

# Install pip packages
RUN python3 -m pip install --upgrade pip wheel
RUN pip install --no-cache -e ".[export]" --ignore-installed albumentations pycocotools pytest-cov ultralytics
RUN pip install --ignore-installed ultralytics
RUN pip install opencv-python==4.8.0.76
RUN python3 -m pip install psutil==5.9.1

# Run exports to AutoInstall packages
# RUN yolo export model=tmp/yolov8n.pt format=edgetpu imgsz=32
# RUN yolo export model=tmp/yolov8n.pt imgsz=32
RUN yolo detect train model=yolov8n.pt imgsz=352 epochs=1 single_cls
# Requires <= Python 3.10, bug with paddlepaddle==2.5.0 https://github.com/PaddlePaddle/X2Paddle/issues/991
RUN pip install --no-cache paddlepaddle==2.4.2 x2paddle
# Fix error: `np.bool` was a deprecated alias for the builtin `bool` segmentation error in Tests
RUN pip install --no-cache numpy==1.23.5
# Remove exported models
RUN rm -rf tmp

# Set environment variables
ENV OMP_NUM_THREADS=1
# Avoid DDP error "MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library" https://github.com/pytorch/pytorch/issues/37377
ENV MKL_THREADING_LAYER=GNU
        
#
ENV USER student 
ENV HOME /home/${USER}
ENV SHELL /bin/bash
ENV DISPLAY :10
#
RUN groupadd -g 1002 student
RUN useradd -g 1002 -u 1002 -m -s /bin/bash ${USER}
#
# RUN chmod -R 777 /usr/src/ultralytics
# RUN chown -R ${USER}:${USER} /usr/src/ultralytics
#
RUN gpasswd -a ${USER} sudo
#
RUN echo "${USER}:student" | chpasswd
#
RUN echo 'Defaults visiblepw'             >> /etc/sudoers
RUN echo '${USER} ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
#
USER student

