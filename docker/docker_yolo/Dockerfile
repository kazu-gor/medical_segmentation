FROM nvcr.io/nvidia/pytorch:20.12-py3

MAINTAINER Soh Yoshida <sohy@kansai-u.ac.jp>

ENV https_proxy http://proxy.itc.kansai-u.ac.jp:8080
ENV http_proxy http://proxy.itc.kansai-u.ac.jp:8080

# Specify number of CPUs can be used while building Tensorflow and OpenCV.
ARG NUM_CPUS_FOR_BUILD=4

#RUN echo -e 'Acquire::http::proxy "http://proxy.itc.kansai-u.ac.jp:8080/";\nAcquire::https::proxy "http://proxy.itc.kansai-u.ac.jp:8080/";\nAcquire::ftp::proxy "http://proxy.itc.kansai-u.ac.jp:8080/";\nAcquire::socks::proxy "socks://proxy.itc.kansai-u.ac.jp:8080/";' >> /etc/apt/apt.conf

#RUN echo -e 'http_proxy="http://proxy.itc.kansai-u.ac.jp:8080/"\nhttps_proxy="http://proxy.itc.kansai-u.ac.jp:8080/"\nftp_proxy="http://proxy.itc.kansai-u.ac.jp:8080/"\nsocks_proxy="http://proxy.itc.kansai-u.ac.jp:8080/"' >> /etc/environment

#RUN echo -e 'https_proxy=http://proxy.itc.kansai-u.ac.jp:8080/\nhttp_proxy=http://proxy.itc.kansai-u.ac.jp:8080/\nftp_proxy=http://proxy.itc.kansai-u.ac.jp:8080/' >>/etc/wgetrc

#RUN sed -i.bak -e "s%http://archive.ubuntu.com/ubuntu/%http://ftp.jaist.ac.jp/pub/Linux/ubuntu/%g" /etc/apt/sources.list

RUN apt-get update -y
RUN apt-get install -y libgl1-mesa-dev

RUN apt-get -y clean all
RUN rm -rf /var/lib/apt/lists/*

# ultralytics
ADD https://ultralytics.com/assets/Arial.ttf https://ultralytics.com/assets/Arial.Unicode.ttf /root/.config/Ultralytics/
WORKDIR /usr/src/ultralytics
RUN git clone https://github.com/ultralytics/ultralytics -b main /usr/src/ultralytics
ADD https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt /usr/src/ultralytics/

# Run exports to AutoInstall packages
RUN yolo export model=tmp/yolov8n.pt format=edgetpu imgsz=32
RUN yolo export model=tmp/yolov8n.pt format=ncnn imgsz=32
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

# Install some useful and machine/deep-learning-related packages for Python3.
RUN pip install --upgrade pip wheel
RUN pip install --upgrade setuptools
RUN pip install \
    opencv-python==4.6.0.66 \
	opencv-python-headless==4.6.0.66 \
	albumentations==1.0.3 \
	scipy==1.6.2 \
	thop \
	pytorch_ranger \
        adabelief-pytorch \
	einops \
	timm==0.3.2 \
	ml_collections \
	tensorboardX \
	ptflops \
	mmcv==1.3.0 \
	cvxpy
        
#
ENV USER student 
ENV HOME /home/${USER}
ENV SHELL /bin/bash
ENV DISPLAY :10
#
RUN groupadd -g 1002 student
RUN useradd -g 1002 -u 1002 -m -s /bin/bash ${USER}
#
RUN gpasswd -a ${USER} sudo
#
RUN echo "${USER}:student" | chpasswd
#
RUN echo 'Defaults visiblepw'             >> /etc/sudoers
RUN echo '${USER} ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
#
USER student

