ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"
ARG MM="2.8.0"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
RUN pip install mmdet

RUN conda clean --all
ENV FORCE_CUDA="1"

RUN git clone https://github.com/open-mmlab/mmtracking.git /mmtracking
WORKDIR /mmtracking
RUN pip install -r requirements/build.txt
RUN pip install -v -e .
