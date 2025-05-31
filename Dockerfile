FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    wget gdb build-essential curl git cmake pkg-config vim \
    python3 python3-pip docker-compose \
    ninja-build pybind11-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install git+https://github.com/Rich-Hall/sentinel1decoder

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && rm requirements.txt

RUN git clone https://github.com/rogersce/cnpy.git && \
    cd cnpy && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make -j$(nproc) && \
    make install && \
    cd /opt && rm -rf cnpy

RUN git clone https://github.com/alandefreitas/matplotplusplus/

RUN pip install torch==2.5.1+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention && \
    git checkout v2.7.4.post1 && \
    pip install packaging && \
    export FLASH_ATTENTION_FORCE_BUILD=1 && \
    pip install .

RUN echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc && \
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/compat:$LD_LIBRARY_PATH' >> ~/.bashrc
