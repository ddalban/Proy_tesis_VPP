FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

# Dependencias del sistema
RUN apt update && apt install -y \
    git \
    nano \
    curl \
    libgl1 \
    libglib2.0-0

WORKDIR /workspace

# Instalar librerías Python
RUN pip install --no-cache-dir \
    jupyterlab \
    matplotlib \
    numpy \
    opencv-python \
    pillow \
    pandas \
    scikit-image \
    timm \
    torchvision

# Clonar Depth Anything V2
RUN git clone https://github.com/DepthAnything/Depth-Anything-V2.git

EXPOSE 8888

CMD ["bash"]

