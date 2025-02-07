# 使用官方 Ubuntu 基础镜像
FROM ubuntu:22.04

# 更新系统并安装必要工具和库
RUN apt-get update && apt-get install -y \
    build-essential \
    openmpi-bin \
    libopenmpi-dev \
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    gfortran \
    vim \
    && apt-get clean

# 设置工作目录
WORKDIR /workspace

# 默认启动 Bash
CMD ["/bin/bash"]
