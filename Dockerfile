FROM python:3.12-slim

# 国内网络可用构建参数切换 PyPI 镜像加速（海外构建可 --build-arg PIP_INDEX_URL= 覆盖）
ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ENV PIP_INDEX_URL=${PIP_INDEX_URL}

# CPU 版 torch 镜像（先装 CPU 版，避免默认拉数 GB 的 CUDA 版 torch，镜像小、下载快）
ARG TORCH_INDEX_URL=https://mirror.sjtu.edu.cn/pytorch-wheels/cpu
ENV TORCH_INDEX_URL=${TORCH_INDEX_URL}

WORKDIR /app

# 先装 CPU torch：requirements 里 sentence-transformers / FlagEmbedding 依赖 torch，会被此版本满足，不再拉 CUDA 版
RUN pip install --no-cache-dir torch --index-url ${TORCH_INDEX_URL}

# 再装其余依赖，利用 Docker 缓存层
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "main.py", "--mode", "api"]
