FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/scripts

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY finetune.py .

CMD ["/bin/bash"]