FROM nvcr.io/nvidia/pytorch:23.11-py3

# RUN pip uninstall -y flash-attn

# RUN pip install flash-attn --no-build-isolation

WORKDIR /workspace

RUN git clone https://github.com/timdettmers/bitsandbytes.git

WORKDIR /workspace/bitsandbytes

RUN CUDA_VERSION=123 make cuda12x && \
    python setup.py install

WORKDIR /workspace

COPY requirements.txt .

RUN python -m pip install -r requirements.txt