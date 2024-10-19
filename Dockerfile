FROM ubuntu:24.04

RUN apt-get update && apt-get install -y cmake build-essential python3 python3-pip && apt-get autoclean

RUN pip install --break-system-packages ytsaurus-client
RUN CMAKE_ARGS="-DGGML_RPC=on" pip install --break-system-packages llama-cpp-python
