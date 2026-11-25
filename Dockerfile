FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel

WORKDIR /workspace

RUN pip3 install --no-cache-dir --break-system-packages\
  langchain-openai \
  langgraph \
  opencv-python \
  httpx

COPY . /workspace
