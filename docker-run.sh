#!/bin/bash

docker run -d \
  --name autolab \
  --restart unless-stopped \
  --gpus all \
  --network host \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$(pwd)":/workspace \
  -w /workspace \
  -e LLAMA_BASE_URL="http://localhost:8080" \
  autolab \
  tail -f /dev/null