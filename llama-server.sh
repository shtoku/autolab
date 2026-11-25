#!/bin/bash

~/src/llama.cpp/build/bin/llama-server \
  --models-preset ~/llm/models.ini \
  --models-max 1 \
  --host 0.0.0.0 \
  --port 8080
