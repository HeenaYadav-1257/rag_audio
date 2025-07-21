#!/bin/bash
export TRANSFORMERS_OFFLINE=1
export SENTENCE_TRANSFORMERS_HOME=./all-mpnet-base-v2

echo "[✓] Starting Gradio interface on http://localhost:7860"
python ui.py

