#!/usr/bin/env bash
# Reproduce every result in the README (about 1.5-2 h on a laptop CPU; much faster on a GPU).
set -e
cd "$(dirname "$0")/src"
python prepare_data.py
python run_pipeline.py   # transformer -> probe -> mT5 -> Gemma -> LLaMA -> compare
