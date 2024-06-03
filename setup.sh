#!/bin/bash
cd notebooks/experiments/collocation/lm-evaluation-harness
pip install -e .
cd ../../../..
#pip install -r requirements.txt  # Install Python dependencies.
cat requirements.txt | xargs -n 1 -L 1 pip install
pip uninstall apex