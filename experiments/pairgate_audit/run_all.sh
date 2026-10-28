#!/bin/sh
set -eu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python src/experiment.py --model RF128
python src/experiment.py --model MLP64
python src/host_inference_benchmark.py --model RF128
python src/host_inference_benchmark.py --model MLP64
python src/random_gate_controls.py
python src/statistical_audit.py
python src/crypto_cost.py
python -m pytest -q tests
