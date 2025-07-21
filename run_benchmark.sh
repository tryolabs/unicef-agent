#!/bin/bash

# Remove existing session ID to create a fresh one for this benchmark run
rm benchmark/results/.session_id

# Add current directory to PYTHONPATH to ensure tests module can be found
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the benchmark tests with the specified number of parallel workers
uv run pytest benchmark/run_benchmark.py