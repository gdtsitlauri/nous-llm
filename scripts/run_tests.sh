#!/bin/bash
set -e
cd "$(dirname "$0")/.."
echo "Running NOUS unit tests..."
python -m pytest tests/ -v --tb=short 2>&1
echo "Tests complete."
