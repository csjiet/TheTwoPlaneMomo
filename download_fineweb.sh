#!/bin/bash
set -euo pipefail

echo "Submitting FineWeb tokenization job..."
sbatch tokenize_fineweb.sbatch
