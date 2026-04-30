#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# --- ERROR HANDLING ---
handle_error() {
    echo ""
    echo "==================================================="
    echo "❌ ERROR: Pipeline crashed at line $1"
    echo "==================================================="
    echo ""
}

trap 'handle_error $LINENO' ERR

# ----------------------

# 1. Activate the Python/VMTK Virtual Environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate VmtkMonai

# 2. Setup inputs and variables
INPUT_FILE="$1"
GPU_ID="${2:-0}"  # This means: use $2 if provided, otherwise default to "0"

# Safety check: Ensure an input file was actually provided
if [ -z "$INPUT_FILE" ]; then
    echo "❌ ERROR: No input file provided."
    echo "Usage: ./script.sh path/to/your/image.nii.gz [gpu_id]"
    exit 1
fi

FILENAME=$(basename "$INPUT_FILE")
BASENAME="${FILENAME%%.nii.gz}"
OUT_DIR="output"
SRC_DIR="src"

# Define the paths for the script and model
PIPELINE_SCRIPT="${SRC_DIR}/main_pipeline.py"
MODEL_PATH="${SRC_DIR}/best-model-epoch=1339-val_dice=0.9170.ckpt"

# Ensure output directory exists
mkdir -p "$OUT_DIR"

echo "Starting unified automated pipeline for ${BASENAME}..."

# 3. Run the full Unified Python Pipeline
# (The python script now handles the generation of all .seg.nii.gz, .vtp, and .csv files)
python "$PIPELINE_SCRIPT" \
    --model_path "$MODEL_PATH" \
    --input_image "$INPUT_FILE" \
    --output_dir "$OUT_DIR" \
    --gpu "$GPU_ID" \
    --radius 4.0 \
    --distance_back 3.0

echo "✅ Pipeline complete! All files successfully saved to ${OUT_DIR}/"