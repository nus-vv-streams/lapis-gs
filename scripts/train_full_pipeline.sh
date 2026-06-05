#!/bin/bash
#SBATCH --job-name=lapis-topdown
#SBATCH --partition=gpu-long
#SBATCH --time=2-00:00:00
#SBATCH --mem-per-cpu=64G
#SBATCH --gres=gpu:1

# Top-down LapisGS experiment driver.
# Invokes train_full_pipeline_topdown.py per (dataset, scene).
#
# Every variable below can be overridden via the environment, e.g.:
#   DATASET=nerf_synthetic SCENES="lego chair" LAYER_SIZE=45000 \
#       sbatch scripts/train_full_pipeline.sh
# When unset, the defaults baked into the script are used.

# ---- paths / output naming ----
MODEL_BASE="${MODEL_BASE:-/home/e/e0686126/gs/model}"
DATASET_BASE="${DATASET_BASE:-/home/e/e0686126/gs/source}"
METHOD="${METHOD:-lapis_topdown}"

# ---- training hyperparameters ----
LAMBDA_DSSIM="${LAMBDA_DSSIM:-0.2}"
N_LAYERS="${N_LAYERS:-4}"

# --layer_size:
#   empty / unset  -> equal-split mode (no prune, full model partitioned into N near-equal bands)
#   positive int   -> L3GS mode (prune to N*LAYER_SIZE first, then split into N bands of d)
LAYER_SIZE="${LAYER_SIZE:-}"

# DYNAMIC_OPACITY:
#   "yes" (default) -> LapisGS regime: ancestors frozen except opacity (--dynamic_opacity)
#   "no"            -> L3GS regime:    ancestors fully frozen (--no_dynamic_opacity)
DYNAMIC_OPACITY="${DYNAMIC_OPACITY:-yes}"

# ---- which dataset + which scenes to run ----
# One dataset per invocation. To run multiple datasets, submit multiple jobs.
# SCENES is a space-separated list (gets word-split into an array).
DATASET="${DATASET:-db}"
SCENES="${SCENES:-playroom drjohnson}"

# ---- pipeline call ----
# Per-dataset extras.
extra_args=()
if [[ "${DATASET}" == "nerf_synthetic" ]]; then
    extra_args+=("-w")   # white background for NeRF synthetic
fi
if [[ -n "${LAYER_SIZE}" ]]; then
    extra_args+=("--layer_size" "${LAYER_SIZE}")
fi
if [[ "${DYNAMIC_OPACITY}" == "no" ]]; then
    extra_args+=("--no_dynamic_opacity")
fi

for scene in ${SCENES}; do
    echo "=== dataset=${DATASET}, scene=${scene} ==="

    srun python -u ./train_full_pipeline_topdown.py \
        --model_base "${MODEL_BASE}" \
        --dataset_base "${DATASET_BASE}" \
        --dataset_name "${DATASET}" \
        --scene "${scene}" \
        --method "${METHOD}" \
        --lambda_dssim "${LAMBDA_DSSIM}" \
        --n_layers "${N_LAYERS}" \
        "${extra_args[@]}"
done
