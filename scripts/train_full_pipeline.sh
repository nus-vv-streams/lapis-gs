#!/bin/bash
#SBATCH --job-name=lapis-topdown
#SBATCH --partition=gpu-long
#SBATCH --time=2-00:00:00
#SBATCH --mem-per-cpu=64G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuangshi@comp.nus.edu.sg


# Top-down LapisGS experiment driver.
# Invokes train_full_pipeline_topdown.py per (dataset, scene), then renders and
# computes metrics for each of the N produced layers (L1_res8 .. LN_res1).
#
# Every variable below can be overridden via the environment, e.g.:
#   DATASET=nerf_synthetic SCENES="lego chair" LAYER_SIZE=45000 \
#       sbatch scripts/train_full_pipeline.sh

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
# One dataset per invocation. SCENES is a space-separated list.
DATASET="${DATASET:-db}"
SCENES="${SCENES:-playroom drjohnson}"

# ---- post-training evaluation toggles ----
RUN_TRAIN="${RUN_TRAIN:-yes}"
RUN_RENDER="${RUN_RENDER:-yes}"
RUN_METRICS="${RUN_METRICS:-yes}"

# ---- assemble pipeline extras ----
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
    echo "=========================================="
    echo "=== dataset=${DATASET}, scene=${scene} ==="
    echo "=========================================="

    if [[ "${RUN_TRAIN}" == "yes" ]]; then
        srun python -u ./train_full_pipeline_topdown.py \
            --model_base "${MODEL_BASE}" \
            --dataset_base "${DATASET_BASE}" \
            --dataset_name "${DATASET}" \
            --scene "${scene}" \
            --method "${METHOD}" \
            --lambda_dssim "${LAMBDA_DSSIM}" \
            --n_layers "${N_LAYERS}" \
            "${extra_args[@]}"
    else
        echo "[skip] RUN_TRAIN=no — skipping training"
    fi

    # Render + metrics for each produced layer (0-indexed). Layer k trains at
    # resolution 2^(N - 1 - k): k=0 -> coarsest (res8 for N=4), k=N-1 -> res1.
    if [[ "${RUN_RENDER}" == "yes" || "${RUN_METRICS}" == "yes" ]]; then
        for ((k = 0; k < N_LAYERS; k++)); do
            res=$((2 ** (N_LAYERS - 1 - k)))
            layer_dir="${MODEL_BASE}/${DATASET}/${scene}/${METHOD}/L${k}_res${res}"

            if [[ ! -d "${layer_dir}" ]]; then
                echo "[warn] layer dir not found, skipping eval: ${layer_dir}"
                continue
            fi

            echo "--- eval L${k}_res${res} (${layer_dir}) ---"
            if [[ "${RUN_RENDER}" == "yes" ]]; then
                srun python -u ./render.py -m "${layer_dir}" --skip_train
            fi
            if [[ "${RUN_METRICS}" == "yes" ]]; then
                srun python -u ./metrics.py -m "${layer_dir}"
            fi
        done
    fi
done
