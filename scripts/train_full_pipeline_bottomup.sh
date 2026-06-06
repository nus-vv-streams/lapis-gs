#!/bin/bash
#SBATCH --job-name=lapis-bottomup
#SBATCH --partition=gpu-long
#SBATCH --time=2-00:00:00
#SBATCH --mem-per-cpu=64G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuangshi@comp.nus.edu.sg


# Bottom-up LapisGS experiment driver (original method).
# Invokes train_full_pipeline.py per (dataset, scene), then renders and
# computes metrics for each of the 4 trained resolution layers
# ({scene}_res8 .. {scene}_res1).
#
# Every variable below can be overridden via the environment, e.g.:
#   DATASET=nerf_synthetic SCENES="lego chair" DYNAMIC_OPACITY=no \
#       sbatch scripts/train_full_pipeline_bottomup.sh

# ---- paths / output naming ----
MODEL_BASE="${MODEL_BASE:-/home/e/e0686126/gs/model}"
DATASET_BASE="${DATASET_BASE:-/home/e/e0686126/gs/source}"
# METHOD is the OUTPUT folder name (decoupled from the regime, which is derived
# from DYNAMIC_OPACITY below).
METHOD="${METHOD:-bottomup_opacity}"

# ---- training hyperparameters ----
LAMBDA_DSSIM="${LAMBDA_DSSIM:-0.2}"
ITERATIONS="${ITERATIONS:-30000}"

# DYNAMIC_OPACITY (selects the training regime passed to train_full_pipeline.py):
#   "yes" (default) -> LapisGS regime ("lapis"):  ancestors frozen except opacity
#   "no"            -> freeze regime  ("freeze"): ancestors fully frozen
DYNAMIC_OPACITY="${DYNAMIC_OPACITY:-yes}"
if [[ "${DYNAMIC_OPACITY}" == "no" ]]; then
    REGIME="freeze"
else
    REGIME="lapis"
fi

# EVAL:
#   "yes" (default) -> use the standard train/test split (--eval), test cameras
#                      go to test/ and metrics are computed against them.
#   "no"            -> no eval split (--no_eval); render/metrics will be no-ops.
EVAL="${EVAL:-yes}"

# ---- which dataset + which scenes to run ----
# One dataset per invocation. SCENES is a space-separated list.
DATASET="${DATASET:-db}"
SCENES="${SCENES:-playroom drjohnson}"

# ---- post-training evaluation toggles ----
RUN_TRAIN="${RUN_TRAIN:-yes}"
RUN_RENDER="${RUN_RENDER:-yes}"
RUN_METRICS="${RUN_METRICS:-yes}"

# CLEANUP (default "yes"): after metrics are computed for a layer, delete the
# saved PLY point cloud and the rendered images to save disk. A tiny
# model_size.json sidecar is written first so plotting (model size on x-axis)
# still works after cleanup. Only fires when results.json exists for that
# layer, so partial runs don't delete their own inputs.
CLEANUP="${CLEANUP:-yes}"

# ---- assemble pipeline extras ----
extra_args=()
if [[ "${DATASET}" == "nerf_synthetic" ]]; then
    extra_args+=("-w")   # white background for NeRF synthetic
fi
if [[ "${EVAL}" == "no" ]]; then
    extra_args+=("--no_eval")
fi

# Resolutions produced by the bottom-up pipeline (coarse -> fine).
RES_SCALES=(8 4 2 1)

for scene in ${SCENES}; do
    echo "=========================================="
    echo "=== dataset=${DATASET}, scene=${scene} ==="
    echo "=========================================="

    if [[ "${RUN_TRAIN}" == "yes" ]]; then
        srun python -u ./train_full_pipeline.py \
            --model_base "${MODEL_BASE}" \
            --dataset_base "${DATASET_BASE}" \
            --dataset_name "${DATASET}" \
            --scene "${scene}" \
            --method "${METHOD}" \
            --regime "${REGIME}" \
            --lambda_dssim "${LAMBDA_DSSIM}" \
            --iterations "${ITERATIONS}" \
            "${extra_args[@]}"
    else
        echo "[skip] RUN_TRAIN=no — skipping training"
    fi

    # Render + metrics per resolution layer. Bottom-up dirs are named
    # {scene}_res{R}: res8 = coarsest, res1 = finest.
    if [[ "${RUN_RENDER}" == "yes" || "${RUN_METRICS}" == "yes" ]]; then
        for res in "${RES_SCALES[@]}"; do
            layer_dir="${MODEL_BASE}/${DATASET}/${scene}/${METHOD}/${scene}_res${res}"

            if [[ ! -d "${layer_dir}" ]]; then
                echo "[warn] layer dir not found, skipping eval: ${layer_dir}"
                continue
            fi

            echo "--- eval ${scene}_res${res} (${layer_dir}) ---"
            if [[ "${RUN_RENDER}" == "yes" ]]; then
                srun python -u ./render.py -m "${layer_dir}" --skip_train
            fi
            if [[ "${RUN_METRICS}" == "yes" ]]; then
                srun python -u ./metrics.py -m "${layer_dir}"
            fi

            if [[ "${CLEANUP}" == "yes" && -f "${layer_dir}/results.json" ]]; then
                echo "--- cleanup ${layer_dir} (PLY + renders) ---"
                srun python -u ./scripts/record_model_size.py -m "${layer_dir}" || true
                find "${layer_dir}/point_cloud" -name "point_cloud.ply" -delete 2>/dev/null || true
                rm -rf "${layer_dir}"/train/ours_* "${layer_dir}"/test/ours_* 2>/dev/null || true
            fi
        done
    fi
done
