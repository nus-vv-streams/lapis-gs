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

# EVAL:
#   "yes" (default) -> use the standard train/test split (--eval), test cameras
#                      go to test/ and metrics are computed against them.
#   "no"            -> no eval split (--no_eval); ALL cameras become train, the
#                      test set will be empty and render/metrics will be no-ops.
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
if [[ -n "${LAYER_SIZE}" ]]; then
    extra_args+=("--layer_size" "${LAYER_SIZE}")
fi
if [[ "${DYNAMIC_OPACITY}" == "no" ]]; then
    extra_args+=("--no_dynamic_opacity")
fi
if [[ "${EVAL}" == "no" ]]; then
    extra_args+=("--no_eval")
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

            if [[ "${CLEANUP}" == "yes" && -f "${layer_dir}/results.json" ]]; then
                echo "--- cleanup ${layer_dir} (PLY + renders) ---"
                srun python -u ./scripts/record_model_size.py -m "${layer_dir}" || true
                find "${layer_dir}/point_cloud" -name "point_cloud.ply" -delete 2>/dev/null || true
                rm -rf "${layer_dir}"/train/ours_* "${layer_dir}"/test/ours_* 2>/dev/null || true
            fi
        done
    fi

    # ---- intermediate cleanup: full pretrain + pruned model + buckets ----
    # These are the scaffolding the top-down pipeline produces on the way to the
    # L*_res* LODs. Once the finest LOD has results.json (pipeline finished
    # end-to-end for this scene), they're no longer needed.
    method_dir="${MODEL_BASE}/${DATASET}/${scene}/${METHOD}"
    finest_lod_results="${method_dir}/L$((N_LAYERS - 1))_res1/results.json"
    if [[ "${CLEANUP}" == "yes" && -f "${finest_lod_results}" ]]; then
        full_pretrain="${method_dir}/${scene}_full_res1"
        if [[ -d "${full_pretrain}" ]]; then
            echo "--- cleanup ${full_pretrain} (PLY) ---"
            find "${full_pretrain}/point_cloud" -name "point_cloud.ply" -delete 2>/dev/null || true
        fi

        # L3GS mode produces {scene}_pruned_<N>; equal-split mode skips it.
        for pruned_dir in "${method_dir}/${scene}_pruned_"*; do
            [[ -d "${pruned_dir}" ]] || continue
            echo "--- cleanup ${pruned_dir} (PLY) ---"
            find "${pruned_dir}/point_cloud" -name "point_cloud.ply" -delete 2>/dev/null || true
        done

        buckets_dir="${method_dir}/buckets"
        if [[ -d "${buckets_dir}" ]]; then
            echo "--- cleanup ${buckets_dir} (whole dir) ---"
            rm -rf "${buckets_dir}"
        fi
    fi
done
