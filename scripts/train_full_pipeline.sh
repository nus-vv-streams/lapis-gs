#!/bin/bash
#SBATCH --job-name=gs
#SBATCH --partition=gpu-long
#SBATCH --time=2-00:00:00
#SBATCH --mem-per-cpu=256G
#SBATCH --gres=gpu:1

MODEL_BASE="/home/e/e0686126/gs/model"
DATASET_BASE="/home/e/e0686126/gs/source"
METHOD="lapis"
LAMBDA_DSSIM="0.2"

datasets=("nerf_synthetic") # ("db" "tandt" "nerf_synthetic" "360")
scene_groups=
#(
#  "playroom drjohnson"
#  "train truck"
#  "chair drums ficus hotdog materials mic ship lego"
#  "bonsai counter garden kitchen room bicycle flowers stump treehill"
#)

for i in "${!datasets[@]}"; do
    dataset="${datasets[$i]}"

    for scene in ${scene_groups[$i]}; do
      echo "=== dataset=${dataset}, scene=${scene} ==="

      # train_full_pipeline.py already runs train + render + metrics for res8/4/2/1
      srun python -u lapis_clean/train_full_pipeline_topdown.py \
        --model_base "${MODEL_BASE}" \
        --dataset_base "${DATASET_BASE}" \
        --dataset_name "${dataset}" \
        --scene "${scene}" \
        --method "${METHOD}" \
        --lambda_dssim "${LAMBDA_DSSIM}" \ 
        --n_layers 4 \

    done
done