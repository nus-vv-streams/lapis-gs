
<div align="center">
    <h1>
        <img src="/images/title.png" alt="icon" style="height: 1em; vertical-align: middle; margin-right: 0.1em;">
        <strong>LapisGS: </strong>Layered Progressive 3D Gaussian Splatting for Adaptive Streaming
    </h1>
</div>

<div align="center">
    <a href="https://yuang-ian.github.io" target='_blank'>Yuang Shi</a><sup>1</sup>,
    <a href="https://scholar.google.com/citations?user=PbKu-PsAAAAJ&hl=en" target='_blank'>Simone Gasparini</a><sup>2</sup>,
    <a href="https://scholar.google.de/citations?user=H8QDhhAAAAAJ&hl=en" target='_blank'>Géraldine Morin</a><sup>2</sup>,
    <a href="https://www.comp.nus.edu.sg/~ooiwt/" target='_blank'>Wei Tsang Ooi</a><sup>1</sup>,
    <p>
        <sup>1</sup>National University of Singapore,
        <sup>2</sup>IRIT - Université de Toulouse
    </p>
    <p>
    International Conference on 3D Vision (3DV), 2025
    </p>
</div>


<div align="center">
    <a href="http://arxiv.org/abs/2408.14823" target='_blank'>
        <img src="https://img.shields.io/badge/Paper-%F0%9F%93%83-blue">
    </a>
    <a href="https://yuang-ian.github.io/lapisgs/" target='_blank'>
        <img src="https://img.shields.io/badge/Project-%F0%9F%94%97-yellow">
    </a>
</div> <br> <br>



<p align="center">
  <a href="">
    <img src="/images/teaser.png" alt="teaser" width="80%">
  </a>
</p>

<p align="center">
    We introduce <strong><i>LapisGS</i></strong>*, a layered progressive 3DGS, for adaptive streaming and view-adaptive rendering.
</p>

<p align="center">
    <span class="small">
        *<i>Lapis</i> means ”layer” in Malay, the national language of Singapore --- the host of 3DV'25. The logo in the title depicts <a href="https://en.wikipedia.org/wiki/Kue_lapis">kuih lapis</a>, or ”layered cake”, a local delight in Singapore and neighboring countries. The authors are glad to serve kuih lapis at the conference 🍰.
    </span>
</p>
<br>


## News

- We updated the codebase to include a new top-down pipeline inspired by [L3GS](https://github.com/mavens-lab/layered_3d_gaussian_splats), which trains a full-resolution model first and then partitions it into equal-size layers. The original bottom-up pipeline is still available as the `lapis` method, while the new top-down pipeline is available as `lapis_topdown`. 
- 🏆 **2025/09**: **Best Paper Award** at ACM SIGCOMM EMS'25. Based on LapisGS, we built a system called NETSPLAT ([Short Paper](https://dl.acm.org/doi/10.1145/3746441.3748225)) that leverages data plane programmability to provide network assistance for 3DGS streaming. It is WiP, but we won the Best Paper Award at ACM SIGCOMM EMS'25...Again!
- 🏆 **2025/04**: **Best Paper Award** at ACM MMSys'25. We extend LapisGS to dynamic scenes (Dynamic-LapisGS, Check the [Code](https://github.com/nus-vv-streams/dynamic-lapis-gs/)). Based on Dynamic-LapisGS, we built the first ever dynamic 3DGS streaming system named LTS ([Paper](https://doi.org/10.1145/3712676.3714445)), and won the Best Paper Award at ACM MMSys'25!


## Method overview

LapisGS produces a layered 3DGS that can be streamed and decoded progressively. This repository now ships **two ways** to construct that layered structure:

- **Bottom-up** (original, as published at 3DV'25): train the coarsest layer first on a downsampled image, then for each finer resolution freeze the previous layer and train *new* Gaussians on top.
- **Top-down** (new feature added, inspired by L3GS): train a single full-resolution 3DGS model first, score every Gaussian by LightGaussian-style importance, partition the model into N layers, then fine-tune each layer from coarse to fine with the previous layers frozen (except opacity). Optionally prune the full model to a fixed splat budget before partitioning. Refer to [L3GS](https://github.com/mavens-lab/layered_3d_gaussian_splats) for details.

Both pipelines share the same per-layer training primitive ([`train.py`](train.py)) and the same notion of a frozen base + dynamic-opacity ancestor (LapisGS regime) versus fully-frozen ancestor (L3GS "freeze" regime).


## Setup

Our work is built on the codebase of the original 3D Gaussian Splatting method. Please refer to the [original 3D Gaussian Splatting repository](https://github.com/graphdeco-inria/gaussian-splatting) for details about requirements.

The top-down pipeline additionally needs an importance-scoring CUDA rasterizer (a variant of `diff-gaussian-rasterization` that exposes a per-Gaussian opacity-accumulation score, from [LightGaussian](https://github.com/VITA-Group/LightGaussian)). It is vendored inside this repository at [`submodules/compress-diff-gaussian-rasterization`](submodules/compress-diff-gaussian-rasterization) and installed under the package name `dgr_l3gs` so it coexists with the standard rasterizer.

The cleanest install path uses the provided conda environment, which builds all three submodules in one shot:

```bash
conda env create -f environment.yml
conda activate gaussian_splatting
```

If you prefer to manage the env yourself:

```bash
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install submodules/compress-diff-gaussian-rasterization   # required for top-down only
```

You only need the third one if you intend to run the top-down pipeline. Bottom-up works without it.


## Pre-processing

### Data preparation

The first step is to generate the multi-resolution data for the training part. In our paper, we downsample the original data by factors of 2×, 4×, and 8×, to have four levels of resolution.

We provide a script to generate multi-resolution data for four datasets we used in the paper: Synthetic Blender (*nerf_synthetic*), Mip-NeRF360 (*360*), Tanks&Temples (*tandt*), and Deep Blending (*db*). You can run the script with the following command:
```bash
python dataset_prepare.py --source_base <path to the source dataset root directory> --dataset_name <name of the dataset> --output_base <path to the output dataset root directory>
```

Note that that similar to MipNeRF360 and original 3DGS, we target images at resolutions in the 1-1.6K pixel range. Therefore, for the *360* dataset, images whose width exceeds 1600 pixels will be automatically resized to 1600 pixels.

### Dataset structure

For example, we can generate the dataset hierarchy of dataset *db* with the following command:
```bash
python dataset_prepare.py --source_base ./raw_dataset --dataset_name db --output_base ./source
```

You should have the following file structure for the model training:
```
project
└── raw_dataset # source_base
    ├── db # dataset_name
    │   └── playroom # scene
    │       ├── sparse
    │       └── images
└── source # output_base
    ├── db # dataset_name
    │   └── playroom # scene
    │       └── playroom_res1
    │           ├── sparse
    │           └── images
    │       └── playroom_res2
    │           ├── sparse
    │           └── images
    │       └── playroom_res4
    │           ├── sparse
    │           └── images
    │       └── playroom_res8
    │           ├── sparse
    │           └── images
```

As for NeRF Synthetic dataset, the structure is as follows:
```
project
└── source
    ├── nerf_synthetic
    │   └── lego
    │       └── lego_res1
    │           ├── train
    │           ├── test
    │           ├── transforms_test.json
    │           └── transforms_train.json
    │       └── lego_res2
    │           ├── train
    │           ├── test
    │           ├── transforms_test.json
    │           └── transforms_train.json
    │       └── lego_res4
    │           ├── train
    │           ├── test
    │           ├── transforms_test.json
    │           └── transforms_train.json
    │       └── lego_res8
    │           ├── train
    │           ├── test
    │           ├── transforms_test.json
    │           └── transforms_train.json
```


## Running

There are two pipelines and one SLURM driver per pipeline. Pick the bottom-up one to reproduce the published LapisGS; pick the top-down one to construct the layers from a full-resolution model first.

### Bottom-up pipeline (original LapisGS)

```bash
python train_full_pipeline.py \
    --model_base <output root> --dataset_base <source root> \
    --dataset_name <dataset> --scene <scene> \
    --method lapis
```

<details>
<summary><span style="font-weight: bold;">Please click here to see the arguments for the `train_full_pipeline.py` script.</span></summary>

| Parameter | Type | Description |
| :-------: | :--: | :---------: |
| `--model_base`   | `str`   | Path to the output model root directory. |
| `--dataset_base` | `str`   | Path to the source root directory. |
| `--dataset_name` | `str`   | Name of the dataset of scenes. |
| `--scene`        | `str`   | Name of the scene. |
| `--method`       | `str`   | Output folder name (any string). When set to `"lapis"` or `"freeze"`, `--regime` is auto-derived for backward compatibility. |
| `--regime`       | `str`   | Training regime: `"lapis"` (LapisGS — ancestor opacity stays trainable, the proposed method) or `"freeze"` (L3GS-style — ancestors fully frozen). Required when `--method` is neither `"lapis"` nor `"freeze"`. |
| `--lambda_dssim` | `float` | D-SSIM weight for the per-layer training loss. Default `0.2`. |
| `--iterations`   | `int`   | Iterations per layer. Default `30000`. |
| `--no_eval`      | flag    | Disable the train/test split (no test cameras saved into `cfg_args`). By default `--eval` is passed to `train.py`. |
| `-w`             | flag    | White background (use for NeRF synthetic scenes). |

</details>
<br>

For example, to train the proposed *lapis* method on the *playroom* scene of dataset *db*:
```bash
python train_full_pipeline.py \
    --model_base ./model --dataset_base ./source \
    --dataset_name db --scene playroom --method lapis
```

The file structure after training:
```
project
└── source # dataset_base
    ├── db # dataset_name
    │   └── playroom # scene
    │       ├── playroom_res1
    │       ├── playroom_res2
    │       ├── playroom_res4
    │       └── playroom_res8
└── model # model_base
    ├── db # dataset_name
    │   └── playroom # scene
    │       └── lapis # method
    │           ├── playroom_res1   # finest LOD (contains layers 1..4)
    │           ├── playroom_res2
    │           ├── playroom_res4
    │           └── playroom_res8   # coarsest LOD (layer 1 only)
```

### Top-down pipeline (importance-scored)

```bash
python train_full_pipeline_topdown.py \
    --model_base <output root> --dataset_base <source root> \
    --dataset_name <dataset> --scene <scene> \
    --method <output folder name> \
    [--layer_size <int>]    # see below
```

The pipeline runs in three or four steps depending on whether `--layer_size` is set:

1. **Full-resolution pretrain** on `{scene}_res1` (standard 3DGS, densification on).
2. **(Optional, L3GS mode only)** Prune the model to `n_layers * layer_size` splats and fine-tune the survivors to recover quality. See [`prune_finetune.py`](prune_finetune.py). Skipped when `--layer_size` is unset.
3. **Score + partition.** Run [`score_sort_partition.py`](score_sort_partition.py): score every Gaussian by the L3GS volume-weighted opacity-sum, sort the model by descending importance, and write `n_layers` bucket PLYs. Two modes:
   - `--layer_size <d>` — *L3GS mode*: each bucket has exactly `d` splats (paired with step 2's prune to `N*d`).
   - `--layer_size` unset — *equal-split mode*: split the full model into `n_layers` near-equal bands summing to `M`.
4. **Layered fine-tune** from coarsest to finest: layer `k` is initialised with bucket `k`, stacked on top of the cumulative frozen base (layers 0..k-1), and fine-tuned at resolution `2^(n_layers-1-k)`. Layer ancestors are either dynamic-opacity (LapisGS regime, default) or fully frozen (L3GS regime, `--no_dynamic_opacity`).

<details>
<summary><span style="font-weight: bold;">Please click here to see the arguments for the `train_full_pipeline_topdown.py` script.</span></summary>

| Parameter | Type | Description |
| :-------: | :--: | :---------: |
| `--model_base`             | `str`   | Output model root. |
| `--dataset_base`           | `str`   | Source root. |
| `--dataset_name`           | `str`   | Dataset folder name. |
| `--scene`                  | `str`   | Scene name. |
| `--method`                 | `str`   | Output method subfolder name. Default `"lapis_topdown"`. |
| `--n_layers`               | `int`   | Number of LOD layers. Default `4`. Layer `k` trains at resolution `2^(n_layers-1-k)`, so for `N=4`: L0=res8, L1=res4, L2=res2, L3=res1. |
| `--layer_size`             | `int`   | Splats per layer bucket. Set (e.g. `45000`) for L3GS mode (prune to `N*layer_size`, then partition into `N` bands of `d`). Unset for equal-split mode (no prune; partition the full model into `N` near-equal bands). |
| `--full_iterations`        | `int`   | Iterations for the full-resolution pretrain (step 1). Default `30000`. |
| `--prune_iterations_total` | `int`   | Total iterations for the prune+finetune stage (step 2). Default `35000` (L3GS default). |
| `--prune_at`               | `int`   | Iteration at which the one-shot prune fires. Default `30001`. |
| `--prune_type`             | `str`   | Importance scoring variant for pruning: `v_important_score` (default), `important_score`, `max_v_important_score`, `count`, `opacity`. |
| `--v_pow`                  | `float` | Volume-weighting exponent in the importance score. Default `0.1` (L3GS default). |
| `--layer_iterations`       | `int`   | Iterations for each per-layer fine-tune (step 4). Default `30000`. |
| `--lambda_dssim`           | `float` | D-SSIM weight. Default `0.2`. |
| `--no_dynamic_opacity`     | flag    | Fully freeze ancestor layers (L3GS regime) instead of leaving their opacity trainable (LapisGS regime, default). |
| `--no_eval`                | flag    | Disable the train/test split. |
| `-w`                       | flag    | White background (NeRF synthetic). |

</details>
<br>

The file structure after a top-down run (equal-split mode, `N=4`, method name `lapis_topdown`):

```
model/db/playroom/lapis_topdown/
├── playroom_full_res1/         # step 1: full 3DGS pretrain
├── buckets/                    # step 3: sorted-and-partitioned artefacts
│   ├── sorted_full.ply
│   ├── imp_score.npz
│   ├── sort_index.npy
│   └── layer_0.ply .. layer_3.ply
├── L0_res8/                    # step 4 — coarsest LOD (= layer 0 only)
├── L1_res4/                    # = layers 0+1 cumulative
├── L2_res2/                    # = layers 0+1+2
└── L3_res1/                    # = layers 0+1+2+3 (finest LOD)
```

In L3GS mode (`--layer_size` set) there is an additional intermediate dir:

```
└── playroom_pruned_<target_num>/   # step 2: pruned + recovery-finetuned
```

Each `L*_res*` directory is a standalone, renderable LapisGS model — its PLY contains layers `0..k` concatenated, with the frozen prefix first. `render.py` and `metrics.py` work on a layer directory directly.

## Evaluation

The fastest path is the driver scripts: by default they invoke `render.py` and `metrics.py` per layer after training. If you want to do it manually:

```bash
python render.py  -m <path to trained layer model> --skip_train
python metrics.py -m <path to trained layer model>
```

`--skip_train` writes only the test renders (the train-set renders are large and rarely needed). To render only the top-N most important Gaussians from an importance-sorted model (e.g. `buckets/sorted_full.ply`):

```bash
python render.py -m <model dir> --skip_train --num_gaussians 45000
```

## Comparison of top-down vs bottom-up pipelines

### DB (playroom + drjohnson) as an example

The eight runs we report are the cross-product of `{top-down, bottom-up} × {with, without dynamic-opacity}`, plus two extra top-down runs with the L3GS-mode `LAYER_SIZE` set to 100k and 400k splats per layer. All runs use `--lambda_dssim=0.2` and 30000 iterations per layer, averaged across DB's two scenes.

#### 1. Top-down vs bottom-up

<p align="center">
    <a href="">
        <img src="/images/comparison_db_row1.png" alt="top-down vs bottom-up on DB" width="95%">
    </a>
</p>

Top-down wins on **both axes simultaneously** — better quality at a smaller model size. At every LOD, top-down models are ~30–35% smaller than bottom-up at the same level, with equal or better PSNR/SSIM/LPIPS.

| Regime  | LOD       | Bottom-up                                  | Top-down                                  | Δ (top-down − bottom-up) |
|---------|-----------|--------------------------------------------|-------------------------------------------|--------------------------|
| opacity | L0 (res8) | 263 MB / 27.01 / 0.841 / 0.142             | **167 MB / 27.24 / 0.845 / 0.138**        | −36 % size, +0.23 PSNR   |
| opacity | L1 (res4) | 494 MB / 28.29 / 0.871 / 0.145             | **333 MB / 28.46 / 0.874 / 0.140**        | −33 % size, +0.17 PSNR   |
| opacity | L2 (res2) | 747 MB / 28.93 / 0.889 / 0.174             | **500 MB / 29.06 / 0.890 / 0.171**        | −33 % size, +0.13 PSNR   |
| opacity | L3 (res1) | 1003 MB / 28.90 / 0.890 / 0.276            | **666 MB / 28.93 / 0.887 / 0.280**        | −34 % size, ≈ same       |


The size advantage comes from a structural difference between the two pipelines. Bottom-up *adds* new Gaussians at each finer resolution on top of a frozen base, so the total splat count grows monotonically without any global budget. Top-down trains one full-detail model and then partitions it into N equal-size buckets, so the total splat count is the size of the original 3DGS model — typically much smaller than the sum of LapisGS's coarse base plus four densification passes.

#### 2. Effect of per-layer splat count (top-down)

<p align="center">
    <a href="">
        <img src="/images/comparison_db_row2.png" alt="effect of per-layer splat count on DB" width="95%">
    </a>
</p>

The L3GS mode lets you set the per-layer splat budget directly via `LAYER_SIZE`, which controls both the prune target and the bucket size. Comparing three top-down runs with dynamic opacity:

| Variant                     | L0 size | L0 PSNR | L1 size | L1 PSNR | L2 size | L2 PSNR | L3 size | L3 PSNR |
|-----------------------------|---------|---------|---------|---------|---------|---------|---------|---------|
| `topdown_opacity_100k`      |  24 MB  | 27.39   |  47 MB  | 28.09   |  71 MB  | 28.46   |  95 MB  | 28.39   |
| `topdown_opacity_400k`      |  95 MB  | 27.34   | 189 MB  | 28.45   | 284 MB  | 29.01   | 378 MB  | 28.92   |
| `topdown_opacity` (no prune)| 167 MB  | 27.24   | 333 MB  | 28.46   | 500 MB  | 29.06   | 666 MB  | 28.93   |

Two takeaways:

- **The coarsest LOD is nearly insensitive to the layer-size budget.** L0's PSNR is 27.24–27.39 across an 8x size range, because at res8 there isn't enough image detail to discriminate. So small-budget configurations are essentially free at the coarsest LOD.
- **The finer LODs are more sensitive to the layer-size budget, but with diminishing returns.** L1 grows from 28.09 at 100k to 28.45 at 400k (+0.36), but only to 28.46 at no prune (+0.01). L2 grows from 28.46 at 100k to 29.01 at 400k (+0.55), but only to 29.06 at no prune (+0.05). L3 grows from 28.39 at 100k to 28.92 at 400k (+0.53), but only to 28.93 at no prune (+0.01).


#### 3. With vs without dynamic-opacity optimization (top-down)

<p align="center">
    <a href="">
        <img src="/images/comparison_db_row3.png" alt="dynamic-opacity vs full-freeze on DB" width="95%">
    </a>
</p>

The two runs use **identical** training data, splat budgets, and freeze plumbing. They differ only in whether ancestor layers' opacity is allowed to re-tune as new layers are added on top.

| LOD       | `topdown_no_opacity`           | `topdown_opacity`              | Δ PSNR | Δ SSIM | Δ LPIPS |
|-----------|--------------------------------|--------------------------------|--------|--------|---------|
| L0 (res8) | 167 MB / 27.28 / 0.844 / 0.137 | 170 MB / 27.24 / 0.845 / 0.138 |  −0.04 | +0.001 | +0.001  |
| L1 (res4) | 333 MB / 28.20 / 0.865 / 0.151 | 339 MB / 28.46 / 0.874 / 0.140 |  +0.26 | +0.009 | −0.011  |
| L2 (res2) | 500 MB / 28.61 / 0.876 / 0.191 | 509 MB / 29.06 / 0.890 / 0.171 |  +0.45 | +0.014 | −0.020  |
| L3 (res1) | 667 MB / 28.44 / 0.871 / 0.301 | 677 MB / 28.93 / 0.887 / 0.280 |  +0.49 | +0.016 | −0.021  |

Two observations:

- **At L0 there is no effect**, as expected: the base layer has no ancestors, so the dynamic-opacity flag has nothing to act on. The two runs are within floating-point noise of each other at L0 for all metrics.
- **The benefit grows monotonically with the number of frozen layers above it.** Each newly-added layer gets to re-tune the opacity of every layer beneath it, so the gap widens from 0 PSNR at L0, to +0.26 at L1, to +0.49 at L3. SSIM and LPIPS show the same monotone trend.

The size column is nearly unchanged because dynamic opacity only re-tunes a single floating-point value per ancestor splat. So this is essentially free quality: at the finest LOD the top-down LapisGS regime delivers +0.49 PSNR / +0.016 SSIM / −0.021 LPIPS over the equivalent L3GS-style full-freeze regime at the exact same model size.





## Misc

### How to extract the enhanced layer

Note that for *both* pipelines, the higher-LOD checkpoint contains the lower-LOD splats. For bottom-up, `<scene>_res1` is the highest resolution and `<scene>_res8` is the lowest; for top-down, `L{N-1}_res1` is the highest and `L0_res8` is the lowest.

We construct the merged GS with a specially designed order: the lower layers come first as the foundation base, and the enhanced layer is stitched behind the foundation base, as shown in the figure below. As the foundation base is frozen to adaptive control, one can easily extract the enhanced layer by performing the operation `GS[size_of_foundation_layers:]`.

<p align="center">
    <a href="">
        <img src="/images/model_structure.png" alt="model_structure" width="70%">
    </a>
</p>

### Lambda D-SSIM and model size

Both pipelines now default `--lambda_dssim 0.2` (standard 3DGS), which keeps model sizes moderate and works well on real-world scenes. On simple synthetic objects (e.g. NeRF Synthetic *lego*), the low-resolution layers may underdensify because L1 is insensitive to low-res artefacts; if you see degenerate coarse layers or low PSNR there, raise `--lambda_dssim` (we've used up to 0.8). The opposite knob — raising `--densify_grad_threshold` — is the right lever if you instead want *smaller* models at the current `--lambda_dssim`.



### SLURM driver scripts

For HPC users we ship two job scripts under `scripts/` that wrap the Python pipelines, add per-layer render + metrics, and (optionally) clean up large artefacts after the metrics are computed:

- [`scripts/train_full_pipeline_bottomup.sh`](scripts/train_full_pipeline_bottomup.sh)
- [`scripts/train_full_pipeline_topdown.sh`](scripts/train_full_pipeline_topdown.sh)

Each one reads its configuration from environment variables, so you typically do not need to edit the script — just override the variables on the `sbatch` command line.

<details>
<summary><span style="font-weight: bold;">Common environment variables (both scripts).</span></summary>

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_BASE`        | `/home/e/…/gs/model`  | Output root passed as `--model_base`. |
| `DATASET_BASE`      | `/home/e/…/gs/source` | Source root passed as `--dataset_base`. |
| `DATASET`           | `db`                  | Dataset folder name. One dataset per invocation. |
| `SCENES`            | `playroom drjohnson`  | Space-separated list of scene names. |
| `METHOD`            | varies                | Output method subfolder name (see per-script defaults). |
| `LAMBDA_DSSIM`      | `0.2`                 | D-SSIM weight. |
| `DYNAMIC_OPACITY`   | `yes`                 | `yes` → LapisGS regime; `no` → L3GS-style full freeze. |
| `EVAL`              | `yes`                 | `yes` → standard train/test split; `no` → no test cameras. |
| `RUN_TRAIN`         | `yes`                 | Run the training pipeline for the scene. |
| `RUN_RENDER`        | `yes`                 | Render test images for each LOD. |
| `RUN_METRICS`       | `yes`                 | Compute PSNR/SSIM/LPIPS for each LOD. |
| `CLEANUP`           | `yes`                 | After metrics for a layer, write `model_size.json` and delete the PLY + rendered images. |

</details>
<br>

Top-down-only extras: `N_LAYERS` (default `4`) and `LAYER_SIZE` (unset → equal-split; set → L3GS mode). Bottom-up-only extras: `ITERATIONS` (default `30000`).

A few common invocations:

```bash
# Bottom-up LapisGS, dynamic-opacity regime
METHOD=bottomup_opacity DYNAMIC_OPACITY=yes \
    DATASET=db SCENES="playroom drjohnson" \
    sbatch scripts/train_full_pipeline_bottomup.sh

# Bottom-up "freeze" regime (no dynamic opacity)
METHOD=bottomup_no_opacity DYNAMIC_OPACITY=no \
    DATASET=db SCENES="playroom drjohnson" \
    sbatch scripts/train_full_pipeline_bottomup.sh

# Top-down, equal-split, dynamic-opacity regime (LapisGS-style top-down)
METHOD=topdown_opacity DYNAMIC_OPACITY=yes \
    DATASET=db SCENES="playroom drjohnson" \
    sbatch scripts/train_full_pipeline_topdown.sh

# Top-down, L3GS mode with 4 layers of 45k each
METHOD=topdown_l3gs45k LAYER_SIZE=45000 DYNAMIC_OPACITY=no \
    DATASET=db SCENES="playroom drjohnson" \
    sbatch scripts/train_full_pipeline_topdown.sh

# NeRF synthetic — white background is detected automatically from DATASET
DATASET=nerf_synthetic SCENES="lego chair" \
    METHOD=topdown_opacity \
    sbatch scripts/train_full_pipeline_topdown.sh
```

#### Cleanup behaviour

With `CLEANUP=yes` (default), after `metrics.py` writes `results.json` for a layer the driver also:

1. Runs [`scripts/record_model_size.py`](scripts/record_model_size.py) to save a ~150-byte `model_size.json` sidecar capturing the PLY's byte size and splat count.
2. Deletes `point_cloud/iteration_*/point_cloud.ply` and the rendered images under `train/ours_*/` and `test/ours_*/`.

For the top-down script, when the finest LOD's `results.json` exists, the driver additionally deletes the *intermediate* artefacts that are no longer needed once all LODs are trained: the full pretrain PLY under `{scene}_full_res1/`, the L3GS pruned model PLY under `{scene}_pruned_*/` (if present), and the entire `buckets/` directory.

What is *kept* per layer dir:

```
cfg_args            # so render.py / metrics.py can still resolve source_path / sh_degree / bg
results.json        # quality metrics
per_view.json       # per-image quality
model_size.json     # splat count + PLY bytes
```

Set `CLEANUP=no` to keep PLYs and images around for re-rendering or debugging. The plot script gracefully falls back to `model_size.json` when the PLY is gone, so the model-size axis still works after cleanup.

### CUDA out-of-memory error

If a per-layer training step OOMs, the most direct knobs are:

1. **`--densify_grad_threshold`** — raise it to densify less aggressively (fewer splats).
2. **`--densify_until_iter`** — lower it to stop densifying earlier.
3. **`--lambda_dssim`** — lower it (less SSIM, less large-feature emphasis).

In top-down mode you can also cap the final splat budget directly via `--layer_size`: the prune step deletes the lowest-importance Gaussians until exactly `n_layers * layer_size` remain, which bounds peak memory in the per-layer fine-tunes.



## Citation

If you find our code or paper useful, please cite

```latex
@inproceedings{shi2025lapisgs,
  author        = {Shi, Yuang and 
                   Morin, G{\'e}raldine and 
                   Gasparini, Simone and 
                   Ooi, Wei Tsang},
  title         = {{LapisGS}: Layered Progressive {3D} {Gaussian} Splatting for Adaptive Streaming},
  booktitle     = {Proceedings of the 2025 International Conference on 3D Vision (3DV)},
  pages         = {991--1000},
  year          = {2025},
  organization  = {IEEE}
}
```

The top-down pipeline uses the importance-scoring rasterizer and one-shot prune-to-target recipe from L3GS and LightGaussian; if you use that pipeline, please cite both as well:

```latex
@inproceedings{tsai2025l3gs,
  title={L3GS: Layered 3D Gaussian splats for efficient 3D scene delivery},
  author={Tsai, Yi-Zhen and Zhang, Xuechen and Li, Zheng and Chen, Jiasi},
  booktitle={Proceedings of the 31st Annual International Conference on Mobile Computing and Networking},
  pages={453--467},
  year={2025}
}

@article{fan2024lightgaussian,
  title={LightGaussian: Unbounded 3d gaussian compression with 15x reduction and 200+ fps},
  author={Fan, Zhiwen and Wang, Kevin and Wen, Kairun and Zhu, Zehao and Xu, Dejia and Wang, Zhangyang},
  journal={Advances in neural information processing systems},
  volume={37},
  pages={140138--140158},
  year={2024}
}
```


## The First Streaming System for Dynamic 3DGS

Based on our LapisGS, we built the first ever dynamic 3DGS streaming system named LTS, which achieves superior performance in both live streaming and on-demand streaming. Our work "LTS: A DASH Streaming System for Dynamic Multi-Layer 3D Gaussian Splatting Scenes" won the 🏆**Best Paper Award**🏆 at ACM MMSys'25 in March 2025. Access to the [Paper](https://drive.google.com/file/d/1iDz1ExOd1LrPhA7fv4DbLUbzn-Jioihn/view?usp=share_link).

Related to LapisGS, I extended it to support dynamic scenes, and released the [Code](https://github.com/nus-vv-streams/dynamic-lapis-gs/) of Dynamic-LapisGS.

If you find our code or paper useful, please cite

```latex
@inproceedings{sun2025lts,
  author       = {Yuan{-}Chun Sun and
                  Yuang Shi and
                  Cheng{-}Tse Lee and
                  Mufeng Zhu and
                  Wei Tsang Ooi and
                  Yao Liu and
                  Chun{-}Ying Huang and
                  Cheng{-}Hsin Hsu},
  title        = {{LTS:} {A} {DASH} Streaming System for Dynamic Multi-Layer {3D} {Gaussian}
                  Splatting Scenes},
  booktitle    = {Proceedings of the 16th {ACM} Multimedia Systems Conference, MMSys
                  2025, Stellenbosch, South Africa, 31 March 2025 - 4 April 2025},
  pages        = {136--147},
  publisher    = {{ACM}},
  year         = {2025},
  url          = {https://doi.org/10.1145/3712676.3714445},
  doi          = {10.1145/3712676.3714445},
}
```
