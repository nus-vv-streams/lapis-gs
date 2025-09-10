
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

- 🏆 **2025/09**: **Best Paper Award** at ACM SIGCOMM EMS'25. Based on LapisGS, we built a system called NETSPLAT ([Short Paper](https://dl.acm.org/doi/10.1145/3746441.3748225)) that leverages data plane programmability to provide network assistance for 3DGS streaming. It is WiP, but we won the Best Paper Award at ACM SIGCOMM EMS'25...Again!
- 🏆 **2025/04**: **Best Paper Award** at ACM MMSys'25. We extend LapisGS to dynamic scenes (Dynamic-LapisGS, Check the [Code](https://github.com/nus-vv-streams/dynamic-lapis-gs/)). Based on Dynamic-LapisGS, we built the first ever dynamic 3DGS streaming system named LTS ([Paper](https://doi.org/10.1145/3712676.3714445)), and won the Best Paper Award at ACM MMSys'25!


## Setup

Our work is built on the codebase of the original 3D Gaussian Splatting method. Please refer to the [original 3D Gaussian Splatting repository](https://github.com/graphdeco-inria/gaussian-splatting) for details about requirements.

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


```bash
python train_full_pipeline.py --model_base <path to the output model root directory> --dataset_base <path to the source root directory> --dataset_name <name of the dataset> --scene <name of the scene> --method <name of the method>
```

<details>
<summary><span style="font-weight: bold;">Please click here to see the arguments for the `train_full_pipeline.py` script.</span></summary>

| Parameter | Type | Description |
| :-------: | :--: | :---------: |
| `--model_base`   | `str` | Path to the output model root directory.|
| `--dataset_base` | `str` | Path to the source root directory. |
| `--dataset_name` | `str` | Name of the dataset of scenes. |
| `--scene`        | `str` | Name of the scene. |
| `--method`       | `str` | Name of the method we build the LOD 3DGS. Can be `"lapis"` (the proposed method) and `"freeze"` (freeze all attributes of the previous layer(s) when train the current layer). |

</details>
<br>

For example, we train the model for the scene *playroom* from dataset *db* with the proposed method *lapis*, using the command:
```bash
python train_full_pipeline.py --model_base ./model --dataset_base ./source --dataset_name db --scene playroom --method lapis
```

The file structure after training should be as follows:
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
    │           ├── playroom_res1
    │           ├── playroom_res2
    │           ├── playroom_res4
    │           └── playroom_res8
```

### How to extract the enhanced layer

Note that \<scene\>_res1 is the highest resolution, and \<scene\>_res8 is the lowest resolution. The model is trained from the lowest resolution to the highest resolution. The model stored in the higher resolution folder contains not only the higher layer but also the lower layer(s).

We construct the merged GS with a specially designed order: the lower layers come first as the foundation base, and the enhanced layer is stiched behind the foundation base, as shown in the figure below. As the foundation base is frozen to optimization and adaptive control, one can easily extract the enhanced layer by performing the operation like GS[size_of_foundation_layers:].

<p align="center">
    <a href="">
        <img src="/images/model_structure.png" alt="model_structure" width="70%">
    </a>
</p>

### CUDA out-of-memory error

Through experiments, we found that the default loss function is not sensitive to low-resolution images, making optimization and desification failed. It is because in the default loss function, L1 loss is attached much more importance (0.8), but L1 loss is not sensitive to finer details, blurriness, or low-resolution artifacts. Therefore, the loss, computed from the default loss function, would be small at low layers, disabling the parameter update and adaptive control for the low-layer Gaussian splats. Therefore, we set lambda_dssim to 0.8 to emphasize the structural similarity loss, which is more sensitive to low-resolution artifacts and then causes much heavier desification, finally producing bigger 3DGS model. 

To reduce the model size, you may try to 1) lower down the lambda_dssim, or 2) increase the densification threshold. Also, generally speaking, it is not necessary to make it SSIM-sensitive for complex scenes. For example, we note that training LapisGS for complex scene *playroom* with default lambda_dssim 0.2 can still produce reasonable layered structure, while it fails for simple object *lego*.



## Evaluation

We use the following command to evaluate the model:
```bash
python render.py -m <path to trained model> # Generate renderings
python metrics.py -m <path to trained model> # Compute error metrics on renderings
```


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