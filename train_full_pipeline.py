import os
import sys
from argparse import ArgumentParser


if __name__ == "__main__":

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_base', type=str, required=True, help="Path to the model root directory")
    parser.add_argument('--dataset_base', type=str, required=True, help="Path to the dataset root directory")
    parser.add_argument('--dataset_name', type=str, required=True, help="Name of the dataset")
    parser.add_argument('--scene', type=str, required=True, help="Name of the scene")
    parser.add_argument('--method', type=str, required=True, help="Name of the method", choices=["lapis", "freeze"])
    args = parser.parse_args(sys.argv[1:])


    model_base = args.model_base
    dataset_base = args.dataset_base
    dataset_name = args.dataset_name
    scene = args.scene
    method = args.method

    resolution_scales = [8, 4, 2, 1]
    train_bin = "train.py"

    train_command = ""


    for idx, resolution in enumerate(resolution_scales):
        print(f"Training {method} model for {scene} at resolution {resolution}")
        
        model_dir = os.path.join(model_base, dataset_name, scene, method, f"{scene}_res{resolution}")
        os.makedirs(model_dir, exist_ok=True) # mkdir model_dir if not exists
        source_dir = os.path.join(dataset_base, dataset_name, scene, f"{scene}_res{resolution}")
        
        if method == "lapis":
            if resolution == 8:
                train_command = f"python {train_bin} -s {source_dir} -m {model_dir} --data_device cuda --lambda_dssim 0.8"
            else:
                train_command = f"python {train_bin} -s {source_dir} -m {model_dir} --data_device cuda --lambda_dssim 0.8 --dynamic_opacity --foundation_gs_path {model_base}/{dataset_name}/{scene}/{method}/{scene}_res{resolution_scales[idx-1]}/point_cloud/iteration_30000/point_cloud.ply"
        elif method == "freeze":
            if resolution == 8:
                train_command = f"python {train_bin} -s {source_dir} -m {model_dir} --data_device cuda --lambda_dssim 0.8"
            else:
                train_command = f"python {train_bin} -s {source_dir} -m {model_dir} --data_device cuda --lambda_dssim 0.8 --foundation_gs_path {model_base}/{dataset_name}/{scene}/{method}/{scene}_res{resolution_scales[idx-1]}/point_cloud/iteration_30000/point_cloud.ply"

        # run the command lines
        os.system(train_command)