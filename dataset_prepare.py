import os
import sys
from PIL import Image
from argparse import ArgumentParser



def downsample_image(raw_dataset_path, image_path, ds_factor):
    im = Image.open(os.path.join(raw_dataset_path, image_path)) 
    orig_width, orig_height = im.size 
    new_size = round(orig_width/(ds_factor)), round(orig_height/(ds_factor))
    im1 = im.resize(new_size)
    
    return im1


if __name__ == "__main__":

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--source_base', type=str, required=True, help="Path to the source dataset root directory")
    parser.add_argument('--dataset_name', type=str, required=True, help="Name of the dataset")
    parser.add_argument('--output_base', type=str, required=True, help="Path to the output root directory")
    args = parser.parse_args(sys.argv[1:])
    
    
    dataset_base = args.source_base
    dataset_name = args.dataset_name
    output_base = args.output_base
    ds_factors = [1, 2, 4, 8] # Downsampling factors: 1 means original size, 2 means half size, 4 means quarter size, 8 means eighth size
    dataset_scene_map = {"db": ["playroom","drjohnson"], 
                         "tandt": ["train", "truck"], 
                         "nerf_synthetic": ["lego", "chair", "drums", "ficus", "hotdog", "materials", "mic", "ship",],
                         "360": ["bonsai", "counter", "flowers", "garden", "kitchen", "room", "treehill"]}
    
    if dataset_name == "db" or dataset_name == "tandt":
        for scene in dataset_scene_map[dataset_name]:
            dataset_root = os.path.join(dataset_base, dataset_name)
            raw_dataset_path = os.path.join(dataset_root, scene, "images")

            for ds_factor in ds_factors:
                print(f"Processing resolution 1/{ds_factor} for {dataset_name} {scene}")
                
                output_dir = os.path.join(output_base, dataset_name, scene, f"{scene}_res{ds_factor}", "images")
                os.makedirs(output_dir, exist_ok=True)
                
                # downsample the images
                for image_path in os.listdir(raw_dataset_path):
                    if image_path.endswith("jpg") and not image_path.startswith("."):
                        ds_image = downsample_image(raw_dataset_path, image_path, ds_factor)
                        ds_image.save(os.path.join(output_dir, image_path))

                # copy the folder sparse to the new folder
                sparse_path = os.path.join(dataset_root, scene, "sparse")
                output_sparse_path = os.path.join(output_base, dataset_name, scene, f"{scene}_res{ds_factor}")
                os.system(f"cp -r {sparse_path} {output_sparse_path}")
    elif dataset_name == "360":
        for scene in dataset_scene_map[dataset_name]:
            dataset_root = os.path.join(dataset_base, dataset_name)
            raw_dataset_path = os.path.join(dataset_root, scene, "images")

            for ds_factor in ds_factors:
                print(f"Processing resolution 1/{ds_factor} for {dataset_name} {scene}")
                
                output_dir = os.path.join(output_base, dataset_name, scene, f"{scene}_res{ds_factor}", "images")
                os.makedirs(output_dir, exist_ok=True)
                
                # downsample the images
                if ds_factor == 1:
                    for image_path in os.listdir(raw_dataset_path):
                        if image_path.endswith("JPG") and not image_path.startswith("."):
                            im = Image.open(os.path.join(raw_dataset_path, image_path)) 
                            orig_width, orig_height = im.size 
                            # downsample the image to 1600 pixel-width, following the original 3DGS settings.
                            ds_image = downsample_image(raw_dataset_path, image_path, float(orig_width/1600))
                            ds_image.save(os.path.join(output_dir, image_path))
                else:
                    res1_dataset_path = os.path.join(output_base, dataset_name, scene, f"{scene}_res1", "images")
                    for image_path in os.listdir(res1_dataset_path):
                        if image_path.endswith("JPG") and not image_path.startswith("."):
                            ds_image = downsample_image(res1_dataset_path, image_path, ds_factor)
                            ds_image.save(os.path.join(output_dir, image_path))

                # copy the folder sparse to the new folder
                sparse_path = os.path.join(dataset_root, scene, "sparse")
                output_sparse_path = os.path.join(output_base, dataset_name, scene, f"{scene}_res{ds_factor}")
                os.system(f"cp -r {sparse_path} {output_sparse_path}")
    elif dataset_name == "nerf_synthetic":
        splits = ["train","test"]
        for scene in dataset_scene_map[dataset_name]:
            for ds_factor in ds_factors:
                print(f"Processing resolution 1/{ds_factor} for {dataset_name} {scene}")
                
                for split in splits:
                    dataset_root = os.path.join(dataset_base, dataset_name)
                    raw_dataset_path = os.path.join(dataset_root, scene, split)
                    json_transforms_path = os.path.join(dataset_root, scene, f"transforms_{split}.json")
                    
                    output_dir = os.path.join(output_base, dataset_name, scene, f"{scene}_res{ds_factor}", split)
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # downsample the images
                    for image_path in os.listdir(raw_dataset_path):
                        if image_path.endswith("png") and not image_path.startswith("."):
                            ds_image = downsample_image(raw_dataset_path, image_path, ds_factor)
                            ds_image.save(os.path.join(output_dir, image_path))

                    # copy the transform_matrix to the new folder
                    output_json_transforms_path = os.path.join(output_base, dataset_name, scene, f"{scene}_res{ds_factor}")
                    os.system(f"cp -r {json_transforms_path} {output_json_transforms_path}")
    else:
        print("Dataset name not recognized. Please check the dataset_name argument.")
        sys.exit(1)