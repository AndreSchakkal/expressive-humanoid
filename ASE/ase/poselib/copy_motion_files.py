import os
import shutil
import yaml
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Copy files listed in config.yaml from the source folder to the destination folder."
    )
    # parser.add_argument("config", default="ASE/ase/poselib/data/configs/motions_50.yaml", help="Path to the config.yaml file")
    # parser.add_argument("dest", required=False, default="ASE/ase/poselib/data/configs/motions_50.yaml/50", help="Destination folder where files will be copied")
    args = parser.parse_args()
    # args.config = "data/configs/motions_50.yaml"
    # args.dest = "data/configs/50/"

    args.config = "data/configs/motions_50+andre_additions.yaml"
    args.dest = "data/configs/50+andre_additions/"
    
    # Load the YAML config file.
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    motions = config.get("motions", {})
    if "root" not in motions:
        print("Error: 'root' key not found in the motions section of the config.")
        return

    src_root = motions["root"]
    src_root = "data/g1_retarget_npy"

    # Create destination directory if it doesn't exist.
    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    # Iterate over each key in motions (skip the 'root' key).
    for name, details in motions.items():
        if name == "root":
            continue
        # name = name + ".npy"
        name = name + "_key_bodies.npy"
        # The file/directory name is assumed to be the key.
        src_path = os.path.join(src_root, name)
        dest_path = os.path.join(args.dest, name)

        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
            print(f"Copied {src_path} to {dest_path}")
        else:
            print(f"Warning: {src_path} does not exist.")

if __name__ == "__main__":
    main()
