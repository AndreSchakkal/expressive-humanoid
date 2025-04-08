import os
import yaml
import re

# Function to load the old config file
def load_old_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

# Function to create the new config file for CMU dataset
def create_new_config(old_config, dataset_folder, new_config_file):
    new_config = {'motions': {}}
    
    # Iterate through the files in the dataset folder
    for file in os.listdir(dataset_folder):
        print(f"Checking file: {file}")  # Debugging: Print each file being checked
        if file.endswith("poses.npy") and "CMU" in file:  # Only process CMU files
            print(f"Found CMU file: {file}")  # Debugging: Check found CMU files
            
            # Updated regex to support 1-3 digits in each section
            match = re.match(r"0-CMU_(\d{1,3})_(\d{1,3})_(\d{1,3})_poses.npy", file)
            if match:
                # Use the full filename (without the '.npy' extension) as the key
                motion_key = f"0-CMU_{match.group(1)}_{match.group(2)}_{match.group(3)}_poses"
                print(f"Mapped file {file} to motion key: {motion_key}")  # Debugging: Mapped key

                # Extract matching configuration from the old config
                # If this key exists in the old config, copy its properties to the new config
                print(match.group(2) + "_" + match.group(3))
                if match.group(2) + "_" + match.group(3) in old_config['motions']:
                    new_config['motions'][motion_key] = old_config['motions'][match.group(2) + "_" + match.group(3)]
                else:
                    print(f"No match found for {motion_key} in old config.")  # Debugging: No match case
    
    # Write the new config to a file
    with open(new_config_file, 'w') as file:
        yaml.dump(new_config, file, default_flow_style=False)

    # Now add quotation marks manually to the keys in the generated file
    with open(new_config_file, 'r') as file:
        content = file.read()

    # Replace all motion keys with quoted versions
    for key in new_config['motions']:
        quoted_key = f"'{key}'"
        content = content.replace(key, quoted_key)

    # Write the updated content back to the file with quoted keys
    with open(new_config_file, 'w') as file:
        file.write(content)

    print(f"New config file '{new_config_file}' created successfully with quoted keys.")

# Main script
def main():
    old_config_file = '/home/schakkal/expressive-humanoid/ASE/ase/poselib/data/configs/additions.yaml'  # Path to your old config file
    dataset_folder = '/home/schakkal/expressive-humanoid/ASE/ase/poselib/data/g1_retarget_npy'  # Path to your dataset folder with .npy files

    new_config_file = '/home/schakkal/expressive-humanoid/ASE/ase/poselib/data/configs/additions_g1.yaml'  # Path where the new config file will be saved

    # Load the old config
    old_config = load_old_config(old_config_file)

    # Create the new config for all files ending with poses.npy
    create_new_config(old_config, dataset_folder, new_config_file)

if __name__ == '__main__':
    main()
