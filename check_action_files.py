import os
import json
import numpy as np
from pathlib import Path

def check_action_files(data_dir):
    data_dir = Path(data_dir)
    required_action_files = [
        "driving_command.bin",
        "joint_pos.bin",
        "l_hand_closure.bin",
        "neck_desired.bin",
        "r_hand_closure.bin"
    ]

    # Paths to essential files
    video_tokens_path = data_dir / "video.bin"
    metadata_path = data_dir / "metadata.json"
    action_dir = data_dir / "actions"

    # Check if essential files exist
    if not video_tokens_path.exists():
        print(f"Error: 'video.bin' not found in {data_dir}")
        return
    if not metadata_path.exists():
        print(f"Error: 'metadata.json' not found in {data_dir}")
        return
    if not action_dir.exists():
        print(f"Error: 'actions' directory not found in {data_dir}")
        return

    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    num_images = metadata["num_images"]
    s = metadata["s"]
    print(f"Number of images (frames) in 'video.bin': {num_images}")
    print(f"Image size (s x s): {s} x {s}")

    # Check the length of 'video.bin'
    try:
        token_dtype = np.dtype(metadata.get("token_dtype", "uint32"))
        video_data = np.memmap(video_tokens_path, dtype=token_dtype, mode='r')
        expected_video_size = num_images * s * s
        if video_data.size != expected_video_size:
            print(f"Error: 'video.bin' size mismatch. Expected {expected_video_size}, got {video_data.size}")
            return
        else:
            print("'video.bin' size matches the expected length.")
    except Exception as e:
        print(f"Error loading 'video.bin': {e}")
        return

    # Check each required action file
    for action_file in required_action_files:
        action_path = action_dir / action_file
        if not action_path.exists():
            print(f"Error: Required action file '{action_file}' not found in {action_dir}")
            continue  # Continue to check other files

        # Determine the expected shape and dtype for each action
        if action_file == "joint_pos.bin":
            expected_shape = (num_images, 21)
            action_dtype = np.float32
        elif action_file == "driving_command.bin":
            expected_shape = (num_images, 2)
            action_dtype = np.float32
        elif action_file in ["l_hand_closure.bin", "r_hand_closure.bin", "neck_desired.bin"]:
            expected_shape = (num_images, 1)
            action_dtype = np.float32
        else:
            print(f"Unknown action file '{action_file}'")
            continue

        # Load the action file
        try:
            action_data = np.memmap(
                action_path,
                dtype=action_dtype,
                mode='r',
                shape=expected_shape
            )
            print(f"'{action_file}' loaded successfully with shape {action_data.shape}")
        except ValueError as e:
            print(f"Error loading '{action_file}': {e}")
            # Attempt to load without specifying shape to get total size
            try:
                action_data = np.fromfile(action_path, dtype=action_dtype)
                actual_size = action_data.size
                expected_size = np.prod(expected_shape)
                if actual_size != expected_size:
                    print(f"Error: '{action_file}' size mismatch. Expected {expected_size}, got {actual_size}")
                else:
                    print(f"'{action_file}' size matches the expected total size.")
            except Exception as e:
                print(f"Error reading '{action_file}': {e}")
        except Exception as e:
            print(f"Error loading '{action_file}': {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check action files in the validation data directory")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the validation data directory')
    args = parser.parse_args()
    check_action_files(args.data_dir)
