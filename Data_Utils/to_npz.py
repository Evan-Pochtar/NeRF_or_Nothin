import os
import json
import numpy as np
import cv2
from PIL import Image

def create_npz(data_dir, output_file="lego_data_200_200.npz"):
    intrinsics_file = os.path.join(data_dir, "camera_intrinsics.json")
    with open(intrinsics_file, "r") as f:
        intrinsics = json.load(f)
    focal_length = np.float32(50)
    render_data_file = os.path.join(data_dir, "render_data.json")
    with open(render_data_file, "r") as f:
        render_data = json.load(f)
    images = []
    poses = []
    for key, value in render_data.items():
        image_path = value["rgb_file"]
        with Image.open(image_path) as img:
            rgb_image = np.array(img.convert("RGB"), dtype=np.uint8)
        rgb_image = rgb_image.astype(np.float32) / 255.0
        rgb_image = cv2.GaussianBlur(rgb_image, (5, 5), 1)
        images.append(rgb_image)
        rotation_matrix = np.array(value["pose"], dtype=np.float32)[:3, :3]
        pose = np.array(value["pose"], dtype=np.float32)
        poses.append(pose)
    poses = np.array(poses)
    poses = np.array(poses, dtype=np.float32)
    np.savez_compressed(output_file, images=images, poses=poses, focal=focal_length)
dir = "/home/will/Documents/csci5561/scenes/lego_output_final/"
create_npz(dir)
