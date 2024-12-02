import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from skimage.metrics import structural_similarity as ssim
import cv2
import os

def load_depth_map(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: Missing file - {file_path}")
    image = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Error: Unable to read file - {file_path}")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image.astype(np.float32)

def calculate_errors(gt, pred):
    mae = mean_absolute_error(gt.flatten(), pred.flatten())
    mse = mean_squared_error(gt.flatten(), pred.flatten())
    ssim_score = ssim(gt, pred, data_range=gt.max() - gt.min())
    return mae, mse, ssim_score

if __name__ == "__main__":
    try:
        ground_truth = load_depth_map("ground_truth_depth.png")
        nerf_depth = load_depth_map("nerf_depth.png")
        midas_depth = load_depth_map("midas_depth.png")
        stereo_depth = load_depth_map("stereo_depth.png")
    except (FileNotFoundError, ValueError) as e:
        print(e)
        exit(1)

    errors = {}
    for method, depth_map in [("NERF", nerf_depth), ("MiDaS", midas_depth), ("Stereo Vision", stereo_depth)]:
        mae, mse, ssim_score = calculate_errors(ground_truth, depth_map)
        errors[method] = {"MAE": mae, "MSE": mse, "SSIM": ssim_score}

    for method, metrics in errors.items():
        print(f"Errors for {method}:")
        print(f"  MAE: {metrics['MAE']:.4f}")
        print(f"  MSE: {metrics['MSE']:.4f}")
        print(f"  SSIM: {metrics['SSIM']:.4f}")
