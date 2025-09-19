import numpy as np
from scipy.ndimage import gaussian_filter
import os
import cv2
import json
from tqdm import tqdm

def generate_fixed_kernel_densitymap(image, points, sigma=1):  # Reduced to 1 for small pellets
    h, w = image.shape[:2]
    densitymap = np.zeros((h, w), dtype=np.float32)

    print(f"Image size: {w}x{h}, #points={len(points)}")

    valid = 0
    for x, y in points:
        if 0 <= x < w and 0 <= y < h:
            densitymap[int(y), int(x)] = 1
            valid += 1
    print(f" Placed {valid} points inside image")

    if valid > 0:
        densitymap = gaussian_filter(densitymap, sigma=sigma, mode='constant')
        
        # Ensure exact count preservation
        if densitymap.sum() > 0:
            densitymap = densitymap * (len(points) / densitymap.sum())

    print(f" Density map sum: {densitymap.sum():.6f} (target: {len(points)})")
    return densitymap


def load_via_points(json_path, image_filename):
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Find the entry for the image
    for key, value in data.items():
        if value['filename'] == image_filename:
            regions = value['regions']
            points = []
            # Handle both dict and list formats
            if isinstance(regions, dict):
                region_iter = regions.values()
            elif isinstance(regions, list):
                region_iter = regions
            else:
                region_iter = []
            for region in region_iter:
                shape = region.get('shape_attributes', {})
                if 'cx' in shape and 'cy' in shape:
                    cx = shape['cx']
                    cy = shape['cy']
                    points.append([cx, cy])
            return points
    return []

if __name__ == '__main__':
    for phase in ['train', 'test']:
        images_dir = f'../data/{phase}_data/images/'
        densitymaps_dir = f'../data/{phase}_data/densitymaps/'
        json_path = '../data/original_796.json'

        if not os.path.exists(densitymaps_dir):
            os.makedirs(densitymaps_dir)

        if not os.path.exists(images_dir):
            print(f'Warning: {images_dir} does not exist. Skipping.')
            continue

        image_file_list = os.listdir(images_dir)
        for image_file in tqdm(image_file_list, desc=f"Processing {phase}"):
            image_path = os.path.join(images_dir, image_file)
            image = cv2.imread(image_path)
            if image is None:   
                print(f"⚠️ Skipping unreadable file: {image_file}")
                continue

            points = load_via_points(json_path, image_file)
            print(f"{image_file}: points found: {len(points)}")
            densitymap = generate_fixed_kernel_densitymap(image, points, sigma=1)  # Using sigma=1

            save_name = os.path.splitext(image_file)[0] + ".npy"
            np.save(os.path.join(densitymaps_dir, save_name), densitymap)

        print(f"✅ Improved density maps generated for {phase}_data.")
