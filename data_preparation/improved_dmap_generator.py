import numpy as np
from scipy.ndimage import gaussian_filter
import os
import cv2
import json
from tqdm import tqdm
import math

def calculate_adaptive_sigma(points, image_shape, min_sigma=0.5, max_sigma=3.0):
    """
    Calculate adaptive sigma based on local density of pellets
    For pellet counting, we want smaller sigma for dense regions and larger for sparse regions
    """
    if len(points) == 0:
        return min_sigma
    
    h, w = image_shape[:2]
    # Calculate average nearest neighbor distance
    if len(points) < 2:
        return min_sigma
    
    distances = []
    for i, (x1, y1) in enumerate(points):
        min_dist = float('inf')
        for j, (x2, y2) in enumerate(points):
            if i != j:
                dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                if dist < min_dist:
                    min_dist = dist
        if min_dist != float('inf'):
            distances.append(min_dist)
    
    if distances:
        avg_distance = np.mean(distances)
        # Adaptive sigma based on average distance between pellets
        # For pellets, sigma should be roughly 1/4 to 1/3 of average distance
        adaptive_sigma = max(min_sigma, min(max_sigma, avg_distance / 4.0))
        return adaptive_sigma
    
    return min_sigma

def generate_adaptive_densitymap(image, points, use_adaptive=True, base_sigma=1.0):
    """
    Generate density map with adaptive Gaussian kernels for pellet counting
    """
    h, w = image.shape[:2]
    densitymap = np.zeros((h, w), dtype=np.float32)

    print(f"Image size: {w}x{h}, #points={len(points)}")

    if len(points) == 0:
        return densitymap

    # Calculate adaptive sigma if enabled
    if use_adaptive:
        sigma = calculate_adaptive_sigma(points, image.shape)
        print(f" Using adaptive sigma: {sigma:.2f}")
    else:
        sigma = base_sigma
        print(f" Using fixed sigma: {sigma}")

    # Place points in density map
    valid_points = 0
    for x, y in points:
        if 0 <= x < w and 0 <= y < h:
            densitymap[int(y), int(x)] = 1
            valid_points += 1
    
    print(f" Placed {valid_points} valid points (out of {len(points)} total)")

    if valid_points > 0:
        # Apply Gaussian filter
        densitymap = gaussian_filter(densitymap, sigma=sigma, mode='constant')
        
        # Normalize to preserve exact count
        if densitymap.sum() > 0:
            densitymap = densitymap * (len(points) / densitymap.sum())

    print(f" Final density map sum: {densitymap.sum():.6f} (target: {len(points)})")
    return densitymap

def load_via_points(json_path, image_filename):
    """Load point annotations from VIA JSON format"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Find the entry for the image
        for key, value in data.items():
            if value.get('filename') == image_filename:
                regions = value.get('regions', [])
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
    except Exception as e:
        print(f"Error loading annotations for {image_filename}: {e}")
    
    return []

def validate_annotations(points, image_shape):
    """Validate that all annotation points are within image boundaries"""
    h, w = image_shape[:2]
    valid_points = []
    invalid_count = 0
    
    for x, y in points:
        if 0 <= x < w and 0 <= y < h:
            valid_points.append([x, y])
        else:
            invalid_count += 1
            print(f"  Warning: Point ({x}, {y}) is outside image bounds ({w}x{h})")
    
    if invalid_count > 0:
        print(f"  Removed {invalid_count} invalid points")
    
    return valid_points

def generate_density_maps(use_adaptive_sigma=True, base_sigma=1.0):
    """Main function to generate density maps for all data"""
    
    for phase in ['train', 'test']:
        print(f"\n=== Processing {phase} data ===")
        
        images_dir = f'../data/{phase}_data/images/'
        densitymaps_dir = f'../data/{phase}_data/densitymaps/'
        
        # Update JSON path to match your annotation file
        json_files = [
            '../data/via_export_json.json',
            '../data/original_796.json'  # Fallback
        ]
        
        json_path = None
        for json_file in json_files:
            if os.path.exists(json_file):
                json_path = json_file
                break
        
        if json_path is None:
            print(f"Error: No annotation file found. Checked: {json_files}")
            continue
        
        print(f"Using annotation file: {json_path}")

        if not os.path.exists(densitymaps_dir):
            os.makedirs(densitymaps_dir)

        if not os.path.exists(images_dir):
            print(f'Warning: {images_dir} does not exist. Skipping.')
            continue

        image_file_list = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Found {len(image_file_list)} images")
        
        # Statistics tracking
        total_pellets = 0
        processed_images = 0
        
        for image_file in tqdm(image_file_list, desc=f"Processing {phase}"):
            image_path = os.path.join(images_dir, image_file)
            image = cv2.imread(image_path)
            
            if image is None:   
                print(f"⚠️ Skipping unreadable file: {image_file}")
                continue

            # Load annotations
            points = load_via_points(json_path, image_file)
            
            # Validate annotations
            points = validate_annotations(points, image.shape)
            
            print(f"\n{image_file}: {len(points)} pellets found")
            total_pellets += len(points)
            
            # Generate density map
            densitymap = generate_adaptive_densitymap(
                image, 
                points, 
                use_adaptive=use_adaptive_sigma, 
                base_sigma=base_sigma
            )

            # Save density map
            save_name = os.path.splitext(image_file)[0] + ".npy"
            save_path = os.path.join(densitymaps_dir, save_name)
            np.save(save_path, densitymap)
            
            processed_images += 1

        print(f"\n✅ {phase} data summary:")
        print(f"   - Processed images: {processed_images}")
        print(f"   - Total pellets: {total_pellets}")
        print(f"   - Average pellets per image: {total_pellets/max(processed_images,1):.1f}")

if __name__ == '__main__':
    print("=== MCNN Pellet Counter - Improved Density Map Generator ===")
    print("This version includes:")
    print("- Adaptive Gaussian sigma based on pellet density")
    print("- Better annotation validation")
    print("- Improved error handling")
    print("- Enhanced statistics tracking")
    
    # You can choose between adaptive or fixed sigma
    USE_ADAPTIVE_SIGMA = True  # Set to False for fixed sigma
    BASE_SIGMA = 1.0  # Used when adaptive is False, or as base for adaptive
    
    print(f"\nConfiguration:")
    print(f"- Adaptive sigma: {USE_ADAPTIVE_SIGMA}")
    print(f"- Base sigma: {BASE_SIGMA}")
    
    generate_density_maps(use_adaptive_sigma=USE_ADAPTIVE_SIGMA, base_sigma=BASE_SIGMA)
    
    print("\n🎉 Density map generation completed!")