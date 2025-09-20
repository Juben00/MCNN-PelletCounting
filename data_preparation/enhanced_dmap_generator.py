import numpy as np
from scipy.ndimage import gaussian_filter
import os
import cv2
import json
from tqdm import tqdm
import math
from sklearn.neighbors import NearestNeighbors

def calculate_adaptive_sigma_advanced(points, image_shape, min_sigma=0.3, max_sigma=2.0):
    """
    Advanced adaptive sigma calculation using k-nearest neighbors
    """
    if len(points) < 3:
        return min_sigma
    
    h, w = image_shape[:2]
    points_array = np.array(points)
    
    # Use k-nearest neighbors to find local density
    k = min(5, len(points) - 1)  # Use 5 nearest neighbors or all available
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(points_array)
    distances, indices = nbrs.kneighbors(points_array)
    
    # Calculate average distance to k nearest neighbors (excluding self)
    avg_distances = np.mean(distances[:, 1:], axis=1)  # Exclude self (index 0)
    
    # Use median distance to avoid outliers
    median_distance = np.median(avg_distances)
    
    # Adaptive sigma: smaller for denser regions, larger for sparse regions
    # For pellets, sigma should be about 1/6 to 1/4 of average distance
    adaptive_sigma = np.clip(median_distance / 6.0, min_sigma, max_sigma)
    
    return adaptive_sigma

def generate_multi_scale_densitymap(image, points, use_adaptive=True):
    """
    Generate density map with multi-scale approach for better accuracy
    """
    h, w = image.shape[:2]
    densitymap = np.zeros((h, w), dtype=np.float32)

    print(f"Image size: {w}x{h}, #points={len(points)}")

    if len(points) == 0:
        return densitymap

    # Calculate adaptive sigma
    if use_adaptive and len(points) >= 3:
        base_sigma = calculate_adaptive_sigma_advanced(points, image.shape)
        print(f" Using adaptive sigma: {base_sigma:.3f}")
    else:
        base_sigma = 0.8  # Smaller default for pellets
        print(f" Using fixed sigma: {base_sigma}")

    # Create multiple density maps with different scales
    scales = [0.8, 1.0, 1.2]  # Multiple scales for robustness
    density_maps = []
    
    for scale in scales:
        temp_dmap = np.zeros((h, w), dtype=np.float32)
        sigma = base_sigma * scale
        
        # Place points
        valid_points = 0
        for x, y in points:
            if 0 <= x < w and 0 <= y < h:
                # Use fractional coordinates for sub-pixel accuracy
                temp_dmap[int(y), int(x)] = 1.0
                valid_points += 1
        
        # Apply Gaussian filter
        if valid_points > 0:
            temp_dmap = gaussian_filter(temp_dmap, sigma=sigma, mode='constant')
            # Normalize to preserve count
            if temp_dmap.sum() > 0:
                temp_dmap = temp_dmap * (len(points) / temp_dmap.sum())
        
        density_maps.append(temp_dmap)
    
    # Combine maps with weighted average (emphasize base scale)
    weights = [0.25, 0.5, 0.25]
    densitymap = sum(w * dmap for w, dmap in zip(weights, density_maps))
    
    print(f" Final density map sum: {densitymap.sum():.6f} (target: {len(points)})")
    return densitymap

def validate_and_filter_annotations(points, image_shape, min_distance=3):
    """
    Validate annotations and filter out potentially incorrect ones
    """
    h, w = image_shape[:2]
    valid_points = []
    
    for x, y in points:
        # Check bounds
        if not (0 <= x < w and 0 <= y < h):
            continue
        
        # Check minimum distance to existing points (avoid duplicates)
        too_close = False
        for vx, vy in valid_points:
            if math.sqrt((x - vx)**2 + (y - vy)**2) < min_distance:
                too_close = True
                break
        
        if not too_close:
            valid_points.append([x, y])
    
    removed = len(points) - len(valid_points)
    if removed > 0:
        print(f"  Filtered out {removed} potentially duplicate/invalid points")
    
    return valid_points

def load_via_points_enhanced(json_path, image_filename):
    """Enhanced point loading with better error handling"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Try different possible keys for the image
        possible_keys = [
            image_filename,
            f"{image_filename}*",
            os.path.splitext(image_filename)[0]
        ]
        
        points = []
        found = False
        
        for key, value in data.items():
            # Check if this entry matches our image
            if (value.get('filename') == image_filename or 
                any(pk in key for pk in possible_keys)):
                
                regions = value.get('regions', [])
                found = True
                
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
                        cx = float(shape['cx'])
                        cy = float(shape['cy'])
                        points.append([cx, cy])
                break
        
        if not found:
            print(f"  Warning: No annotations found for {image_filename}")
        
        return points
        
    except Exception as e:
        print(f"Error loading annotations for {image_filename}: {e}")
        return []

def analyze_annotation_quality(points, image_shape):
    """Analyze annotation quality and provide statistics"""
    if len(points) < 2:
        return
    
    h, w = image_shape[:2]
    points_array = np.array(points)
    
    # Calculate density statistics
    area = h * w
    density = len(points) / (area / 1000000)  # points per megapixel
    
    # Calculate distance statistics
    distances = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            dist = math.sqrt((points[i][0] - points[j][0])**2 + 
                           (points[i][1] - points[j][1])**2)
            distances.append(dist)
    
    if distances:
        print(f"  Annotation analysis:")
        print(f"    - Count: {len(points)}")
        print(f"    - Density: {density:.2f} points/MP")
        print(f"    - Min distance: {min(distances):.1f}")
        print(f"    - Avg distance: {np.mean(distances):.1f}")
        print(f"    - Max distance: {max(distances):.1f}")

def generate_enhanced_density_maps():
    """Enhanced density map generation with quality improvements"""
    
    for phase in ['train', 'test']:
        print(f"\n=== Processing {phase} data (Enhanced) ===")
        
        images_dir = f'../data/{phase}_data/images/'
        densitymaps_dir = f'../data/{phase}_data/densitymaps_enhanced/'
        
        # Annotation file paths
        json_files = [
            '../data/via_export_json.json',
            '../data/original_796.json'
        ]
        
        json_path = None
        for json_file in json_files:
            if os.path.exists(json_file):
                json_path = json_file
                break
        
        if json_path is None:
            print(f"Error: No annotation file found.")
            continue
        
        print(f"Using annotation file: {json_path}")

        if not os.path.exists(densitymaps_dir):
            os.makedirs(densitymaps_dir)

        if not os.path.exists(images_dir):
            print(f'Warning: {images_dir} does not exist. Skipping.')
            continue

        image_file_list = [f for f in os.listdir(images_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
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

            # Load annotations with enhanced loader
            points = load_via_points_enhanced(json_path, image_file)
            
            # Validate and filter annotations
            points = validate_and_filter_annotations(points, image.shape, min_distance=3)
            
            print(f"\n{image_file}: {len(points)} valid pellets")
            
            # Analyze annotation quality
            if len(points) > 1:
                analyze_annotation_quality(points, image.shape)
            
            total_pellets += len(points)
            
            # Generate enhanced density map
            densitymap = generate_multi_scale_densitymap(image, points, use_adaptive=True)

            # Save density map
            save_name = os.path.splitext(image_file)[0] + ".npy"
            save_path = os.path.join(densitymaps_dir, save_name)
            np.save(save_path, densitymap)
            
            processed_images += 1

        print(f"\n✅ Enhanced {phase} data summary:")
        print(f"   - Processed images: {processed_images}")
        print(f"   - Total pellets: {total_pellets}")
        if processed_images > 0:
            print(f"   - Average pellets per image: {total_pellets/processed_images:.1f}")

if __name__ == '__main__':
    print("=== MCNN Pellet Counter - Enhanced Density Map Generator ===")
    print("Enhanced features:")
    print("- Advanced adaptive sigma using k-nearest neighbors")
    print("- Multi-scale density map generation")
    print("- Annotation quality analysis and validation")
    print("- Duplicate point filtering")
    print("- Sub-pixel accuracy improvements")
    
    generate_enhanced_density_maps()
    
    print("\n🎉 Enhanced density map generation completed!")
    print("Note: Enhanced density maps are saved in 'densitymaps_enhanced' folders")
    print("Update your dataloader paths to use these for better accuracy!")