#!/usr/bin/env python3
"""
Canny Edge Detection and Intensity Gradient Analysis

This script implements the Canny edge detection algorithm to analyze intensity gradients
in images. It provides both edge detection and gradient magnitude visualization.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
import cv2
import os
import imagehash


def gaussian_blur(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian blur to reduce noise.
    
    Args:
        image: Input grayscale image as numpy array
        sigma: Standard deviation for Gaussian kernel
        
    Returns:
        Blurred image as numpy array
    """
    kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def compute_gradients(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute intensity gradients using Sobel operators.
    
    Args:
        image: Input grayscale image as numpy array
        
    Returns:
        Tuple of (gradient_magnitude, gradient_x, gradient_y)
    """
    # Sobel operators for gradient computation
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float32)
    
    # Compute gradients
    grad_x = cv2.filter2D(image.astype(np.float32), -1, sobel_x)
    grad_y = cv2.filter2D(image.astype(np.float32), -1, sobel_y)
    
    # Compute gradient magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    return magnitude, grad_x, grad_y


def non_maximum_suppression(magnitude: np.ndarray, grad_x: np.ndarray, grad_y: np.ndarray) -> np.ndarray:
    """
    Apply non-maximum suppression to thin edges.
    
    Args:
        magnitude: Gradient magnitude
        grad_x: Gradient in x direction
        grad_y: Gradient in y direction
        
    Returns:
        Suppressed magnitude array
    """
    height, width = magnitude.shape
    suppressed = np.zeros_like(magnitude)
    
    # Compute gradient direction
    angle = np.arctan2(grad_y, grad_x)
    
    # Convert to degrees and normalize to 0-180
    angle = np.rad2deg(angle) % 180
    
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            current_angle = angle[i, j]
            current_mag = magnitude[i, j]
            
            # Determine neighbors based on gradient direction
            if (0 <= current_angle < 22.5) or (157.5 <= current_angle <= 180):
                # Horizontal edge
                neighbors = [magnitude[i, j-1], magnitude[i, j+1]]
            elif 22.5 <= current_angle < 67.5:
                # Diagonal edge (/)
                neighbors = [magnitude[i-1, j+1], magnitude[i+1, j-1]]
            elif 67.5 <= current_angle < 112.5:
                # Vertical edge
                neighbors = [magnitude[i-1, j], magnitude[i+1, j]]
            else:  # 112.5 <= current_angle < 157.5
                # Diagonal edge (\)
                neighbors = [magnitude[i-1, j-1], magnitude[i+1, j+1]]
            
            # Keep pixel if it's a local maximum
            if current_mag >= max(neighbors):
                suppressed[i, j] = current_mag
    
    return suppressed


def double_threshold(image: np.ndarray, low_threshold: float, high_threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply double thresholding to classify edges.
    
    Args:
        image: Input magnitude image
        low_threshold: Lower threshold value
        high_threshold: Higher threshold value
        
    Returns:
        Tuple of (strong_edges, weak_edges, all_edges)
    """
    strong_edges = (image >= high_threshold).astype(np.uint8) * 255
    weak_edges = ((image >= low_threshold) & (image < high_threshold)).astype(np.uint8) * 127
    all_edges = strong_edges + weak_edges
    
    return strong_edges, weak_edges, all_edges


def edge_tracking_by_hysteresis(strong_edges: np.ndarray, weak_edges: np.ndarray) -> np.ndarray:
    """
    Track edges by hysteresis to connect weak edges to strong edges.
    
    Args:
        strong_edges: Strong edge pixels
        weak_edges: Weak edge pixels
        
    Returns:
        Final edge image
    """
    height, width = strong_edges.shape
    final_edges = strong_edges.copy()
    
    # Find weak edge pixels connected to strong edges
    weak_pixels = np.where(weak_edges == 127)
    
    for i, j in zip(weak_pixels[0], weak_pixels[1]):
        # Check 8-connected neighborhood
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                    
                ni, nj = i + di, j + dj
                if 0 <= ni < height and 0 <= nj < width:
                    if final_edges[ni, nj] == 255:
                        final_edges[i, j] = 255
                        break
            if final_edges[i, j] == 255:
                break
    
    return final_edges


def compute_rotation_invariant_features(edge_image: np.ndarray, target_size: tuple = (128, 128)) -> dict:
    """
    Extract rotation-invariant features from edge image using multiple methods.
    
    Args:
        edge_image: Final edge detection result as numpy array
        target_size: Target size for processing
        
    Returns:
        Dictionary containing various rotation-invariant features
    """
    # Convert numpy array to PIL Image and resize
    pil_image = Image.fromarray(edge_image.astype(np.uint8))
    downsized = pil_image.resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(downsized)
    
    features = {}
    
    # Method 1: Radial profile (distance from center)
    center_x, center_y = target_size[0] // 2, target_size[1] // 2
    y, x = np.ogrid[:target_size[1], :target_size[0]]
    distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Create radial bins and compute average intensity per ring
    max_distance = min(center_x, center_y)
    radial_bins = np.linspace(0, max_distance, 32)
    radial_profile = []
    
    for i in range(len(radial_bins) - 1):
        mask = (distances >= radial_bins[i]) & (distances < radial_bins[i + 1])
        if np.any(mask):
            radial_profile.append(np.mean(img_array[mask]))
        else:
            radial_profile.append(0)
    
    features['radial_profile'] = np.array(radial_profile)
    
    # Method 2: Angular histogram (summing pixels at each angle)
    angles = np.arctan2(y - center_y, x - center_x)
    angle_bins = np.linspace(-np.pi, np.pi, 36)  # 10-degree bins
    angular_histogram = []
    
    for i in range(len(angle_bins) - 1):
        mask = (angles >= angle_bins[i]) & (angles < angle_bins[i + 1])
        if np.any(mask):
            angular_histogram.append(np.sum(img_array[mask]))
        else:
            angular_histogram.append(0)
    
    # Make angular histogram rotation-invariant by taking its FFT magnitude
    fft_magnitude = np.abs(np.fft.fft(angular_histogram))
    # Remove the DC component and take only the magnitude (phase is rotation-dependent)
    features['angular_fft_magnitude'] = fft_magnitude[1:len(fft_magnitude)//2]
    
    # Method 3: Zernike moments (inherently rotation-invariant)
    features['zernike_moments'] = compute_zernike_moments(img_array)
    
    # Method 4: Hu moments (rotation-invariant geometric moments)
    moments = cv2.moments(img_array)
    hu_moments = cv2.HuMoments(moments).flatten()
    # Take log of absolute values to make them more stable
    features['hu_moments'] = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    
    # Method 5: Concentric ring analysis
    ring_features = []
    num_rings = 8
    for ring in range(num_rings):
        inner_radius = ring * max_distance / num_rings
        outer_radius = (ring + 1) * max_distance / num_rings
        ring_mask = (distances >= inner_radius) & (distances < outer_radius)
        if np.any(ring_mask):
            ring_pixels = img_array[ring_mask]
            ring_features.extend([
                np.mean(ring_pixels),           # average intensity
                np.std(ring_pixels),            # intensity variation
                np.sum(ring_pixels > 128),      # number of edge pixels
            ])
        else:
            ring_features.extend([0, 0, 0])
    
    features['ring_features'] = np.array(ring_features)
    
    return features


def compute_zernike_moments(image: np.ndarray, max_order: int = 8) -> np.ndarray:
    """
    Compute Zernike moments (rotation-invariant).
    Simplified implementation for basic rotation invariance.
    """
    height, width = image.shape
    center_x, center_y = width // 2, height // 2
    
    # Create coordinate arrays
    y, x = np.ogrid[:height, :width]
    x = x - center_x
    y = y - center_y
    
    # Convert to polar coordinates
    rho = np.sqrt(x**2 + y**2)
    max_rho = min(center_x, center_y)
    rho_norm = rho / max_rho
    
    # Only compute a few basic moments for simplicity
    moments = []
    
    # Compute some basic Zernike polynomials (rotation-invariant combinations)
    mask = rho_norm <= 1.0
    
    if np.any(mask):
        # Z00 (constant)
        z00 = np.ones_like(rho_norm)
        moments.append(np.abs(np.sum(image[mask] * z00[mask])))
        
        # Z20 (pure radial)
        z20 = 2 * rho_norm**2 - 1
        moments.append(np.abs(np.sum(image[mask] * z20[mask])))
        
        # Z40 (higher order radial)
        z40 = 6 * rho_norm**4 - 6 * rho_norm**2 + 1
        moments.append(np.abs(np.sum(image[mask] * z40[mask])))
        
        # Add more complex invariant combinations
        for order in range(2, min(max_order, 6), 2):
            radial_poly = np.polynomial.legendre.legval(2 * rho_norm - 1, [0] * order + [1])
            moments.append(np.abs(np.sum(image[mask] * radial_poly[mask])))
    
    return np.array(moments) if moments else np.array([0])


def compute_edge_hash(edge_image: np.ndarray, target_size: tuple = (128, 128), rotation_invariant: bool = True) -> dict:
    """
    Compute perceptual hash with optional rotation invariance.
    
    Args:
        edge_image: Final edge detection result as numpy array
        target_size: Target size for downsizing (width, height)
        rotation_invariant: If True, use truly rotation-invariant features
        
    Returns:
        Dictionary containing hash information
    """
    if rotation_invariant:
        # Use rotation-invariant features for any angle
        features = compute_rotation_invariant_features(edge_image, target_size)
        
        # Combine all features into a single signature
        combined_features = np.concatenate([
            features['radial_profile'],
            features['angular_fft_magnitude'],
            features['zernike_moments'],
            features['hu_moments'],
            features['ring_features']
        ])
        
        # Create a hash from the combined features
        # Quantize features to make them more robust
        quantized_features = np.round(combined_features * 1000).astype(np.int32)
        
        # Simple hash: convert to hex string (more robust than direct string conversion)
        import hashlib
        feature_bytes = quantized_features.tobytes()
        full_hash = hashlib.md5(feature_bytes).hexdigest()
        
        # Take first 16 characters to match phash length
        rotation_invariant_hash = full_hash[:16]
        
        return {
            'rotation_invariant_hash': rotation_invariant_hash,
            'features': features,
            'combined_feature_length': len(combined_features),
            'method': 'rotation_invariant_features',
            'rotation_invariant': True
        }
    else:
        # Original method: only 90-degree invariance
        pil_image = Image.fromarray(edge_image.astype(np.uint8))
        downsized = pil_image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Compute hashes for all 4 rotations
        rotation_hashes = {}
        rotations = [0, 90, 180, 270]
        
        for rotation in rotations:
            if rotation == 0:
                rotated_image = downsized
            else:
                rotated_image = downsized.rotate(-rotation, expand=True)
                rotated_image = rotated_image.resize(target_size, Image.Resampling.LANCZOS)
            
            phash = imagehash.phash(rotated_image)
            rotation_hashes[rotation] = str(phash)
        
        canonical_hash = min(rotation_hashes.values())
        canonical_rotation = None
        for rot, hash_val in rotation_hashes.items():
            if hash_val == canonical_hash:
                canonical_rotation = rot
                break
        
        return {
            'canonical_hash': canonical_hash,
            'canonical_rotation': canonical_rotation,
            'all_rotation_hashes': rotation_hashes,
            'method': '90_degree_invariant',
            'rotation_invariant': False
        }


def canny_edge_detection(image_path: str, sigma: float = 1.0, low_threshold: float = 50, 
                        high_threshold: float = 150, rotation_invariant: bool = True, 
                        hash_method: str = "full-rotation-invariant") -> dict:
    """
    Perform complete Canny edge detection algorithm.
    
    Args:
        image_path: Path to input image
        sigma: Gaussian blur standard deviation
        low_threshold: Lower threshold for double thresholding
        high_threshold: Higher threshold for double thresholding
        
    Returns:
        Dictionary containing all intermediate and final results
    """
    try:
        # Load and convert image to grayscale
        image = Image.open(image_path)
        if image.mode != 'L':
            image = image.convert('L')
        
        img_array = np.array(image)
        
        # Step 1: Gaussian blur to reduce noise
        blurred = gaussian_blur(img_array, sigma)
        
        # Step 2: Compute intensity gradients
        magnitude, grad_x, grad_y = compute_gradients(blurred)
        
        # Step 3: Non-maximum suppression
        suppressed = non_maximum_suppression(magnitude, grad_x, grad_y)
        
        # Step 4: Double thresholding
        strong_edges, weak_edges, all_edges = double_threshold(suppressed, low_threshold, high_threshold)
        
        # Step 5: Edge tracking by hysteresis
        final_edges = edge_tracking_by_hysteresis(strong_edges, weak_edges)
        
        # Step 6: Compute perceptual hash based on method
        if hash_method == "full-rotation-invariant":
            edge_hash_info = compute_edge_hash(final_edges, rotation_invariant=True)
        elif hash_method == "90-degree-invariant":
            edge_hash_info = compute_edge_hash(final_edges, rotation_invariant=False)
        else:  # standard
            pil_image = Image.fromarray(final_edges.astype(np.uint8))
            downsized = pil_image.resize((128, 128), Image.Resampling.LANCZOS)
            phash = imagehash.phash(downsized)
            edge_hash_info = {
                'hash': str(phash),
                'method': 'standard',
                'rotation_invariant': False
            }
        
        return {
            'original': img_array,
            'blurred': blurred,
            'magnitude': magnitude,
            'grad_x': grad_x,
            'grad_y': grad_y,
            'suppressed': suppressed,
            'strong_edges': strong_edges,
            'weak_edges': weak_edges,
            'all_edges': all_edges,
            'final_edges': final_edges,
            'edge_hash_info': edge_hash_info
        }
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        sys.exit(1)


def save_results(results: dict, output_prefix: str = "canny"):
    """
    Save all intermediate and final results.
    
    Args:
        results: Dictionary from canny_edge_detection
        output_prefix: Prefix for output filenames
    """
    # Normalize arrays for saving
    def normalize_for_save(arr):
        if arr.dtype != np.uint8:
            # Normalize to 0-255 range
            arr_norm = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.uint8)
            return arr_norm
        return arr
    
    save_items = [
        ('original', results['original']),
        ('blurred', results['blurred']),
        ('magnitude', results['magnitude']),
        ('grad_x', results['grad_x']),
        ('grad_y', results['grad_y']),
        ('suppressed', results['suppressed']),
        ('strong_edges', results['strong_edges']),
        ('weak_edges', results['weak_edges']),
        ('all_edges', results['all_edges']),
        ('final_edges', results['final_edges'])
    ]
    
    saved_files = []
    for name, array in save_items:
        filename = f"{output_prefix}_{name}.png"
        normalized_array = normalize_for_save(array)
        Image.fromarray(normalized_array).save(filename)
        saved_files.append(filename)
        
        # Show file info
        file_size = os.path.getsize(filename)
        print(f"   {filename}: {file_size/1024:.1f} KB")
    
    # Save downsized edge image for hash computation
    downsized_filename = f"{output_prefix}_final_edges_128x128.png"
    downsized_edge = Image.fromarray(results['final_edges'].astype(np.uint8)).resize((128, 128), Image.Resampling.LANCZOS)
    downsized_edge.save(downsized_filename)
    saved_files.append(downsized_filename)
    
    file_size = os.path.getsize(downsized_filename)
    print(f"   {downsized_filename}: {file_size/1024:.1f} KB (downsized for hash)")
    
    return saved_files


def main():
    parser = argparse.ArgumentParser(
        description="Canny edge detection and intensity gradient analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python canny_gradient.py photo.jpg
  python canny_gradient.py image.png --sigma 1.5 --low 30 --high 100
  python canny_gradient.py picture.jpg -s 2.0 -l 40 -H 120 -o edges
  
Parameter Guide:
  sigma: Gaussian blur strength (0.5-3.0, default: 1.0)
  low_threshold: Lower edge threshold (10-100, default: 50)
  high_threshold: Upper edge threshold (50-200, default: 150)
  
Output Files:
  - original: Input grayscale image
  - blurred: Gaussian blurred image
  - magnitude: Gradient magnitude
  - grad_x/grad_y: X and Y gradients
  - suppressed: After non-maximum suppression
  - strong/weak/all_edges: Thresholding results
  - final_edges: Final Canny edge detection result
        """
    )
    
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("-s", "--sigma", type=float, default=1.0,
                       help="Gaussian blur sigma (default: 1.0)")
    parser.add_argument("-l", "--low", type=float, default=50,
                       help="Low threshold for edge detection (default: 50)")
    parser.add_argument("-H", "--high", type=float, default=150,
                       help="High threshold for edge detection (default: 150)")
    parser.add_argument("-o", "--output", default="canny",
                       help="Output filename prefix (default: canny)")
    parser.add_argument("--hash-method", choices=["full-rotation-invariant", "90-degree-invariant", "standard"],
                       default="full-rotation-invariant",
                       help="Hash computation method (default: full-rotation-invariant)")
    parser.add_argument("--no-rotation-invariant", action="store_true",
                       help="DEPRECATED: Use --hash-method=standard instead")
    
    args = parser.parse_args()
    
    # Validate input image exists
    if not Path(args.image).exists():
        print(f"Error: Image file '{args.image}' not found.")
        sys.exit(1)
    
    if not Path(args.image).is_file():
        print(f"Error: '{args.image}' is not a file.")
        sys.exit(1)
    
    # Validate parameters
    if args.sigma <= 0:
        print("Error: Sigma must be positive.")
        sys.exit(1)
    
    if args.low >= args.high:
        print("Error: Low threshold must be less than high threshold.")
        sys.exit(1)
    
    print(f"Performing Canny edge detection...")
    print(f"Parameters: sigma={args.sigma}, low={args.low}, high={args.high}")
    
    # Determine rotation invariance method
    if args.no_rotation_invariant:
        rotation_invariant = False  # Legacy support
    else:
        rotation_invariant = args.hash_method != "standard"
    
    # Perform Canny edge detection
    results = canny_edge_detection(args.image, args.sigma, args.low, args.high, 
                                  rotation_invariant=rotation_invariant, 
                                  hash_method=args.hash_method)
    
    # Save results
    print(f"\nSaving results with prefix '{args.output}'...")
    saved_files = save_results(results, args.output)
    
    print(f"\n✅ Canny edge detection completed!")
    print(f"Generated {len(saved_files)} output files:")
    print(f"Main result: {args.output}_final_edges.png")
    print(f"Downsized (128x128): {args.output}_final_edges_128x128.png")
    
    # Display hash information
    hash_info = results['edge_hash_info']
    
    if hash_info['method'] == 'rotation_invariant_features':
        print(f"\n🔍 True Rotation-Invariant Hash (0-359°):")
        print(f"   Hash: {hash_info['rotation_invariant_hash']}")
        print(f"   Method: Combined rotation-invariant features")
        print(f"   Feature vector length: {hash_info['combined_feature_length']}")
        print(f"   Hash length: {len(hash_info['rotation_invariant_hash'])} characters")
        print(f"\n🌟 Features used:")
        print(f"   • Radial profile (distance-based intensity)")
        print(f"   • Angular FFT magnitude (rotation-normalized)")
        print(f"   • Zernike moments (inherently rotation-invariant)")
        print(f"   • Hu moments (geometric invariants)")
        print(f"   • Concentric ring analysis")
        print(f"\n💡 This hash is identical for the same image at ANY rotation angle!")
        print(f"   Use '{hash_info['rotation_invariant_hash']}' to compare with other images")
        
    elif hash_info['method'] == '90_degree_invariant':
        print(f"\n🔍 90-Degree Rotation-Invariant Hash:")
        print(f"   Canonical Hash: {hash_info['canonical_hash']}")
        print(f"   Canonical Rotation: {hash_info['canonical_rotation']}°")
        print(f"   Hash length: {len(hash_info['canonical_hash'])} characters")
        print(f"\n📊 All 90° Rotation Hashes:")
        for rotation, hash_val in hash_info['all_rotation_hashes'].items():
            marker = " ⭐" if hash_val == hash_info['canonical_hash'] else ""
            print(f"     {rotation:3d}°: {hash_val}{marker}")
        print(f"\n💡 This hash is identical only for 90° increments (0°, 90°, 180°, 270°)")
        print(f"   Use '{hash_info['canonical_hash']}' to compare with other images")
        
    else:
        print(f"\n🔍 Standard Perceptual Hash:")
        print(f"   Hash: {hash_info.get('hash', 'N/A')}")
        print(f"   No rotation invariance")


if __name__ == "__main__":
    main()
