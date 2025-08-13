#!/usr/bin/env python3
import cv2
from PIL import Image
import imagehash

def canny_128(image_path):
    """Load image, grayscale, Canny edge detect, resize to 128x128."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    edges = cv2.Canny(img, 100, 200, L2gradient=True)
    resized = cv2.resize(edges, (128, 128), interpolation=cv2.INTER_AREA)
    return Image.fromarray(resized)

def compute_rotation_hashes(image_path):
    """Return a list of 360 dhashes, one per rotation degree, as integers."""
    image = canny_128(image_path)
    hashes = []
    for angle in range(360):
        rotated = image.rotate(angle)
        hash_obj = imagehash.dhash(rotated)
        # Convert hex string to integer directly
        hash_int = int(str(hash_obj), 16)
        hashes.append(hash_int)
    return hashes

def hamming_distance(int_hash1, int_hash2):
    """Compute Hamming distance between two integer hashes."""
    x = int_hash1 ^ int_hash2  # XOR to find differing bits
    return bin(x).count('1')   # Count set bits

def compare_hash_sets(hashes1, hashes2):
    """
    Compare two lists of integer hashes (360 each),
    return the minimum Hamming distance over all rotation pairs.
    """
    min_dist = 64  # max for 64-bit hash
    for h1 in hashes1:
        for h2 in hashes2:
            dist = hamming_distance(h1, h2)
            if dist < min_dist:
                min_dist = dist
                if min_dist == 0:
                    return 0
    return min_dist

if __name__ == "__main__":
    hashes1 = compute_rotation_hashes("image1.jpg")
    hashes2 = compute_rotation_hashes("image2.jpg")
    dist = compare_hash_sets(hashes1, hashes2)
    print(f"Minimum Hamming distance between images: {dist}")
