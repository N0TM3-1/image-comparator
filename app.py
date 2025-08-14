#!/usr/bin/env python3
import cv2
from PIL import Image
import imagehash
import numpy as np

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

def orb_feature_match(img_path1, img_path2, min_matches=5):
    """
    Use ORB to detect and match keypoints between two images.
    Returns True if enough good matches are found.
    """
    img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        raise FileNotFoundError("One or both images not found.")

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return False

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    good_matches = [m for m in matches if m.distance < 50]

    print(f"ORB good matches: {len(good_matches)}")
    return len(good_matches) >= min_matches

def compare_orb_tokens(token1, token2, min_good_matches=10):
    """
    Compare two ORB descriptor sets (tokens).
    Returns True if enough good matches are found, indicating one image may be a crop of the other.
    """
    if token1 is None or token2 is None:
        return False
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(token1, token2)
    good_matches = [m for m in matches if m.distance < 50]
    print(f"ORB good matches: {len(good_matches)}")
    return len(good_matches) >= min_good_matches

def orb_token(image_path):
    """Generate ORB descriptors for a scaled 128x128 grayscale image as a token."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(img, None)
    return des

if __name__ == "__main__":
    hashes1 = compute_rotation_hashes("image1.jpg")
    hashes2 = compute_rotation_hashes("image2.jpg")
    dist = compare_hash_sets(hashes1, hashes2)
    print(f"Minimum Hamming distance between images: {dist}")

    # ORB feature matching for zoom/crop detection
    is_zoomed_or_cropped = orb_feature_match("image1.jpg", "image2.jpg")
    print(f"Zoomed or cropped image detected by ORB: {is_zoomed_or_cropped}")