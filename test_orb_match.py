#!/usr/bin/env python3
import cv2
from app import orb_token

if __name__ == "__main__":
    img1_token = orb_token("image1.jpg")
    for i in range(19):
        test_path = f"images/Untitled{i}.jpg"
        test_token = orb_token(test_path)
        if test_token is not None and img1_token is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(img1_token, test_token)
            good_matches = [m for m in matches if m.distance < 50]
            print(f"File: {test_path}; ORB Good Matches:{len(good_matches)}")
