import cv2
import numpy as np
import logging
from app import orb_feature_match
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def rotate_image(image, angle):
    """Rotate a PIL image by the given angle."""
    return image.rotate(angle)

def flip_image(image, mode):
    """Flip a PIL image vertically, horizontally, or both."""
    if mode == 'vertical':
        return image.transpose(Image.FLIP_TOP_BOTTOM)
    elif mode == 'horizontal':
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    elif mode == 'both':
        return image.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.FLIP_LEFT_RIGHT)
    return image

def pil_to_cv(image):
    """Convert PIL image to OpenCV grayscale numpy array."""
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    pil_img = Image.open("images/Untitled18.jpg")
    fails_log = open('fails.log', 'w')
    for angle in range(360):
        rotated = rotate_image(pil_img, angle)
        states = {
            'base': rotated,
            'vertical': flip_image(rotated, 'vertical'),
            'horizontal': flip_image(rotated, 'horizontal'),
            'both': flip_image(rotated, 'both')
        }
        for state_name, state_img in states.items():
            temp_path = f".temp_rotated.jpg"
            state_img.save(temp_path)
            img1 = cv2.imread("image1.jpg", cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
            orb = cv2.ORB_create()
            kp1, des1 = orb.detectAndCompute(img1, None)
            kp2, des2 = orb.detectAndCompute(img2, None)
            if des1 is not None and des2 is not None:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                good_matches = [m for m in matches if m.distance < 50]
                matched = len(good_matches) >= 5
                log_msg = f"State: {state_name}, Rotation: {angle}, Matched: {str(matched).lower()}, ORB Good Matches: {len(good_matches)}"
            else:
                matched = False
                log_msg = f"State: {state_name}, Rotation: {angle}, Matched: false, ORB Good Matches: 0"
            if matched:
                logging.info(log_msg)
            else:
                logging.error(log_msg)
                fails_log.write(log_msg + '\n')
    fails_log.close()
