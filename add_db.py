#!/usr/bin/env python3

import cv2
from PIL import Image
import imagehash
import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.http import models as http_models
import uuid
import os

# Qdrant cloud credentials
client = QdrantClient(
    host=os.getenv("QDRANT_HOST"),
    api_key=os.getenv("QDRANT_API_KEY")
)

def compute_rotation_hashes(image_path):
    """Return a flattened list of 360 dhash bit vectors (360*64 floats)."""
    from app import canny_128
    def dhash_bits(hash_obj, hash_size=8):
        hash_int = int(str(hash_obj), 16)
        return [(hash_int >> i) & 1 for i in range(hash_size * hash_size)]

    image = canny_128(image_path)
    bits = []
    for angle in range(360):
        rotated = image.rotate(angle)
        hash_obj = imagehash.dhash(rotated)
        bits.extend([float(b) for b in dhash_bits(hash_obj)])  # 64 bits per rotation
    return bits  # Length: 360*64 = 23040

def ensure_collection(collection_name):
    vector_config = {
        "rotation_hashes": http_models.VectorParams(size=23040, distance="Cosine")
    }
    collections = client.get_collections().collections
    if not any(c.name == collection_name for c in collections):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=vector_config
        )

def main(image_path):
    collection_name = "image_rotations"
    ensure_collection(collection_name)
    hashes = compute_rotation_hashes(image_path)
    image_obj = Image.open(image_path).convert('L').resize((128, 128))
    image_hash_str = str(imagehash.dhash(image_obj))
    image_hash_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, image_hash_str))
    upsert_result = client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=image_hash_uuid,
                payload={"image_name": image_path},
                vector={"rotation_hashes": hashes},
            ),
        ],
    )

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python add_db.py <image_path>")
        sys.exit(1)
    main(sys.argv[1])