import sys
sys.path.append('.')
from add_db import compute_rotation_hashes, client

def match_image(image_path, collection_name="image_rotations", top=1):
    # Step 1: Compute 360 hashes for the query image (integers)
    from app import compute_rotation_hashes as compute_hashes_int, hamming_distance, compare_hash_sets
    query_hashes = compute_hashes_int(image_path)  # List of 360 int hashes

    # Step 2: Convert to database vector (flattened bits)
    from add_db import compute_rotation_hashes as compute_hashes_bits
    query_vector = compute_hashes_bits(image_path)  # List of 360*64 floats

    # Step 3: Search for best match in database
    from qdrant_client.http.models import NamedVector
    named_vector = NamedVector(name="rotation_hashes", vector=query_vector)
    results = client.search(
        collection_name,
        named_vector,
        limit=top,
        with_payload=True,
        with_vectors=True
    )
    if not results:
        print("No match found in database.")
        return None

    # Step 4: Reconstruct 360 hashes from best match's vector
    def reconstruct_hashes(db_vector):
        hashes = []
        for i in range(360):
            bits = db_vector[i*64:(i+1)*64]
            hash_int = sum(int(b) << j for j, b in enumerate(bits))
            hashes.append(hash_int)
        return hashes

    best_match = results[0]
    if not best_match.vector or 'rotation_hashes' not in best_match.vector:
        print("Error: No vector found in best match result.")
        return None
    db_vector = best_match.vector['rotation_hashes']
    db_hashes = reconstruct_hashes(db_vector)

    # Step 5: Compare using hamming distance
    min_dist = compare_hash_sets(query_hashes, db_hashes)

    # Step 6: Direct vector comparison (cosine similarity and Euclidean distance)
    import numpy as np
    def cosine_similarity(vec1, vec2):
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

    def euclidean_distance(vec1, vec2):
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return np.linalg.norm(v1 - v2)

    cosine_sim = cosine_similarity(query_vector, db_vector)
    euclid_dist = euclidean_distance(query_vector, db_vector)

    print(f"Best match ID: {best_match.id}, Score: {best_match.score}, Min Hamming distance: {min_dist}")
    print(f"Cosine similarity (vectors): {cosine_sim}")
    print(f"Euclidean distance (vectors): {euclid_dist}")
    return {
        "match_id": best_match.id,
        "score": best_match.score,
        "min_hamming_distance": min_dist,
        "cosine_similarity": cosine_sim,
        "euclidean_distance": euclid_dist,
        "payload": best_match.payload
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python match_image.py <image_path>")
        sys.exit(1)
    match_image(sys.argv[1])