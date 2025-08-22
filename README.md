# Image Comparator

This project provides tools for comparing images using rotation-invariant hashing and vector similarity search. It leverages Qdrant as a vector database to store and search image hashes, enabling fast and robust image matching even under rotation.

## Features

- Compute rotation-invariant hashes for images
- Store image hashes in a Qdrant vector database
- Search for similar images using vector similarity
- Includes test scripts for hash conversion and matching

## Project Structure

- `app.py`: Core image processing and hash computation functions
- `add_db.py`: Adds image hashes to the Qdrant database
- `match_image.py`: Searches for similar images in the database
- `requirements.txt`: Python dependencies
- `test_hash_conversion.py`, `test_orb_match.py`, `test_orb_rotate_match.py`: Test scripts

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/N0TM3-1/image-comparator.git
   cd image-comparator
   ```
2. **Create Virtual Environement and install dependencies**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Configure environment variables**:
   Create a `.env` file in the project root with the following content:
   ```env
   QDRANT_HOST=<your-qdrant-host>
   QDRANT_API_KEY=<your-qdrant-api-key>
   ```
   Replace `<your-qdrant-host>` and `<your-qdrant-api-key>` with your Qdrant Cloud credentials.

## Usage

- To add an image to the database, run:

  ```bash
  python add_db.py <path_to_your_image>
  ```

- To search for a matching image, run:
  ```bash
  python match_image.py <path_to_your_image>
  ```

## Environment Variables

- `QDRANT_HOST`: Qdrant Cloud host URL
- `QDRANT_API_KEY`: Qdrant Cloud API key

## License

This project is licenced under the [GNU General Public License 3.0](LICENSE)
