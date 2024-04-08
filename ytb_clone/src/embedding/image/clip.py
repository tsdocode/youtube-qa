import torch
import clip
from PIL import Image
from typing import List

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def get_embedding(image_paths: List[str]) -> List[List[float]]:
    """
    Retrieves embeddings for a list of image paths using the CLIP model.

    Args:
        image_paths (List[str]): A list of paths to the images for which embeddings need to be generated.

    Returns:
        List[List[float]]: A list of embeddings, where each embedding is a list of floats.

    """
    if len(image_paths) > 8:
        chunks = [
            image_paths[i : i + 8] for i in range(0, len(image_paths), 8)
        ]
        embeddings = []
        for chunk in chunks:
            chunk_embeddings = process_chunk(chunk)
            embeddings.extend(chunk_embeddings)
        return embeddings
    else:
        return process_chunk(image_paths)


def process_chunk(chunk: List[str]) -> List[List[float]]:
    images = [Image.open(i) for i in chunk]
    images = [preprocess(i).to(device) for i in images]
    images = torch.stack(images)

    with torch.no_grad():
        features = model.encode_image(images)

    return features.tolist()


if __name__ == "__main__":
    image_path = "/Users/sangtnguyen/Coding/Personal/youtube-clone/ytb-clone/test/frames/frame0177.png"
    embedding = get_embedding([image_path, image_path])

    print(embedding)
    print(len(embedding))
