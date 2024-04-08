import torch
import clip
from PIL import Image
from typing import List

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def get_embedding(texts) -> List[List[float]]:
    """
    Retrieves embeddings for a list of image paths using the CLIP model.

    Args:
        image_paths (List[str]): A list of paths to the images for which embeddings need to be generated.

    Returns:
        List[List[float]]: A list of embeddings, where each embedding is a list of floats.

    """

    if len(texts) > 8:
        chunks = [texts[i : i + 8] for i in range(0, len(texts), 8)]
        embeddings = []
        for chunk in chunks:
            chunk_embeddings = process_chunk(chunk)
            embeddings.extend(chunk_embeddings)
        return embeddings
    else:
        return process_chunk(texts)


def process_chunk(chunk: List[str]) -> List[List[float]]:
    texts = clip.tokenize(chunk).to(device)

    with torch.no_grad():
        features = model.encode_text(texts)

    return features.tolist()


if __name__ == "__main__":
    embedding = get_embedding(["Hello", "Hi"])

    print(embedding)
    print(len(embedding))
