from openai import OpenAI
from typing import List

client = OpenAI()


def get_embedding(texts: List[str]) -> List[List[float]]:
    """
    Retrieves embeddings for a list of texts using OpenAI's text-embedding-ada-002 model.

    Args:
        texts (List[str]): A list of texts for which embeddings need to be generated.

    Returns:
        List[List[float]]: A list of embeddings, where each embedding is a list of floats.

    """
    response = client.embeddings.create(
        input=texts, model="text-embedding-ada-002"
    )
    result = [i.embedding for i in response.data]
    return result


if __name__ == "__main__":
    result = get_embedding(["Hello there", "Hello there"])
    print(len(result[0]))
