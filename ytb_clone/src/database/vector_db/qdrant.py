import uuid

from qdrant_client import QdrantClient
from qdrant_client.http.models import Batch
from qdrant_client.models import PointStruct
from tqdm import tqdm


class QdrantDB:
    def __init__(self, collection, host, port=6333) -> None:
        self.host = host
        self.port = port
        self.collection = collection

        self.client = QdrantClient(host=self.host, port=self.port)

    def search(self, embedding, filter=None):

        hits = self.client.search(
            collection_name=self.collection,
            query_vector=embedding,
            limit=5,
            query_filter=filter,
        )

        return hits

    def split_insert(self, embeddings, payload):
        payloads = []

        for i in tqdm(range(len(embeddings))):
            payloads.append(payload)

        points_ids = self.batch_insert(vectors=embeddings, payloads=payloads)

        return points_ids

    def batch_insert(self, vectors, payloads):
        points_ids = []
        for _ in vectors:
            points_ids.append(str(uuid.uuid4()))

        self.client.upsert(
            self.collection,
            points=Batch.model_construct(
                ids=points_ids, vectors=vectors, payloads=payloads
            ),
        )

        return points_ids

    def clear(self, filters):
        self.client.delete(
            collection_name=self.collection, points_selector=filters
        )

    def insert(self, vector, payload):
        point_id = str(uuid.uuid4())
        i = self.client.upsert(
            self.collection,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )

        print(i)


if __name__ == "__main__":
    db = QdrantDB(collection="images", host="localhost", port=6333)

    db.insert([1] * 512, {"user": "tsdocode", "session": "assss"})
