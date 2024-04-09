from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
import uvicorn
import json
from dotenv import load_dotenv
from uuid import uuid4
from qdrant_client import models

from fastapi.responses import StreamingResponse

from ytb_clone.src.api.model import VidImportParams, VidQueryParams

from ytb_clone.src.fetch.downloader.youtube import (
    download_video,
    video_to_audio,
    video_to_images,
    audio_to_text,
)

import glob

from ytb_clone.src.database.vector_db.qdrant import QdrantDB
from ytb_clone.src.llm.openai_vision import get_response

from ytb_clone.src.embedding.image.clip import get_embedding as image_embedding
from ytb_clone.src.embedding.text.openai import get_embedding as text_embedding
from ytb_clone.src.embedding.text.clip import (
    get_embedding as clip_text_embedding,
)

path_env = ".env"
load_dotenv(path_env)

app = FastAPI()

images_db = QdrantDB("images", "localhost", 6333)

texts_db = QdrantDB("texts", "localhost", 6333)


def import_images_embedding(images_files, vid_id):
    image_embs = image_embedding(images_files)

    for idx, emb in enumerate(image_embs):
        file_name = images_files[idx].split("/")[-1]
        file_name = file_name.split(".")[0]
        chunk_name = file_name[-4:]

        chunk_id = int(chunk_name)

        offset = 1 if chunk_id == 0 else 2

        payload = {
            "video_id": vid_id,
            "start": chunk_id,
            "end": chunk_id + offset,
            "data": images_files[idx],
        }

        images_db.insert(emb, payload)


def import_texts_embedding(transcribes, vid_id):
    texts = [i["text"] for i in transcribes]
    text_embs = text_embedding(texts)

    for idx, emb in enumerate(text_embs):

        payload = {
            "video_id": vid_id,
            "start": transcribes[idx]["start"],
            "end": transcribes[idx]["end"],
            "data": transcribes[idx]["text"],
        }

        texts_db.insert(emb, payload)


def import_embeddings(images_files, transcribes, vid_id: int):
    try:
        print("Importing image")
        import_images_embedding(images_files, vid_id)

        print("Importing text")

        import_texts_embedding(transcribes, vid_id)
    except Exception as e:
        raise e


@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Youtube RAG"}


def import_video_stream(url=""):
    yield "data:" + json.dumps({"message": "Start processing"}) + "\n\n"

    try:
        video_id = url.split("=")[1]
    except Exception:
        video_id = uuid4()

    yield "data:" + json.dumps({"message": "Downloading.."}) + "\n\n"

    video_meta = download_video(url, video_id)
    video_path = video_meta["output_path"]

    yield "data:" + json.dumps({"message": "Extracting frames from video"}) + "\n\n"

    images_path = video_to_images(video_path, vid_id=video_id)

    image_files = glob.glob(f"{images_path}/*.png")

    yield "data:" + json.dumps({"message": "Transcribing video"}) + "\n\n"

    audio_path = video_to_audio(video_path, video_id)
    transcribes = audio_to_text(audio_path, video_id)

    yield "data:" + json.dumps(
        {"message": "Importing embedding to vector database"}
    ) + "\n\n"

    import_embeddings(image_files, transcribes, video_id)

    yield "data:" + json.dumps(
        {"message": "Import Success!", "video_id": video_id}
    ) + "\n\n"


@app.post("/import", tags=["RAG"])
async def import_video(params: VidImportParams):

    url = params.video_url

    return StreamingResponse(import_video_stream(url), media_type="text/event-stream")


@app.post("/query", tags=["RAG"])
async def query_video(params: VidQueryParams):
    video_id = params.video_id
    question = params.question

    text_emb = text_embedding([question])[0]
    clip_text_emb = clip_text_embedding([question])[0]

    related_texts = texts_db.search(
        text_emb,
        models.Filter(
            must=[
                models.FieldCondition(
                    key="video_id",
                    match=models.MatchValue(value=video_id),
                ),
            ]
        ),
    )

    related_images = images_db.search(
        clip_text_emb,
        models.Filter(
            must=[
                models.FieldCondition(
                    key="video_id",
                    match=models.MatchValue(value=video_id),
                ),
            ]
        ),
    )

    related_texts = [i.payload for i in related_texts]

    related_images = [i.payload for i in related_images]

    merged_data = {}

    added_frame = []

    for item in related_texts:
        start = item["start"]
        end = item["end"]
        merged_data[item["data"]] = {"start": start, "end": end, "frames": []}

        for frame in related_images:
            if frame["start"] >= start and frame["end"] <= end:
                merged_data[item["data"]]["frames"].append(frame["data"])
                added_frame.append(frame["data"])

    no_trans_frame = [i["data"] for i in related_images if i["data"] not in added_frame]

    ytb_url = f"https://www.youtube.com/watch?v={video_id}" + "&t={}s"

    response = get_response(merged_data, no_trans_frame, question, ytb_url)

    return StreamingResponse(response, media_type="text/event-stream")


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def main():
    uvicorn.run(app, host="localhost", port=8001)
    

if __name__ == "__main__":
    main()