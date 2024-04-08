import httpx
import asyncio
import json

import logging
from httpx_sse import connect_sse


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


def test_import_video_stream():
    url = "http://localhost:8001/import"
    
    data = {"video_url": "https://www.youtube.com/watch?v=EDj-Xo8AlSU"}
    data = json.dumps(data)
    timeout = httpx.Timeout(600.0, connect=60.0)

    with httpx.Client(timeout=timeout) as client:
        with connect_sse(client, url=url, method="POST", data=data) as event_source:
            logger.info("Ready")
            for sse in event_source.iter_sse():
                print(sse.event, sse.data, sse.id, sse.retry)
                
                
def test_stream_query():
    url = "http://localhost:8001/query"
    
    data = {
        "video_id": "EDj-Xo8AlSU",
        "question": "Why he love software engineer"
    }
    
    data = json.dumps(data)
    timeout = httpx.Timeout(600.0, connect=60.0)

    with httpx.Client(timeout=timeout) as client:
        with connect_sse(client, url=url, method="POST", data=data) as event_source:
            logger.info("Ready")
            for sse in event_source.iter_sse():
                print(sse.event, sse.data, sse.id, sse.retry)
                
                
test_stream_query()