import httpx
import json
from httpx_sse import connect_sse
import time


def import_video_stream(video_url):
    url = "http://localhost:8001/import"
    data = {"video_url": video_url}
    timeout = httpx.Timeout(60, connect=60.0)

    with httpx.stream("POST", url=url, json=data, timeout=timeout) as r:
        for chunk in r.iter_raw():  # or, for line in r.iter_lines():
            chunk = str(chunk)[7:-5]
            print(chunk)
            yield chunk
            
def get_stream_response(question, video_id):
    url = "http://localhost:8001/query"
    data = {"question": question, "video_id": video_id}
    timeout = httpx.Timeout(60, connect=60.0)

    with httpx.stream("POST", url=url, json=data, timeout=timeout) as r:
        for chunk in r.iter_raw():  # or, for line in r.iter_lines():
            chunk = chunk.decode('utf-8')

            # Remove the leading 'data:' string
            chunk = chunk.replace('data:', '')
            chunk = json.loads(chunk)
            
            yield chunk['text']
            