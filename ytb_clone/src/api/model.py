from pydantic import BaseModel


class VidImportParams(BaseModel):
    video_url: str


class VidQueryParams(BaseModel):
    video_id: str
    question: str
