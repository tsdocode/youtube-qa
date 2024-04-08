from openai import OpenAI
import json

from ytb_clone.src.utils import pil_to_base64, replace_from_pattern_with_youtube_link


client = OpenAI()

INIT = """
You are an Youtube Video Analysis, your ask is answer question based on provided video transcribe and video's images

Here is useful transcribe and frames:

Remember to response with time range references with this format:
[FROM->TO], FROM and TO must be second

Response in markdown also. Response as friendly as you are an people instead of a bot. Do not include any image
"""


def get_response(related_texts, alone_images, question):
    text_chat = []

    for text in related_texts:
        item = related_texts[text]
        text_chat.append(
            {
                "type": "text",
                "text": f"""
                    Transcribe from second {item["start"]} to second {item["end"]}, content: {text}
                """,
            }
        )

        if len(related_texts[text]["frames"]) > 0:
            images = [pil_to_base64(i) for i in related_texts[text]["frames"]]

            text_chat.append(
                {
                    "type": "text",
                    "text": """
                       Here are some image frames from this time range:
                    """,
                }
            )

            for image in images:
                text_chat.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image},
                    }
                )

    text_chat.append(
        {
            "type": "text",
            "text": """
                       Some other frames without transcribe:
                    """,
        }
    )

    alone_images = [pil_to_base64(i) for i in alone_images]

    text_chat.extend(
        [
            {
                "type": "image_url",
                "image_url": {"url": i},
            }
            for i in alone_images
        ]
    )

    import json

    with open("debug.json", "w") as f:
        json.dump(text_chat, f)

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {"role": "system", "content": [{"type": "text", "text": INIT}]},
            {
                "role": "user",
                "content": [
                    *text_chat,
                    {"type": "text", "text": question},
                ],
            },
        ],
        max_tokens=500,
        stream=True,
    )

    return stream_response(response)


def stream_response(response):
    text = ""
    for chunk in response:
        yield "data:" + json.dumps({
            "text": chunk.choices[0].delta.content,
            "end": False
        }) + "\n\n"
        
        if chunk.choices[0].delta.content:
            text += chunk.choices[0].delta.content

    yield "data:" + json.dumps({
        "text": replace_from_pattern_with_youtube_link(text),
        "end": True
    }) + "\n\n"