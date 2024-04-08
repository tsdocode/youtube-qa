from PIL import Image
import base64
import re
import io


def pil_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        image = Image.open(image_file)
        resized_image = image.resize((512, 512))

        # Convert the image to PNG format in memory
        buffered = io.BytesIO()
        resized_image.save(buffered, format="PNG")
        image_data = buffered.getvalue()

        # Encode the PNG byte data to Base64
        base64_data = base64.b64encode(image_data)

        return f"data:image/png;base64,{base64_data.decode('utf-8')}"


def replace_from_pattern_with_youtube_link(text, youtube_link):
    pattern = r"\[(\d+)->(\d+)\]"
    matches = re.findall(pattern, text)

    replaced_text = text
    for match in matches:
        start_time = match[0]
        replaced_text = replaced_text.replace(
            f"[{match[0]}->{match[1]}]",
            f"[{match[0]}->{match[1]}]"
            + f"({youtube_link.format(start_time)})",
        )

    return replaced_text
