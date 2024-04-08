import os
import cv2
import json
import concurrent


from pytube import YouTube
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from concurrent.futures import ThreadPoolExecutor


def download_video(url, vid_id, output_path):
    """
    Download a video from a given url and save it to the output path.

    Parameters:
    url (str): The url of the video to download.
    output_path (str): The path to save the video to.

    Returns:
    dict: A dictionary containing the metadata of the video.
    """
    yt = YouTube(url)

    metadata = {
        "author": yt.author,
        "title": yt.title,
        "view": yt.views,
        "length": yt.length,
    }

    yt.streams.get_highest_resolution().download(
        output_path=output_path, filename=f"{vid_id}.mp4"
    )
    return metadata


def extract_frame(video_path, frame_count, output_folder):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Set the frame position
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

    # Read the frame
    ret, frame = video.read()

    # Save the frame as an image
    image_path = os.path.join(output_folder, f"frame{frame_count:04d}.png")
    cv2.imwrite(image_path, frame)

    # Release the video file
    video.release()


def video_to_images(video_path, output_folder, fps=0.5):
    """
    Convert a video to a sequence of images and save them to the output folder.

    Parameters:
    video_path (str): The path to the video file.
    output_folder (str): The path to the folder to save the images to.
    fps (float): Frames per second to extract from the video (default is 1).

    """
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get the total number of frames
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the frame indices to extract based on the desired fps
    frame_indices = [
        int(frame_number)
        for frame_number in range(
            0, total_frames, int(video.get(cv2.CAP_PROP_FPS) / fps)
        )
    ]

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Initialize thread pool executor
    executor = ThreadPoolExecutor()

    # Extract frames in parallel
    for frame_count in frame_indices:
        print(frame_count)
        executor.submit(extract_frame, video_path, frame_count, output_folder)

    # Shutdown the executor
    executor.shutdown()

    # Release the video file
    video.release()

    return output_folder


def video_to_audio(video_path, output_audio_path):
    """
    Convert a video to audio and save it to the output path.

    Parameters:
    video_path (str): The path to the video file.
    output_audio_path (str): The path to save the audio to.

    """
    clip = VideoFileClip(video_path)
    audio = clip.audio

    audio.write_audiofile(output_audio_path, codec="pcm_s16le")


def process_chunk(chunk_id, start_time, end_time, audio_chunk, recognizer):
    """
    Process an individual audio chunk and save the transcription along with its start and end times to a JSON file.

    Parameters:
    chunk_id (int): The identifier for the audio chunk.
    start_time (int): The start time of the chunk in seconds.
    end_time (int): The end time of the chunk in seconds.
    audio_chunk (AudioData): The chunk of audio data to be transcribed.
    recognizer (Recognizer): The speech recognition recognizer instance.
    """
    try:
        # Recognize the speech in the chunk
        text = recognizer.recognize_whisper_api(audio_chunk)
        # Save the transcription to a JSON file
        with open(f"transcribes/chunk_{chunk_id}.json", "w") as file:
            json.dump(
                {"start": start_time, "end": end_time, "text": text}, file
            )
        print(f"Chunk {chunk_id} processed successfully.")
    except sr.UnknownValueError:
        print(
            f"Chunk {chunk_id}: Speech recognition could not understand the audio."
        )
    except sr.RequestError as e:
        print(f"Chunk {chunk_id}: Could not request results from service; {e}")


def audio_to_text(audio_path):
    """
    Convert audio to text using the SpeechRecognition library by processing the audio in chunks.
    Each chunk is processed in a separate thread, and the results, including start and end times, are saved to individual JSON files.
    These are then merged to get the final transcription in a single JSON file, and individual chunk files are removed.

    Parameters:
    audio_path (str): The path to the audio file.
    """
    recognizer = sr.Recognizer()
    audio = sr.AudioFile(audio_path)
    os.makedirs("transcribes", exist_ok=True)
    chunk_duration = 30  # Duration of each chunk in seconds

    with audio as source, concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        chunk_id = 0
        start_time = 0

        while True:
            try:
                # Record a chunk of the audio data
                audio_chunk = recognizer.record(
                    source, duration=chunk_duration
                )

                if len(audio_chunk.frame_data) == 0:
                    break
                end_time = start_time + chunk_duration
                futures.append(
                    executor.submit(
                        process_chunk,
                        chunk_id,
                        start_time,
                        end_time,
                        audio_chunk,
                        recognizer,
                    )
                )
                start_time += chunk_duration
                chunk_id += 1

            except EOFError:
                # Reached the end of the audio file
                break

        # Wait for all threads to complete
        concurrent.futures.wait(futures)

    # Merge JSON files into a single JSON file
    transcriptions = []
    for i in range(chunk_id):
        try:
            with open(f"transcribes/chunk_{i}.json", "r") as file:
                transcriptions.append(json.load(file))
            # Remove the chunk file after adding its content to the transcriptions list
            os.remove(f"transcribes/chunk_{i}.json")
        except FileNotFoundError:
            print(f"Warning: File transcribes/chunk_{i}.json not found.")

    with open("transcribes/final_transcription.json", "w") as outfile:
        json.dump(transcriptions, outfile)

    return transcriptions
