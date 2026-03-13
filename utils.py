# --- Few-Shot Inference: 6 BSL examples from tests/samples + 1 query from tests/hello_test.mp4 ---
import glob, json, os, base64
import cv2
import numpy as np
from VideoFTTools import VideoExtractor


def resize_base64_frame(frame_base64, resize_factor):
    """Decode a base64 JPEG frame, resize it, and return the new base64 string."""
    img_bytes = base64.b64decode(frame_base64)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    img_resized = cv2.resize(img, (w // resize_factor, h // resize_factor))
    _, buffer = cv2.imencode('.jpg', img_resized)
    return base64.b64encode(buffer).decode('utf-8')

def system_message_for_few_shot_examples(samples_path, aoai_model_name, no_frames, resize=2):

    # 1. Load all sample videos from tests/samples and derive labels from filenames
    sample_videos = sorted(glob.glob(os.path.join(samples_path, "*.mp4")))
    # Derive label from filename: replace "_" with " "
    sample_items = []
    for video_path in sample_videos:
        filename = os.path.splitext(os.path.basename(video_path))[0]
        label = filename.replace("_", " ")
        sample_items.append({"label": label, "video_path": video_path})

    few_shot_n = len(sample_items)
    print(f"Found {few_shot_n} few-shot example videos:")
    for item in sample_items:
        print(f"  {item['label']} → {item['video_path']}")

    # 2. Extract frames for each example and for the query video
    example_frames = []
    for item in sample_items:
        extractor = VideoExtractor(item["video_path"])
        frames = extractor.extract_n_video_frames(n=no_frames)
        #frames = extractor.extract_video_frames(interval_seconds)
        # Resize frames to reduce token consumption
        if resize > 1:
            for f in frames:
                f["frame_base64"] = resize_base64_frame(f["frame_base64"], resize)
        example_frames.append({"label": item["label"], "frames": frames})
        #print(f"  Example: {item['label']} → {item['video_path']}")

    # 3. Build the few-shot messages
    all_labels = [item["label"] for item in sample_items]
    all_labels_str = ", ".join(all_labels)

    few_shot_system_message = f"""You are an expert in British Sign Language (BSL) recognition from video.
    You are provided with a series of extracted frames from videos showing a person performing signs in BSL. Each frame includes a timestamp in the lower-left corner, formatted as 'video_time: mm:ss:msec'.

    First, you will see {few_shot_n} labeled examples of video frame sequences, each demonstrating a different BSL concept (word or phrase).
    Then, you will receive a new unlabeled video frame sequence. Your task is to identify the BSL concept being signed in the final sequence, based on the labeled examples.

    The possible BSL concepts are:
    {all_labels_str}

    Respond with a valid JSON object in this exact format:
    {{[
        "concept": "The identified BSL concept from the provided list, spelled exactly as given."
      ]
    }}
    """

    if aoai_model_name.startswith("gpt-5"):
        system_messages = [{"role": "developer", "content": few_shot_system_message}]
    else:
        system_messages = [{"role": "system", "content": few_shot_system_message}]

    # Add few-shot examples as user/assistant turns
    for i, example in enumerate(example_frames):
        label = example["label"]
        frames_b64 = [f["frame_base64"] for f in example["frames"]]

        # User turn: show frames with label context
        system_messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": f"Example {i+1}: Here are the frames from a video of a person signing in BSL. What concept is being signed?"},
                *[{"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{b64}", "detail": "low"}} for b64 in frames_b64],
            ]
        })
        # Assistant turn: ground truth label
        system_messages.append({
            "role": "assistant",
            "content": json.dumps({"concept": label})
        })
    
    return system_messages