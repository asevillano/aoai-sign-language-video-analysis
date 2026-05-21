# Import libraries
import streamlit as st
import cv2
import os
import time
import json
from dotenv import load_dotenv
from openai import AzureOpenAI
import base64
import subprocess
import yt_dlp
from yt_dlp.utils import download_range_func
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from utils import *

# Default configuration
SEGMENT_DURATION = 10 # In seconds, Set to 0 to not split the video
DEFAULT_TEMPERATURE = 0.5
RESIZE_OF_FRAMES = 2
SECONDS_PER_FRAME = 1

#SYSTEM_PROMPT = "You are a helpful assistant that describes in detail a video."
#USER_PROMPT = "These are the frames from the video."
USER_PROMPT = "Now identify the BSL concept or concepts (could be more than one) being signed in this new video:"

# Load configuration
load_dotenv(override=True)

# Initialize OpenAI client with Entra ID authentication
credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(
    credential, "https://cognitiveservices.azure.com/.default"
)
aoai_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_ad_token_provider=token_provider,
    api_version="2025-04-01-preview"
)
aoai_model_name = os.getenv("AOAI_MODEL_NAME", "gpt-4.1")

@st.cache_data(show_spinner="Loading few-shot examples...")
def load_few_shot_messages(samples_path, model_name, no_frames, resize):
    """Load few-shot messages only once and cache them across Streamlit reruns."""
    messages = system_message_for_few_shot_examples(samples_path, model_name, no_frames, resize)
    # Extract text-only system prompt for sidebar display
    text = ""
    for msg in messages:
        if msg["role"] in ("system", "developer"):
            text = msg["content"]
            break
    return messages, text

SYSTEM_PROMPT, SYSTEM_PROMPT_TEXT = load_few_shot_messages(
    os.path.join("tests", "samples-app"), aoai_model_name, no_frames=5, resize=RESIZE_OF_FRAMES
)

# Function to encode a local video into frames
def process_video(video_path, seconds_per_frame=SECONDS_PER_FRAME, resize=RESIZE_OF_FRAMES, output_dir='', temperature = DEFAULT_TEMPERATURE):
    base64Frames = []

    # Prepare the video analysis
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(fps * seconds_per_frame)
    curr_frame=0

    # Prepare to write the frames to disk
    if output_dir != '': # if we want to write the frame to disk
        output_dir = 'frames'
        os.makedirs(output_dir, exist_ok=True)
        frame_count = 1

    # Loop through the video and extract frames at specified sampling rate
    while curr_frame < total_frames - 1:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break

        # Resize the frame to save tokens and get faster answer from the model. If resize==0 don't resize
        if resize != 0:
            height, width, _ = frame.shape
            frame = cv2.resize(frame, (width // resize, height // resize))

        _, buffer = cv2.imencode(".jpg", frame)

        # Save frame as JPG file
        if output_dir != '': # if we want to write the frame to disk
            frame_filename = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_frame_{frame_count}.jpg")
            print(f'Saving frame {frame_filename}')
            with open(frame_filename, "wb") as f:
                f.write(buffer)            
            frame_count += 1

        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip
    video.release()
    print(f"Extracted {len(base64Frames)} frames")
    
    return base64Frames

# Function to analyze the video
def analyze_video(base64frames, few_shot_messages, user_prompt, temperature):

    #print(f'FEW-SHOT MESSAGES: {len(few_shot_messages)} messages')
    #print(f'USER PROMPT:   [{user_prompt}]')

    try:
        # Build messages: few-shot messages + query message with frames
        messages = list(few_shot_messages)
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                *[{"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "auto"}} for x in base64frames],
            ]
        })

        response = aoai_client.chat.completions.create(
            model=aoai_model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=4096
        )

        json_response = json.loads(response.model_dump_json())
        #print(f'RESPONSE: [{response.model_dump_json(indent=2)}]')
        response = json_response['choices'][0]['message']['content']

    except Exception as ex:
        print(f'ERROR: {ex}')
        response = f'ERROR: {ex}'

    return response

# Split the video in segments of N seconds (by default 50 seconds). If segment_length is 0 the full video is processed
def split_video(video_path, output_dir, segment_length=50):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()  # Release the video file handle
    
    if segment_length == 0: # Do not split
        segment_length = int(duration)

    for start_time in range(0, int(duration), segment_length):
        end_time = min(start_time + segment_length, duration)
        output_file = os.path.join(output_dir, f'{os.path.splitext(os.path.basename(video_path))[0]}_segment_{start_time}-{end_time}_secs.mp4')
        # Use ffmpeg with re-encoding for precise cuts (ffmpeg_extract_subclip uses -c copy
        # which only cuts at keyframes, causing short or empty segments)
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-i", video_path,
            "-t", str(end_time - start_time),
            "-c:v", "libx264", "-preset", "fast",
            "-c:a", "aac",
            "-avoid_negative_ts", "make_zero",
            output_file
        ]
        print(f"Splitting segment: {start_time}s - {end_time}s -> {output_file}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ffmpeg error: {result.stderr}")
        yield output_file

# Process the video
def execute_video_processing(st, segment_path, few_shot_messages, user_prompt, temperature):
    # Show the video on the screen
    st.divider()
    st.write(f"Video: {segment_path}:")
    st.video(segment_path)

    with st.spinner(f"Analyzing video segment: {segment_path}"):
        # Extract 1 frame per second. Adjust the `seconds_per_frame` parameter to change the sampling rate
        with st.spinner(f"Extracting frames..."):
            inicio = time.time()
            if save_frames:
                output_dir = 'frames'
            else:
                output_dir = ''
            base64frames = process_video(segment_path, seconds_per_frame=seconds_per_frame, resize=resize, output_dir=output_dir, temperature=temperature)
            fin = time.time()
            print(f'\t>>>> Frames extraction took {(fin - inicio):.3f} seconds <<<<')
            ##st.write(f'Extracted {len(base64frames)} frames in {(fin - inicio):.3f} seconds')

        # Analyze the video frames
        with st.spinner(f'Analyzing frames with {aoai_model_name}...'):
            inicio = time.time()
            analysis = analyze_video(base64frames, few_shot_messages, user_prompt, temperature)
            fin = time.time()
        print(f'\t>>>> Analysys with {aoai_model_name} took {(fin - inicio):.3f} seconds <<<<')

    ### st.write(f"**Analysis of segment {segment_path}** ({(fin - inicio):.3f} seconds)")
    fin = time.time()
    print(f'\t>>>> {(fin - inicio):.6f} segundos <<<<')
    st.success("Analysis completed.")

    return analysis

# Streamlit User Interface
st.set_page_config(
    page_title="Video Analysis with Azure OpenAI",
    layout="centered",
    initial_sidebar_state="auto",
)
st.image("microsoft.png", width=100)
st.title('Video Analysis with Azure OpenAI')

with st.sidebar:
    file_or_url = st.selectbox("Video source:", ["File", "URL"], index=0, help="Select the source, file or url")
    # file_or_url = "File"
    initial_split = SEGMENT_DURATION
    if file_or_url == "URL":
        continuous_transmision = st.checkbox('Continuous transmision', False, help="Video of a continuous transmision")
        if continuous_transmision:
            initial_split = SEGMENT_DURATION
        
    seconds_split = st.number_input('Number of seconds to split the video', initial_split, help="The video will be processed in smaller segments based on the number of seconds specified in this field. (0 to not split)")
    seconds_per_frame = float(st.text_input('Number of seconds per frame', SECONDS_PER_FRAME, help="The frames will be extracted every number of seconds specified in the field. It can be a decimal number, like 0.5, to extract a frame every half of second."))
    resize = st.number_input("Frames resizing ratio", RESIZE_OF_FRAMES, help="The size of the images will be reduced in proportion to this number while maintaining the height/width ratio. This reduction is useful for improving latency and reducing token consumption (0 to not resize)")
    save_frames = st.checkbox('Save the frames to the folder "frames"', False)
    temperature = float(st.number_input('Temperature for the model', DEFAULT_TEMPERATURE))
    system_prompt_text = st.text_area('System Prompt', SYSTEM_PROMPT_TEXT)
    user_prompt = st.text_area('User Prompt', USER_PROMPT)

# Build the few-shot messages for this run (update system text if user edited it)
few_shot_messages = list(SYSTEM_PROMPT)
if system_prompt_text != SYSTEM_PROMPT_TEXT:
    for _i, _msg in enumerate(few_shot_messages):
        if _msg["role"] in ("system", "developer"):
            few_shot_messages[_i] = {**_msg, "content": system_prompt_text}
            break

# Prepare the segment directory
output_dir = "segments"
os.makedirs(output_dir, exist_ok=True)

# Video file or Video URL
if file_or_url == 'File':
    video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
else:
    url = st.text_area("Enter de url:", value='https://www.youtube.com/watch?v=Y6kHpAeIr4c', height=10)

# Analyze the video when the button is pressed
if st.button("Analyze video", use_container_width=True, type='primary'):

    # Show parameters:
    print(f"PARAMETERS:")
    print(f"file_or_url: {file_or_url}, seconds to split: {seconds_split}")
    print(f"seconds_per_frame: {seconds_per_frame}, resize ratio: {resize}, save_frames: {save_frames}, temperature: {temperature}")

    if file_or_url == 'URL': # Process Youtube video
        st.write(f'Analyzing video from url {url}...')
        
        ydl_opts = {
                #'format': 'best',
                'format': '(bestvideo[vcodec^=av01]/bestvideo[vcodec^=vp9]/bestvideo)+bestaudio/best',
                'outtmpl': 'segment_%(start)s.mp4',
                'force_keyframes_at_cuts': True,
        }
        ydl = yt_dlp.YoutubeDL(ydl_opts)
        if continuous_transmision == False:
            info_dict = ydl.extract_info(url, download=False)
            video_duration = info_dict.get('duration', 0)

            if seconds_split == 0:
                duracion_segmento=video_duration
            else:
                duracion_segmento=seconds_split #SEGMENT_DURATION
        else:
            video_duration = 48*60*60
        
            if seconds_split == 0:
                duracion_segmento=180 # 3 minutes
            else:
                duracion_segmento=seconds_split #SEGMENT_DURATION
        
        for start in range(0, video_duration, duracion_segmento):
            end = start + duracion_segmento
            filename = f'segments/segment_{start}-{end}.mp4'
            with st.spinner(f"Downloading video from second {start} to {end}..."):
                ydl_opts['outtmpl']['default'] = filename
                ydl_opts['download_ranges'] = download_range_func(None, [(start, end)])

                print(f'start: {start}, video_duration: {video_duration}, duracion_segmento: {duracion_segmento}')
                try:
                    ydl.download([url])
                except:
                    break

            if os.path.exists(filename): # ext .mp4
                segment_path = filename
            else:
                segment_path = filename + '.mkv'
                if not os.path.exists(segment_path):
                    segment_path = filename + '.webm'

            print(f"Segment downloaded: {segment_path}")

            # Process the video segment
            analysis = execute_video_processing(st, segment_path, few_shot_messages, user_prompt, temperature)
            st.markdown(f"**Description**: {analysis}", unsafe_allow_html=True)
            #st.write(f"{analysis}")
            
            # Delete the video segment
            os.remove(segment_path)

    else: # Process the fideo file
        if video_file is None:
            st.error("Please upload a video file to analyze.")
        else:
            os.makedirs("temp", exist_ok=True)
            video_path = os.path.join("temp", video_file.name)
            try:
                with open(video_path, "wb") as f:
                    f.write(video_file.getbuffer())

                # Splitting video in segment of N seconds (if seconds is 0 t will not split the video)
                for segment_path in split_video(video_path, output_dir, seconds_split):
                    # Process the video segment
                    analysis = execute_video_processing(st, segment_path, few_shot_messages, user_prompt, temperature)
                    st.write(f"{analysis}")

                    # Delete the video segment
                    #os.remove(segment_path)

            except Exception as ex:
                print(f'ERROR: {ex}')
                st.write(f'ERROR: {ex}')
