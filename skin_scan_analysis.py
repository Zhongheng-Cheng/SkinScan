import streamlit as st
from dotenv import load_dotenv
import os
import tempfile
import cv2
import moviepy.editor as mp
from pydantic import BaseModel
from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.core.program import MultiModalLLMCompletionProgram
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core import SimpleDirectoryReader
from google.api_core import retry
from google.api_core.exceptions import ResourceExhausted
import google.generativeai as genai

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")


# Define the SkinCondition model
class SkinCondition(BaseModel):
    condition_name: str
    symptoms: str
    description: str
    severity: str
    common_treatments: str
    recommendations: str

    def __str__(self):
        attributes = vars(self)
        return '<br>'.join(f"> {key.replace('_', ' ').title()}: {value}" for key, value in attributes.items())

# Define the function to generate a response using the Gemini model
@retry.Retry(deadline=120, initial=1.0, multiplier=1.3, maximum=60.0, predicate=retry.if_exception_type(ResourceExhausted))
def pydantic_gemini(model_name, output_class, image_documents, prompt_template_str):
    gemini_llm = GeminiMultiModal(model_name=model_name)
    llm_program = MultiModalLLMCompletionProgram.from_defaults(
        output_parser=PydanticOutputParser(output_class),
        image_documents=image_documents,
        prompt_template_str=prompt_template_str,
        multi_modal_llm=gemini_llm,
        verbose=True,
    )
    response = llm_program()
    return response

def generate_img_response(img_path, user_info):
    documents = SimpleDirectoryReader(input_files=[img_path])
    documents = documents.load_data()

    prompt_template_str = f"""\
You are an expert dermatologist specializing in skin conditions. Analyze the image provided and come up with a possible diagnosis and a treatment plan. 
Consider the following patient information:
- Age: {user_info['age']}
- Gender: {user_info['gender']}
- Location: {user_info['location']}
- Race: {user_info['race']}
Provide the analysis in detailed paragraphs and include bullet points where necessary.
"""

    pydantic_response = pydantic_gemini(
        "models/gemini-1.5-pro",
        SkinCondition,
        documents,
        prompt_template_str,
    )
    
    return pydantic_response

# Define the function to summarize audio
@retry.Retry(deadline=120, initial=1.0, multiplier=1.3, maximum=60.0, predicate=retry.if_exception_type(ResourceExhausted))
def summarize_audio(audio_file_path, user_info):
    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
    audio_file = genai.upload_file(path=audio_file_path)
    prompt = f"""\
You are a conversational bot collecting demographic information from a patient. Summarize the content, and recommend the appropriate treatment. Additionally, note if the patient should call a dermatologist. 
Consider the following patient information:
- Age: {user_info['age']}
- Gender: {user_info['gender']}
- Location: {user_info['location']}
- Race: {user_info['race']}
"""
    response = model.generate_content([prompt, audio_file])
    return response.text

# Define the function to save the uploaded file
def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error handling uploaded file: {e}")
        return None

# Function to check if two images are similar
def are_images_similar(img1_path, img2_path, threshold=0.95):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    if img1.shape != img2.shape:
        return False
    diff = cv2.absdiff(img1, img2)
    non_zero_count = cv2.countNonZero(diff)
    similarity = 1 - non_zero_count / (img1.shape[0] * img1.shape[1])
    return similarity >= threshold

# Define the function to extract keyframes from video
def extract_keyframes(video_path, num_frames=5):
    video = mp.VideoFileClip(video_path)
    duration = video.duration
    frame_times = [i * duration / num_frames for i in range(num_frames)]
    
    keyframes = []
    for time in frame_times:
        frame = video.get_frame(time)
        frame_path = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg').name
        cv2.imwrite(frame_path, frame)
        keyframes.append(frame_path)
    
    # Filter out similar frames
    unique_keyframes = [keyframes[0]]
    for i in range(1, len(keyframes)):
        if not are_images_similar(unique_keyframes[-1], keyframes[i]):
            unique_keyframes.append(keyframes[i])
    
    return unique_keyframes

#Define the function to analyze video - Only print unique responses
def analyze_video(video_path, user_info):
    keyframes = extract_keyframes(video_path)
    responses = []
    unique_conditions = set()
    for frame in keyframes:
        response = generate_img_response(frame, user_info)
        condition_name = response.condition_name
        if condition_name not in unique_conditions:
            responses.append(response)
            unique_conditions.add(condition_name)
    return responses
  
# Streamlit app
st.title("Skin Condition Analyzer")
st.write("Upload an image, audio, or video file related to the skin condition, and the AI will analyze it to provide a possible diagnosis and treatment plan.")

# Collect user information
user_age = st.text_input("Enter your age:")
user_gender = st.text_input("Enter your gender:")
user_location = st.text_input("Enter your location:")
user_race = st.text_input("Enter your race:")

user_info = {
    "age": user_age,
    "gender": user_gender,
    "location": user_location,
    "race": user_race
}

# Handle image uploads
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    temp_image_path = save_uploaded_file(uploaded_image)
    if temp_image_path:
        response = generate_img_response(temp_image_path, user_info)
        st.write("### Analysis Result")
        st.write(response)
        os.remove(temp_image_path)

# Handle audio uploads and prompt for additional media if necessary
uploaded_audio = st.file_uploader("Choose an audio file...", type=["mp3", "wav", "m4a"])
if uploaded_audio is not None:
    temp_audio_path = save_uploaded_file(uploaded_audio)
    if temp_audio_path:
        response = summarize_audio(temp_audio_path, user_info)
        st.write("### Audio Summary")
        st.write(response)
        os.remove(temp_audio_path)

    # Prompt for additional media input
    st.write("Please provide an image or video for a more comprehensive analysis.")
    additional_image = st.file_uploader("Choose an additional image...", type=["jpg", "jpeg", "png"])
    if additional_image is not None:
        temp_image_path = save_uploaded_file(additional_image)
        if temp_image_path:
            response = generate_img_response(temp_image_path, user_info)
            st.write("### Additional Image Analysis Result")
            st.write(response)
            os.remove(temp_image_path)

    additional_video = st.file_uploader("Choose an additional video...", type=["mp4", "mov", "avi"])
    if additional_video is not None:
        temp_video_path = save_uploaded_file(additional_video)
        if temp_video_path:
            responses = analyze_video(temp_video_path, user_info)
            st.write("### Additional Video Analysis Result")
            for i, response in enumerate(responses, start=1):
                st.write(f"#### Keyframe {i}")
                st.write(response)
            os.remove(temp_video_path)

# Handle video uploads
uploaded_video = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi"])
if uploaded_video is not None:
    temp_video_path = save_uploaded_file(uploaded_video)
    if temp_video_path:
        responses = analyze_video(temp_video_path, user_info)
        st.write("### Video Analysis Result")
        for i, response in enumerate(responses, start=1):
            st.write(f"#### Keyframe {i}")
            st.write(response)
        os.remove(temp_video_path)
