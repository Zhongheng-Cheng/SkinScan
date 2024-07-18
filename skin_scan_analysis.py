import streamlit as st
from dotenv import load_dotenv
import os
import shutil
import tempfile
import cv2
import moviepy.editor as mp
from pydub import AudioSegment
from pydantic import BaseModel
from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.core.multi_modal_llms.generic_utils import load_image_urls
from llama_index.core.program import MultiModalLLMCompletionProgram
from llama_index.core.output_parsers import PydanticOutputParser
#from llama_index.core.schema import TextNode
from llama_index.core import SimpleDirectoryReader
#from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import Settings
from llama_index.core import StorageContext
#import qdrant_client
#from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
#from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
#import genai
import google.generativeai as genai
from google.api_core import retry
from google.api_core.exceptions import ResourceExhausted


load_dotenv("GOOGLE_API_KEY")

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

def generate_img_response(img_path):
    documents = SimpleDirectoryReader(input_files=[img_path])
    documents = documents.load_data()

    prompt_template_str = """\
You are an expert dermatologist specializing in skin conditions. Your job is to analyze the image provided and come up with a possible diagnosis and a treatment plan. 
Describe the physical features of the skin condition in the affected body part in detail. List each feature in bullet points."""

    pydantic_response = pydantic_gemini(
        "models/gemini-1.5-pro",
        SkinCondition,
        documents,
        prompt_template_str,
    )
    
    return pydantic_response

# Define the function to summarize audio
@retry.Retry(deadline=120, initial=1.0, multiplier=1.3, maximum=60.0, predicate=retry.if_exception_type(ResourceExhausted))
def summarize_audio(audio_file_path):
    """Summarize the audio using Google's Generative API."""
    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
    audio_file = genai.upload_file(path=audio_file_path)
    response = model.generate_content(
        [
            "You are a conversational bot collecting demographic information from a patient. Summarize the content, and recommend the appropriate treatment. Additionally, note if the patient should call a dermatologist.",
            audio_file
        ]
    )
    return response.text

# Define the function to save the uploaded file
def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary file and return the path."""
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
def extract_keyframes(video_path, num_frames=1):
    """Extract keyframes from a video file."""
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

# Define the function to analyze video
def analyze_video(video_path):
    keyframes = extract_keyframes(video_path)
    responses = [generate_img_response(frame) for frame in keyframes]
    return responses


# Streamlit app
st.title("Skin Condition Analyzer")
st.write("Upload an image, audio, or video file related to the skin condition, and the AI will analyze it to provide a possible diagnosis and treatment plan.")

# Handle image uploads
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    # Save the uploaded file to a temporary directory
    temp_image_path = save_uploaded_file(uploaded_image)
    if temp_image_path:
        # Generate response using the Gemini model
        response = generate_img_response(temp_image_path)
        # Display the response
        st.write("### Analysis Result")
        st.write(response)
        # Clean up the temporary file
        os.remove(temp_image_path)

# Handle audio uploads
uploaded_audio = st.file_uploader("Choose an audio file...", type=["mp3", "wav", "m4a"])
if uploaded_audio is not None:
    # Save the uploaded file to a temporary directory
    temp_audio_path = save_uploaded_file(uploaded_audio)
    if temp_audio_path:
        # Summarize the audio
        response = summarize_audio(temp_audio_path)
        # Display the response
        st.write("### Audio Summary")
        st.write(response)
        # Clean up the temporary file
        os.remove(temp_audio_path)

# Handle video uploads
uploaded_video = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi"])
if uploaded_video is not None:
    # Save the uploaded file to a temporary directory
    temp_video_path = save_uploaded_file(uploaded_video)
    if temp_video_path:
        # Analyze the video
        responses = analyze_video(temp_video_path)
        # Display the responses
        st.write("### Video Analysis Result")
        for i, response in enumerate(responses, start=1):
            st.write(f"#### Keyframe {i}")
            st.write(response)
        # Clean up the temporary files
        os.remove(temp_video_path)
