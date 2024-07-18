import google.generativeai as genai
import json
import os
from dotenv import load_dotenv
load_dotenv("GOOGLE_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

user_info = {
    "age": 30,
    "gender": "Female",
    "location": "California",
    "race": "Asian"
}
    
prompt_user_info = f"""\
Consider the following patient information:
- Age: {user_info['age']}
- Gender: {user_info['gender']}
- Location: {user_info['location']}
- Race: {user_info['race']}"""

prompt_json_output = """\
Using this JSON schema:
    SkinCondition = {
        "condition_name": str
        "symptoms": str
        "description": str
        "severity": str
        "common_treatments": str
        "recommendations": str
    }
Return a `SkinCondition`"""

prompt_image = f"""\
You are an expert dermatologist specializing in skin conditions. Analyze the image provided and come up with a possible diagnosis and a treatment plan. 
{prompt_user_info}
Provide the analysis in detailed paragraphs and include bullet points where necessary.
{prompt_json_output}
"""

prompt_audio = f"""\
You are a conversational bot collecting demographic information from a patient. Summarize the content, and recommend the appropriate treatment. Additionally, note if the patient should call a dermatologist. 
{prompt_user_info}
"""

def summarize_audio(prompt, audio_file_path) -> dict:
    model = genai.GenerativeModel("models/gemini-1.5-pro-latest", generation_config={"response_mime_type": "application/json"})
    audio_file = genai.upload_file(path=audio_file_path)
    prompt = prompt
    response = model.generate_content([prompt, audio_file])
    return json.loads(response.text)

if __name__ == "__main__":
    result = summarize_audio(prompt_image, "./static/example_uploads/img3.jpeg")
    for key, value in result.items():
        print(f"{key}: {value}")

    