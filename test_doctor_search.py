import os
from dotenv import load_dotenv
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
import google.generativeai as genai
genai.configure(api_key=google_api_key)
from typing import List, Dict
import csv

# Toy function to return doctor information
def find_doctor(location: str) -> Dict:
    """Finds the nearest doctor"""
    with open('doctor_information.csv', newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)
        print("Header:", header)
        
        for row in csvreader:
            print("Row:", row)
    return {
        "name": "John Doe",
        "specialty": "Dermatology"
    }

def recommend_product(keywords: List[str]) -> Dict:
    """Recommend a product based on the keywords for facial issues"""
    return {
        "name": "Sample Product",
        "Description": "Good for treating dermatology issues on face.",
        "Price": 10
    }

model = genai.GenerativeModel(model_name="gemini-1.5-flash", tools=[find_doctor, recommend_product])
chat = model.start_chat()


# prompt = "Hi, how are you doing?"
# prompt = "I have some problems with my face"
# prompt = "I am in New York, which doctor can I turn to right now?"
# prompt = "How can I treat this problem on my face?"
# prompt = "Can you recommend some product for me to use?"

def chat_once(prompt):
    response = chat.send_message(prompt)

    # if Gemini is calling the function
    # it will return the function name and args instead of calling it directly
    func_outputs = {}
    for part in response.parts:
        if text := part.text:
            print(text)
        elif fn := part.function_call:
            args = ", ".join(f"{key}={val}" if type(val) != str else f"{key}='{val}'" for key, val in fn.args.items())

            # manually call the function
            expression = f"{fn.name}({args})"
            print("[Function calling] " + expression)
            result = eval(expression)

            # storing the function output
            # print(result)
            func_outputs[fn.name] = result

    # feed the functions and outputs back into Gemini to generate text response
    if func_outputs:
        response_parts = [
            genai.protos.Part(function_response=genai.protos.FunctionResponse(name=fn, response={"result": val}))
            for fn, val in func_outputs.items()
        ]

        response = chat.send_message(response_parts)
        print(response.text)

if __name__ == "__main__":
    # while True:
        # prompt = input(">>> ")
        # chat_once(prompt)
    find_doctor("ddd")


# Reference: https://ai.google.dev/gemini-api/docs/function-calling/tutorial?lang=python