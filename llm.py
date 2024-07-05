from dotenv import load_dotenv
load_dotenv("GOOGLE_API_KEY")

from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.core.multi_modal_llms.generic_utils import load_image_urls
from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.core.program import MultiModalLLMCompletionProgram
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.schema import TextNode
from llama_index.core import SimpleDirectoryReader
from pydantic import BaseModel

# #gemini_pro = GeminiMultiModal(model_name="models/gemini-pro-vision")
# gemini_pro = GeminiMultiModal(model_name="models/gemini-1.5-pro")

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

def pydantic_gemini(
    model_name, output_class, image_documents, prompt_template_str
):
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

    # prompt_template_str = #"""\
    # You are an AI assistant specializing in dermatology. Your task is to analyze the provided skin condition images and associated metadata to offer a summary and medical advice. \
    # Your responses must be coherent, honest, and formatted as JSON.
    # """
    prompt_template_str = """\
You are an expert dermatologist specializing in skin conditions. Your job is to analyze the provided skin condition images and come up with a possible diagnosis anda treatment plan.
"""

    pydantic_response = pydantic_gemini(
        "models/gemini-1.5-pro",
        SkinCondition,
        documents,
        prompt_template_str,
    )
    
    return str(pydantic_response)

if __name__ == "__main__":
    print(generate_img_response("static/example_images/img_1.jpeg"))