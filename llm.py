from dotenv import load_dotenv
load_dotenv("GOOGLE_API_KEY")
import os, shutil

from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.core.multi_modal_llms.generic_utils import load_image_urls
from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.core.program import MultiModalLLMCompletionProgram
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.schema import TextNode
from llama_index.core import SimpleDirectoryReader
from pydantic import BaseModel
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import Settings
from llama_index.core import StorageContext
import qdrant_client
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini

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
    
    return pydantic_response


def generate_query_engine(pydantic_response):
    text_node = TextNode()
    metadata = {}
    for r in pydantic_response:
        if r[0] == "description":
            text_node.text = r[1]
        else:
            metadata[r[0]] = r[1]
    text_node.metadata = metadata
    nodes = [text_node]

    storage_path = "qdrant_storage"

    # Create a new storage folder when creating a query engine
    if os.path.exists(storage_path):
        shutil.rmtree(storage_path)

    # Initialize Qdrant client with a new storage path
    client = qdrant_client.QdrantClient(path="qdrant_storage")
    vector_store = QdrantVectorStore(client=client, collection_name="collection")

    #client = qdrant_client.QdrantClient(path="qdrant_gemini_3")
    #vector_store = QdrantVectorStore(client=client, collection_name="collection")

    llm = Gemini(
        model="models/gemini-1.5-pro",
        temperature=0.5,
        max_output_tokens=512,
    )
    embed_model = GeminiEmbedding(
        model_name="models/embedding-001"
    )
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        service_context=service_context,
    )

    query_engine = index.as_query_engine(
        similarity_top_k=1,
    )

    return query_engine

def generate_text_response(query_engine, prompt):
    response = str(query_engine.query(prompt))
    return response

if __name__ == "__main__":
    result = generate_img_response("static/example_images/img_1.jpeg")
    print(str(result))
    query_engine = generate_query_engine(result)
    text_response = generate_text_response(query_engine, "What is the diagnosis and treatment")
    print("===")
    print(text_response)