import os
from dotenv import load_dotenv

from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex

from llama_index.core.prompts import PromptTemplate
from llama_index.core import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

load_dotenv("GOOGLE_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")


def get_index(update_data=False, save_index=True):
    Settings.llm = Gemini(model_name="models/gemini-pro", api_key=google_api_key)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)
    Settings.text_splitter = text_splitter

    PERSIST_DIR = "./rag.index"

    if os.path.exists(PERSIST_DIR) and not update_data:
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
        print(f"Loaded model from {PERSIST_DIR}")
    else:
        documents = SimpleDirectoryReader("./rag_data")
        documents = documents.load_data()
        print("Finished loading data")

        index = VectorStoreIndex.from_documents(documents, transformations=[text_splitter])
        print("Finished creating index")
        if save_index:
            index.storage_context.persist(persist_dir=PERSIST_DIR)
            print(f"Finished saving index to {PERSIST_DIR}")

    return index

def get_query_engine():

    template = """
You are a knowledgeable and precise assistant specialized in question-answering tasks, particularly from Dermatology resources.
Your goal is to provide accurate, concise, and contextually relevant answers based on the given information.

Instructions:

Comprehension and Accuracy: Carefully read and comprehend the provided context from the article to ensure accuracy in your response.
Conciseness: Deliver the answer in no more than three sentences, ensuring it is concise and directly addresses the question.
Truthfulness: If the context does not provide enough information to answer the question, clearly state, "I don't know."
Contextual Relevance: Ensure your answer is well-supported by the retrieved context and does not include any information beyond what is provided.

Remember if no context is provided please say you don't know the answer
Here is the question and context for you to work with:

\nQuestion: {question} \nContext: {context} \nAnswer:"""

#     template = """
# You are an expert dermatologist specializing in skin conditions. 
# Try you best to diagnose patient's skin condition. 
# Answer according to the given context where available.

# \nQuestion: {question} \nContext: {context} \nAnswer:"""

    prompt_tmpl = PromptTemplate(
        template = template,
        template_var_mappings = {"query_str": "question", "context_str": "context"},
    )

    index = get_index()

    # configure retriever
    retriever = VectorIndexRetriever(
        index = index,
        similarity_top_k = 10,
    )

    # configure response synthesizer
    response_synthesizer = get_response_synthesizer()

    # assemble query engine
    query_engine = RetrieverQueryEngine(
        retriever = retriever,
        response_synthesizer = response_synthesizer,
    )

    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": prompt_tmpl}
    )

    return query_engine

def query_response(query_engine, question):
    response = query_engine.query(question)
    return response
