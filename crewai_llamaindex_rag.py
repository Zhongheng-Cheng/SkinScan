import os
import warnings
from crewai import Agent, Task, Crew, Process
from crewai_tools import LlamaIndexTool
from dotenv import load_dotenv
load_dotenv()
from llama_index.llms.groq import Groq
from langchain_openai import ChatOpenAI
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_community.chat_message_histories import ChatMessageHistory

#import warnings
import warnings
warnings.simplefilter('ignore', category=UserWarning)



try:
    # Your code here...
    openai_api_key = os.getenv("OPENAI_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = Groq(model="llama3-70b-8192", api_key=groq_api_key)
    chat_llm = ChatOpenAI(
        openai_api_base="https://api.groq.com/openai/v1",
        openai_api_key=groq_api_key,
        model="llama3-70b-8192",
        temperature=0,
        max_tokens=1000,
    )
    dataset_dir = './Dataset'
    pdf_files = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir) if file.endswith('.pdf')]
    reader = SimpleDirectoryReader(input_files=pdf_files)
    docs = reader.load_data()
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
    query_engine = index.as_query_engine(similarity_top_k=5, llm=llm)
    query_tool = LlamaIndexTool.from_query_engine(
        query_engine,
        name="Dermatology_PDF_search",
        description="Use this tool to lookup the information on skin diseases and treatments",
    )
    dermatology_dir_agent = Agent(
        role="Dermatologist",
        goal="Search through the directory to find relevant answers",
        backstory="You are a dermatologist specializing in providing information and advice on skin conditions, treatments, and skincare practices.",
        tools=[query_tool],
        llm=chat_llm
    )
    dir_answering_task = Task(
        description=("Answer dermatology-related user queries with Directory searching. User's question: {question}"
                     "Provide a concise response in 1 paragraph, with bullets for each item."
                     "If the response is a list, include a brief definition for each item."),
        expected_output=("A detailed and accurate response based on your knowledge and search results."),
        agent=dermatology_dir_agent,
        human_input=False
    )
    crew = Crew(
        agents=[dermatology_dir_agent],
        tasks=[dir_answering_task],
        process=Process.sequential,
        human_input=False
    )
    user_question = input("Please enter your question: ")
    if __name__ == "__main__":
    
    #question = "What causes Eczema?"
        inputs = {"question": user_question}
    
    results = crew.kickoff(inputs=inputs)
    print(results)
except Exception as e:
    pass


