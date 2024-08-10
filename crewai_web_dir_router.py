#pip install llama-index-core
import os
import warnings
warnings.filterwarnings('ignore')
from crewai import Agent, Task, Crew
from crewai_tools import (
    SerperDevTool,
    ScrapeWebsiteTool,
    DirectorySearchTool
)
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()


google_api_key = os.getenv("GOOGLE_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
#pdf_search_api_key = os.getenv("PDF_SEARCH_API_KEY")

assert google_api_key, "No google api key"
assert serper_api_key, "No serper api key"
assert openai_api_key, "No openai api key"
#assert pdf_search_api_key, "No pdf search api key"

# Initialize Gemini model
gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    verbose=False,
    temperature=0.5,
    google_api_key=google_api_key
)

search_tool = SerperDevTool(
    n_results=2,
    api_key=serper_api_key)
scrape_tool = ScrapeWebsiteTool()

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.gemini import Gemini
from crewai_tools import LlamaIndexTool

# Initialize the query tool
dataset_dir = './Dataset'
# Get a list of PDF files in the directory
pdf_files = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir) if file.endswith('.pdf')]
#print(pdf_files)
#reader = SimpleDirectoryReader(input_files=['./Dataset/*.pdf'])  # Replace with your PDF directory
#reader = SimpleDirectoryReader(input_files=[os.listdir(dataset_dir)])
reader = SimpleDirectoryReader(input_files=pdf_files)
docs = reader.load_data()
llm = Gemini(model_name="models/gemini-pro")
index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine(similarity_top_k=5, llm=llm)
query_tool = LlamaIndexTool.from_query_engine(
    query_engine,
    name="Dermatology_PDF_search",
    description="Use this tool to lookup the information on skin diseases and treatments",
)

# Integrate query tool with Dermatologist_dir_agent
dermatology_dir_agent = Agent(
    role="Dermatologist",
    goal="Search through the directory to find relevant answers",
    backstory="You are a dermatologist specializing in providing information and advice on skin conditions, treatments, and skincare practices.",
    llm=gemini,
    tools=[query_tool],  # Add query_tool to the agent's tools
    verbose=False,
    human_input=False,
    allow_delegation=False
)

dermatology_web_agent = Agent(
    role="Dermatologist",
    goal="Search the Internet when it is necessary for answering specific questions",
    backstory="You are a dermatologist specializing in providing information and advice on skin conditions, treatments, and skincare practices.",
    llm=gemini,
    tools=[search_tool, scrape_tool],
    #tools=[pdf_search_tool],
    verbose=False,
    human_input=False,
    allow_delegation=False
)


# Define the router tool
class RouterTool:
    def __init__(self):
        self.conditions = {
            "web": lambda x: "internet" in x or "online" in x,
            "directory": lambda x: "pdf" in x or "file" in x
        }

    def route(self, question):
        for route, condition in self.conditions.items():
            if condition(question):
                return route
        return "directory"  # Default route

# Initialize the router tool
router_tool = RouterTool()

# Define the router agent and task
router_agent = Agent(
    role="Router",
    goal="Route the question to the appropriate agent",
    backstory="You are a router agent specializing in directing questions to the right source.",
    llm=gemini,
    tools=[router_tool],
    verbose=False,
    human_input=False,
    allow_delegation=False
)

router_task = Task(
    description="Route the question to either the web or directory agent",
    expected_output="The routed question",
    agent=router_agent,
    llm=gemini,
    verbose=False,
    human_input=False
)

dir_answering_task = Task(
    description=("Answer dermatology-related user queries with Directory searching. User's question: {question}"
                 "Provide a concise response in 1 paragraph, with bullets for each item."
                 "If the response is a list, include a brief definition for each item."),
    expected_output=("A detailed and accurate response based on your knowledge and search results."),
    agent=dermatology_dir_agent,
    #llm=gemini,
    verbose=False,
    #tools=[pdf_search_tool],
    human_input=False
)

web_answering_task = Task(
    description = (
        "Answer dermatology-related user queries with web searching. User's question: {question}"
        "Provide a concise response in 1 paragraph, with bullets for each item. "
        "If the response is a list, include a brief definition for each item. "
    ),
    expected_output = (
        "A detailed and accurate response based on your knowledge and search results."
    ),
    agent = dermatology_web_agent,
    #llm = gemini,
    verbose = False,
    human_input=False
)

from crewai import Process
# Update the crew import  with the router agent and task
crew = Crew(
    agents=[router_agent, dermatology_dir_agent, dermatology_web_agent],
    tasks=[router_task, dir_answering_task, web_answering_task],
    verbose=False,
    process=Process.sequential,
    human_input=False
)

# Call the router tool to route the question
user_question = input("Please enter your question: ")
if __name__ == "__main__":
    inputs = {"question": user_question}
    route = router_tool.route(user_question)
    if route == "web":
        inputs["agent"] = dermatology_web_agent
    else:
        inputs["agent"] = dermatology_dir_agent
    results = crew.kickoff(inputs=inputs)
    print(results)


