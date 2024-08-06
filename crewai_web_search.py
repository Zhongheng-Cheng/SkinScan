import os
import warnings
warnings.filterwarnings('ignore')
from crewai import Agent, Task, Crew
from crewai_tools import (
    SerperDevTool,
    ScrapeWebsiteTool
)
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")
assert google_api_key, "No google api key"
assert serper_api_key, "No serper api key"

# Initialize Gemini model
gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    verbose=False,
    temperature=0.5,
    google_api_key=google_api_key
)

# Instantiate tools
search_tool = SerperDevTool(api_key=serper_api_key)
scrape_tool = ScrapeWebsiteTool()


dermatology_agent = Agent(
    role='Dermatologist',
    goal='Provide accurate and helpful dermatology-related information and advice. Search the Internet when it is necessary for answering specific questions',
    backstory="""You are a dermatologist specializing in providing information and advice on skin conditions,
    treatments, and skincare practices. Your goal is to assist users with their dermatology-related queries
    by leveraging both search results and advanced language models.""",
    llm = gemini,
    tools = [search_tool, scrape_tool],
    verbose = False,
    # allow_delegation = True,
    # cache = False,  # Disable cache for this agent
)

answering_task = Task(
    description = (
        "Answer dermatology-related user queries with web searching. User's question: {question}"
        "Provide a concise response in 1 paragraph, with bullets for each item. "
        "If the response is a list, include a brief definition for each item. "
    ),
    expected_output = (
        "A detailed and accurate response based on your knowledge and search results."
    ),
    agent = dermatology_agent,
    llm = gemini,
    verbose = False,
)

crew = Crew(
    agents = [dermatology_agent],
    tasks = [answering_task],
    verbose = False,
)

if __name__ == "__main__":

    question = "What causes Eczema?"
    inputs = {"question": question}


    results = crew.kickoff(inputs=inputs)
    print(results)




