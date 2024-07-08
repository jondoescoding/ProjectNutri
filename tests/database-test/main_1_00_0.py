# Import necessary libraries
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()

# API KEYS
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')

# Ensure API keys are set
if not LANGCHAIN_API_KEY or not openai_api_key:
    raise ValueError("API keys for Langchain or OpenAI are not set.")

# Set the path to your database file
DB_PATH = r"D:\CODING\projectNutri\tests\database-test\client_health_improved.db"

# Connect to the database
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

# Create a SQLDatabaseChain to create and execute SQL queries
# LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, api_key=openai_api_key)
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=False)

# Load the PDF file
pdf_loader = PyPDFLoader(r"D:\CODING\projectNutri\unsolved-cases\Jonathan_White_Health_Details.pdf")
context = pdf_loader.load()

# Now you can use db_chain.run("Your question here") to query the database
# For example:
# Pass the loaded PDF context to the chain and ask for a meal plan
response = db_chain.invoke("Generate a meal plan for the following client", context=context)
print(response)