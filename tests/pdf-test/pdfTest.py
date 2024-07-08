# Import necessary libraries
import os
from dotenv import load_dotenv

# LANGCHAIN
from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader


# Load environment variables
load_dotenv()

# API KEYS
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')

def extract_pdf_data(pdf_directory):
    loader = PyPDFDirectoryLoader(path=pdf_directory)
    docs = loader.load()
    return docs


# Step 2: Load into vector store
def load_into_vector_store(texts, chunk_size=400, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    splitText = text_splitter.split_documents(texts)
    
    vectorstore = Chroma.from_documents(documents=splitText, embedding=HuggingFaceEmbeddings())
    
    return vectorstore

# Function for joining each page of the split document
def format_docs(docs) -> str:
    '''Format the docs.'''
    return " ".join(eachPage.page_content for eachPage in docs)

def query_vector_store_with_llm(vectorstore, pdf_file_path):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, api_key=openai_api_key)
    
    retriever = vectorstore.as_retriever()
    
    #### RETRIEVAL and GENERATION ####
    prompt = hub.pull("jondoescoding/rag-prompt-nutritionist")
    print("Pulled prompt")

    rag_chain = (
    {"context": retriever | format_docs, "clientHealthData": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
    
    # Load and extract text from the PDF file
    pdf_loader = PyPDFLoader(file_path=pdf_file_path)
    loadedPDF = pdf_loader.load()

    textSplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splitText = textSplitter.split_documents(loadedPDF)
    
    # Use the extracted text as context for the query
    clientHealthData = f"{splitText}"
    response = rag_chain.invoke(clientHealthData)
    return response


# Step 4: Send output of the LLM to a file
def write_response_to_file(response, file_path):
    with open(file_path, 'w') as file:
        file.write(response)

# Usage
# TEXT EXTRACTION
pdf_texts = extract_pdf_data(r"D:\CODING\projectNutri\solved-cases")
print(f"Loaded PDFs.")

# VECTORSTORE LOADING
vectorstore = load_into_vector_store(pdf_texts)
print("Loaded data into vector store.")

# QUERY
pdf_file_path = r"D:\CODING\projectNutri\unsolved-cases\Jonathan_White_Health_Details.pdf"
save_path = r"D:\CODING\projectNutri\tests\pdf-test\client_meal_plan_report.txt"
response = query_vector_store_with_llm(vectorstore, pdf_file_path)

# WRITE ANSWER TO FILE
write_response_to_file(response, save_path)
print("Written LLM response to file.")