# OS 
import os
import pprint
from io import BytesIO
from dotenv import load_dotenv

# PYTHON LIBRARIES

# IMAGING
import pytesseract
import pypdfium2 as pdfium
from pytesseract import image_to_string
from PIL import Image


# LANGCHAIN
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader 
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnablePick  
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings


load_dotenv()
print("Environment variables loaded.")

# API KEYS
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')

#### INDEXING ####



# Function to convert PDF to images
def convert_pdf_to_images(pdf_path, scale=300/72):
    """Converts each page of a PDF file into a list of images."""
    pdf_file = pdfium.PdfDocument(pdf_path)

    page_indices = [i for i in range(len(pdf_file))]

    renderer = pdf_file.render(
        pdfium.PdfBitmap.to_pil,
        page_indices=page_indices,
        scale=scale,
    )

    final_images = []

    for i, image in zip(page_indices, renderer):

        image_byte_array = BytesIO()
        image.save(image_byte_array, format='jpeg', optimize=True)
        image_byte_array = image_byte_array.getvalue()
        final_images.append(dict({i: image_byte_array}))

    return final_images

# Function to extract text from a list of images
def extract_text_from_images(images):
    """Extracts text from each image using pytesseract."""
    image_list = [list(data.values())[0] for data in images]
    image_content = []

    for index, image_bytes in enumerate(image_list):

        image = Image.open(BytesIO(image_bytes))
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        raw_text = str(image_to_string(image))
        image_content.append(raw_text)

    return "\n".join(image_content)

# Function to read the list of processed files
def read_processed_files(filepath):
    print("Reading list of processed files")
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return set(f.read().splitlines())
    return set()

# Function to write the list of processed files
def write_processed_files(filepath, processed_files):
    print("New file being added to the list")
    with open(filepath, 'w') as f:
        for file in processed_files:
            f.write(f"{file}\n")

# Function for joining each page of the split document
def format_docs(docs) -> str:
    '''Format the docs.'''
    return "\n".join(doc.page_content for doc in docs)

# Path to the directory and processed files tracker
directory_path = r'D:\CODING\projectNutri\solved-cases'
processed_files_tracker = 'processed_files.txt'

# Load documents
loader = PyPDFDirectoryLoader(path=directory_path)

# Read the list of already processed files
processed_files = read_processed_files(processed_files_tracker)

# Get the list of all files in the directory
all_files = set(os.listdir(directory_path))

# Determine new files by subtracting processed files from all files
new_files = all_files - processed_files

# If there are new files, process them

# Embed

# Embeddings Model instantiation
embeddings_model = HuggingFaceEmbeddings()

vectorstore = Chroma(embedding_function=embeddings_model)

retriever = vectorstore.as_retriever()

if new_files:
    print("New files found")
    # Update the list of processed files
    processed_files.update(new_files)
    write_processed_files(processed_files_tracker, processed_files)

    # Load and process new documents
    docs = loader.load()  # Make sure the loader supports loading specific files
    print(f"Loaded {len(new_files)} new documents.")

    # Splitting the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    print(f"Split new documents into {len(splits)} chunks.")

    # Embed the documents or chunks
    embedded_documents = embeddings_model.embed_texts([doc.page_content if isinstance(doc.page_content, str) else " ".join(doc.page_content.values()) for doc in splits])

    # Add the embedded documents to the vector store
    vectorstore.add_documents(embedded_documents)
    print("Documents have been embedded and added to the vectorstore.")
else:
    print("No new documents to process.")


#### RETRIEVAL and GENERATION ####
prompt = hub.pull("jondoescoding/rag-prompt-nutritionist")
print("Pulled prompt")

# LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, api_key=openai_api_key)

# Pre-processing
# Convert the PDF document to images
pdf_images = convert_pdf_to_images(r'D:\CODING\projectNutri\unsolved-cases\Jonathan_White_Health_Details.pdf')
print("Converted PDF to images")

# Extract text from the images
extracted_text = extract_text_from_images(pdf_images)
print("Extracted text from images")

# Combine the text from all pages into a single string
healthDoc = " ".join(extracted_text)
instruction = "Generate a full report for the client. Use other clients to base your diagnosis from. State which client(s) you used"

# RAG PROMPT
chain = (
    {"clientHealthData": retriever | format_docs, "task": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Question
print("Sending Question to LLM")
# Open a file in write mode
with open('output_report.txt', 'w') as file:
    pprint.pprint(
        chain.invoke(" ".join([instruction, healthDoc])),
        stream=file  # Redirect the output to the file
    )