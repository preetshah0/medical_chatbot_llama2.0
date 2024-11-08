from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import os
from  langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


#Initializing the Pinecone
import pinecone

# Initialize Pinecone client
pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment=os.environ.get("PINECONE_API_ENV")
)

# Define the index name
index_name = "medical-chatbot"

# Creating embeddings for each of the text chunks and storing them
docsearch = pc.from_texts(
    [t.page_content for t in text_chunks],
    embeddings,
    index_name=index_name
)
