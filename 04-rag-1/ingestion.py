from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

# Initialize resources
load_dotenv()
pdf_path = Path(__file__).parent / 'Physics.pdf'

# Load Data Source
pdf_loader = PyPDFLoader(file_path=pdf_path)
docs = pdf_loader.load()

# Data Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=500
)
chunked_docs = text_splitter.split_documents(documents=docs)

# Create vector embeddings
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

# Store Embeddings in a Vector Database (Qdrant DB)
qdrant_vector_store = QdrantVectorStore.from_documents(
    documents=chunked_docs,
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="local-pdf-rag"
)

print("Ingestion and indexing completed...")