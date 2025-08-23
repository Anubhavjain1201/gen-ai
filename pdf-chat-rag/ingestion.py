from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

# Load data source
def load_pdf(file_path):
    pdf_loader = PyPDFLoader(
        file_path=file_path
    )
    docs = pdf_loader.load()
    return docs

# Data chunking
def chunk_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 700,
        chunk_overlap = 300
    )
    split_docs = text_splitter.split_documents(docs)
    return split_docs

# Create vector embeddings and load into vector db
def setup_vector_db(split_docs):
    # Initialize embedding model
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    # setup vector store
    QdrantVectorStore.from_documents(
        collection_name="pdf-rag",
        embedding=embedding_model,
        documents=split_docs,
        url="http://localhost:6333"
    )