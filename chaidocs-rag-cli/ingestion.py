from langchain_community.document_loaders import WebBaseLoader
from bs4 import SoupStrainer
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

urls = [
    'https://docs.chaicode.com/youtube/chai-aur-html/introduction/',
    'https://docs.chaicode.com/youtube/chai-aur-html/emmit-crash-course/',
    'https://docs.chaicode.com/youtube/chai-aur-html/html-tags/',

    'https://docs.chaicode.com/youtube/chai-aur-git/introduction/',
    'https://docs.chaicode.com/youtube/chai-aur-git/terminology/',
    'https://docs.chaicode.com/youtube/chai-aur-git/behind-the-scenes/',
    'https://docs.chaicode.com/youtube/chai-aur-git/branches/',
    'https://docs.chaicode.com/youtube/chai-aur-git/diff-stash-tags/',
    'https://docs.chaicode.com/youtube/chai-aur-git/managing-history/',
    'https://docs.chaicode.com/youtube/chai-aur-git/github/'
]

web_loader = WebBaseLoader(
    web_paths=urls,
    bs_kwargs={"parse_only": SoupStrainer("main")}
)
documents = web_loader.load()


# Data Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=300
)
chunked_docs = text_splitter.split_documents(documents=documents)

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

# Recursive loader

# def content_extractor(html: str) -> str:
#     soup = BeautifulSoup(html, "lxml")

#     # Remove irrelevant elements (navbars, headers, footers, scripts, styles)
#     for tag in soup(["nav", "footer", "header", "aside", "script", "style"]):
#         tag.decompose()

#     content = soup.find("main") or soup.find("div", {"class": "sl-markdown-content"})

#     if content: 
#         text = content.get_text(separator="\n", strip=True)
#     else:
#         text = soup.get_text(separator="\n", strip=True)

#     # Normalize multiple newlines
#     text = re.sub(r"\n\n+", "\n\n", text).strip()
#     return text

# recursive_loader = RecursiveUrlLoader(
#     url="https://docs.chaicode.com/youtube/getting-started/",
#     extractor=content_extractor,
#     continue_on_failure=True,
#     prevent_outside=True,
#     max_depth=1000
# )

# docs = recursive_loader.load()

# for doc in docs:
#     print(doc.metadata)
#     print()
# docs[1]