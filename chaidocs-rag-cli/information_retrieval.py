from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

# Initialize resources
load_dotenv()

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

qdrant_vector_store = QdrantVectorStore.from_existing_collection(
    collection_name="local-pdf-rag",
    embedding=embedding_model,
    url="http://localhost:6333"
)

openai_client = OpenAI()

# Take user input
input_query = input("> ")

# Perform vector similarity search
search_results = qdrant_vector_store.similarity_search(
    query=input_query
)

# print(search_results)

# Response construction through LLM
search_context = "\n\n".join(
    [f"Webpage Content: {result.page_content}\n Source Webpage: {result.metadata["source"]}" 
     for result in search_results]
)

SYSTEM_PROMPT = f"""
You are a helpful assistant that answers user queries based on the context that is provided to you.
The context that you get is scraped and collected from a website: docs.chaicode.com
It contains docs for useful projects and technologies.

## Rules
    1. You never answer any query that doesn't relate to the context that is provided.
    2. You only answer query based on the given context and don't include any outside information
    3. You don't create any bias of any information outside of the context.
    4. You provide the user with the right webpage link to know more about the content

## Available Context
Context: {search_context}
"""

chat_response = openai_client.responses.create(
    model="gpt-4.1-mini",
    input=[
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": input_query,
        }
    ]
)

print(chat_response.output_text)

