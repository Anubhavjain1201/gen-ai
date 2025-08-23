from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

# Create vector_store
def create_vector_store():
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    qdrant_vector_store = QdrantVectorStore.from_existing_collection(
        collection_name="pdf-rag",
        embedding=embedding_model,
        url="http://localhost:6333"
    )

    return qdrant_vector_store


# Define system prompt
def create_system_prompt(search_context):
    system_prompt = f"""
    You are a helpful assistant that answers user queries based on the context that is provided to you.
    The context that you get is fetched from a pdf file.
    It features page content and page number.

    ## Rules
        1. You never answer any query that doesn't relate to the context that is provided.
        2. You only answer query based on the given context and don't include any outside information
        3. You don't create any bias of any information outside of the context.
        4. You guide the user to open the right page in the file for more information.

    ## Available Context
    Context: {search_context}
    """

    return system_prompt


# Answer user query
def answer_query(input_query, vector_store: QdrantVectorStore):

    similar_results = vector_store.similarity_search(query=input_query)

    # Prepare search_context
    search_context = "\n\n".join(
        [f"Page Content: {result.page_content}\n Page Number: {result.metadata["page_label"]}\n File: {result.metadata["title"]}" 
        for result in similar_results]
    )

    # get system prompt
    system_prompt = create_system_prompt(search_context=search_context)

    # Get LLM response
    openai_client = OpenAI()
    chat_response = openai_client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": input_query,
            }
        ],
        stream=True
    )

    for event in chat_response:
        if event.type == "response.output_text.delta":
            yield event.delta
