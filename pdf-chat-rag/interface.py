import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
import ingestion as ig
import information_retrieval as ir

# intialize environment variables
load_dotenv()

def construct_filepath(filename):
    file_path = Path(__file__).parent / filename
    return file_path

st.title("Chat with your PDF")
st.write("Built with ❤️ using Streamlit")
st.warning("Doesn't have chat memory..")

# Chat-history = [Message]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Get the file from the user
uploaded_file = st.file_uploader(label="Upload your PDF", type="pdf")

if uploaded_file is not None:
    file_path = construct_filepath(uploaded_file.name)

    with st.spinner("Preparing your pdf for chatting..."):
        # save the file locally
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # begin the ingestion task
        if "vector_store" not in st.session_state:
            st.session_state.vector_store = ig.setup_vector_db(ig.chunk_docs(ig.load_pdf(file_path)))

    st.success("Your PDF is ready for Q&A")


# Message:{
#   "role": "user/assistant",
#   "content": "query/response"
# }
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# Get input query through chat-input
if user_input_query := st.chat_input("Ask your question.."):

    # Add user's input to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input_query})

    # Show user's input
    with st.chat_message("user"):
        st.markdown(user_input_query)

    # Show GPT response
    with st.chat_message("ai"):
        full_response = st.write_stream(ir.answer_query(user_input_query, st.session_state.vector_store))
    
    # Save GPT response to history
    st.session_state.chat_history.append({"role": "ai", "content": full_response})

