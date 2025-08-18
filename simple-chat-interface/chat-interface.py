from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st

# Load and Initialize
load_dotenv()
openai_client = OpenAI()

# Method to generate streaming response
def create_streaming_api_response(prompt_text):
    stream_response = openai_client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "user",
                "content": prompt_text
            }
        ],
        stream=True
    )

    for event in stream_response:
        if event.type == "response.output_text.delta":
            yield event.delta


# Streamlit powered interface for chatting
st.title("My Own GPT")
st.write("Built with ❤️ using Streamlit")
st.warning("It doesn't have chat memory capabilities yet.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if input_text := st.chat_input("Interact"):

    # Add user's input to chat history
    st.session_state.messages.append({"role": "user", "content": input_text})

    # Show user's input
    with st.chat_message("user"):
        st.markdown(input_text)

    # Show GPT response
    with st.chat_message("ai"):
        full_response = st.write_stream(create_streaming_api_response(input_text))
    
    # Save GPT response to history
    st.session_state.messages.append({"role": "ai", "content": full_response})