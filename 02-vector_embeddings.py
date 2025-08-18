from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

openai_client = OpenAI()

text1 = "dog chases cat"
text2 = "cat chases dog"

response = openai_client.embeddings.create(
    model="text-embedding-3-small",
    input=[text1, text2]
)

print(len(response.data[0].embedding))