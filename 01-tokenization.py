import tiktoken

encoder = tiktoken.encoding_for_model("gpt-4o")

text_to_encode = "Hi this is Anubhav learning GenAI"

tokens = encoder.encode(text_to_encode)
# print(tokens)

decoded_text = encoder.decode(tokens)
print(decoded_text)




