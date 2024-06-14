MODEL = "res/llama-2-7b-chat.ggmlv3.q4_1.bin"
# The model was selected from the following website that matches our requirements from the 
# model the best : https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML
# There are various version of the model and see the "max RAM" and "Use case" section 
# to figure out which one is the best 
MODEL_TYPE = "llama"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.7
CONTEXT_LENGTH = 2048
INPUT_VARIABLES = ["context", "question"]
SEPARATORS = "\n"
CHUNK_SIZE = 450
CHUNK_OVERLAP = 190
MODEL_KWARGS = {'device': 'cuda'}
EMBEDDER = "thenlper/gte-large"
CHAIN_TYPE = "stuff"
SEARCH_KWARGS = {'k': 2}
PROMPT_TEMPLATE = """
You are a great researcher. With the information 
provided understand in deep and try to answer the question. 
If you cant answer the question based on the information 
either say you cant find an answer or unable to find an answer.
So try to understand in depth about the context and answer only 
based on the information provided. Dont generate irrelevant answers.

Context: {context}
Question: {question}
Do provide only helpful answers

Answer:
"""
