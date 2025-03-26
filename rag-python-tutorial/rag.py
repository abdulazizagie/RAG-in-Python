import os
import signal
import sys

import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Set your Gemini API key
GEMINI_API_KEY = "AIzaSyCtPwXH7tfxqoRVSQdVEyaMjJMbRoDUA88"  # Replace with your actual API key

# Configure Gemini AI once
genai.configure(api_key=GEMINI_API_KEY)

# Load the model once
model = genai.GenerativeModel("gemini-1.5-pro-latest")


def generate_answer(prompt):
    """Generates an answer using Gemini 1.5 Pro."""
    answer = model.generate_content(prompt)
    return answer.text


def generate_rag_prompt(query, context):
    """Generates a RAG prompt using retrieved context."""
    escaped = context.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = f"""
You are a helpful and informative bot that answers questions using text from the reference context included below.
Be sure to respond in a complete sentence, being comprehensive, including all relevant background information.
However, you are talking to a non-technical audience, so be sure to break down complicated concepts and 
strike a friendly and conversational tone. 
If the context is irrelevant to the answer, you may ignore it.

QUESTION: '{query}'
CONTEXT: '{escaped}'

ANSWER:
"""
    return prompt


def get_relevant_context_from_db(query):
    """Fetches relevant context from ChromaDB."""
    context = ""
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./chroma_db_nccn", embedding_function=embedding_function)
    
    # Retrieve top 6 most relevant results
    search_results = vector_db.similarity_search(query, k=6)
    for result in search_results:
        context += result.page_content + "\n"
    
    return context


# Handle Ctrl+C gracefully
def signal_handler(sig, frame):
    print("\nThanks for using Gemini. :)")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

# Initial introduction
welcome_text = generate_answer("Can you quickly introduce yourself")
print(welcome_text)

# Continuous Query Loop
while True:
    print("-----------------------------------------------------------------------\n")
    print("What would you like to ask?")
    query = input("Query: ")
    
    context = get_relevant_context_from_db(query)
    prompt = generate_rag_prompt(query, context)
    
    answer = generate_answer(prompt)
    print(answer)
