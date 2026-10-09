from langchain_ollama import ChatOllama

# Connect to the locally installed AI model
llm = ChatOllama(
    model="llama3.2",
    temperature=0.7
)

# Ask a question
response = llm.invoke(
    "Explain the meaning of law in simple words."
)

# Display the answer
print("\nAI Chatbot:", response.content)