from fastapi import FastAPI, HTTPException
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import re

app = FastAPI()

# Initialize ChromaDB and OpenAI
embedding_model = OpenAIEmbeddings()
chroma_db = Chroma.from_existing_collection("your-chroma-collection", embedding_function=embedding_model)


# API endpoint to ask the chatbot a question
@app.post("/ask")
async def ask_chatbot(query: str):
    # Sanitize user input
    sanitized_query = sanitize_input(query)

    # Retrieve relevant data from Chroma
    retriever = chroma_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    qa_chain = RetrievalQA(combine_docs_chain=OpenAI(), retriever=retriever)
    answer = qa_chain.run(sanitized_query)
    
    return {"answer": answer}

