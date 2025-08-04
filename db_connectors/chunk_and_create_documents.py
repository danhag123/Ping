# Document Processing and Chunking for Table Tennis Chatbot
# Handles PDF and text document processing, chunking, and storage in vector database

import os
from dotenv import load_dotenv                        # Load environment variables from .env file
import fitz                                          # PyMuPDF - PDF processing library
from .connection_qdrant import add_document_to_store # Custom function to store documents in Qdrant
from langchain_core.documents import Document        # LangChain document structure
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Intelligent text chunking

# Load environment variables from .env file
# This ensures database connections and API keys are available
load_dotenv()

def chunk_content(docs, metadata, is_pdf=False):
    """
    Processes documents (PDF or text) by extracting content, chunking it into smaller pieces,
    and storing the chunks in the vector database for retrieval.
    
    This function is crucial for the RAG system as it:
    1. Extracts text from PDFs or processes text documents
    2. Splits content into manageable chunks for better retrieval
    3. Stores chunks with metadata in the vector database
    
    Args:
        docs (str): File path to PDF or raw text content
        metadata (dict): Metadata to attach to document chunks (e.g., source, title)
        is_pdf (bool): Whether the input is a PDF file path (True) or text content (False)
    
    Returns:
        None: Documents are stored directly in the vector database
    """
    if is_pdf:
        # Extract text from PDF using PyMuPDF
        # This handles PDF parsing and text extraction from all pages
        pdf_text = ""
        with fitz.open(docs) as pdf_document:  # Open PDF file
            for page_num in range(len(pdf_document)):  # Iterate through all pages
                page = pdf_document.load_page(page_num)  # Load individual page
                pdf_text += page.get_text()  # Extract text content from page
        docs = pdf_text  # Replace file path with extracted text content
    else:
        # For non-PDF content, use the input directly as text
        docs = docs
    
    # Convert text content into LangChain Document objects
    # This standardizes the format for processing and adds metadata
    docs = [Document(page_content=docs, metadata=metadata)]
    
    # Initialize text splitter for intelligent chunking
    # RecursiveCharacterTextSplitter preserves semantic meaning by:
    # - chunk_size=1500: Maximum characters per chunk (optimal for embeddings)
    # - chunk_overlap=300: Overlap between chunks to maintain context continuity
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    
    # Split documents into smaller, manageable chunks
    # This creates multiple document chunks that can be independently embedded and retrieved
    all_splits = text_splitter.split_documents(docs)
    
    # Store all document chunks in the vector database (Qdrant)
    # Each chunk will be embedded and made searchable for the RAG system
    add_document_to_store(documents=all_splits)

# Used for testing and manual document processing
# This allows running the script directly to process specific documents
if __name__ == "__main__":
    # Example: Process Swedish table tennis rules PDF
    # This demonstrates how to add a PDF document to the knowledge base
    chunk_content(
        "/data/SBTF-Spelregler.pdf",  # Path to PDF file
        {"source": "Spelregler för bordtennis"},  # Metadata: Swedish table tennis rules
        is_pdf=True  # Specify this is a PDF file
    )