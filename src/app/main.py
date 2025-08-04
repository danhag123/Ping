# Table Tennis Chatbot (Ping) - Main Application
# A RAG (Retrieval-Augmented Generation) chatbot specialized in Swedish table tennis knowledge

import os
import logging
from typing import TypedDict
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from langgraph.graph import StateGraph
from langsmith.run_helpers import traceable
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from graphviz import Digraph
from .memory_store import memory

# =============================================================================
# FASTAPI APPLICATION SETUP
# =============================================================================

# Initialize FastAPI application
app = FastAPI()

# Mount static file directories for serving frontend assets
app.mount("/static", StaticFiles(directory="src/app/static"), name="static")
app.mount("/images", StaticFiles(directory="src/app/static/images"), name="images")

# Serve the main frontend page
@app.get("/")
async def serve_frontend():
    """Serve the main HTML page for the chatbot interface."""
    return FileResponse("src/app/static/index.html")

# Serve the query form page
@app.get("/query-form")
async def serve_query_form():
    """Serve the query form HTML page."""
    return FileResponse("src/app/static/query_form.html")

# Configure CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # WARNING: For development only - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# LOGGING AND ENVIRONMENT SETUP
# =============================================================================

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# AI MODEL INITIALIZATION
# =============================================================================

# Initialize OpenAI language model for text generation
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0.7  # Moderate creativity for natural responses
)

# Initialize OpenAI embeddings model for document similarity search
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",  # High-quality embeddings for better retrieval
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# NOTE: Memory is now imported from memory_store.py with Swedish configuration
# This ensures all conversation summaries are generated in Swedish with proper
# preservation of important information like organization numbers

# =============================================================================
# STATE DEFINITION FOR RAG WORKFLOW
# =============================================================================

class RAGState(TypedDict):
    """
    Defines the state structure that flows through the RAG workflow.
    
    Attributes:
        query: Original user question
        rewritten_query: Optimized version of the query for better retrieval
        documents: Retrieved relevant documents from vector store
        answer: Final generated response
        memory_summary: Summary of conversation history
    """
    query: str
    rewritten_query: str
    documents: list
    answer: str
    memory_summary: str

# =============================================================================
# VECTOR DATABASE SETUP (QDRANT)
# =============================================================================

# Initialize Qdrant client for vector similarity search
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# Ensure the vector collection exists, create if necessary
try:
    # Attempt to access existing collection
    qdrant_client.get_collection(os.getenv("QDRANT_COLLECTION"))
    logger.info("Connected to existing Qdrant collection")
except Exception:
    # Create new collection with appropriate vector configuration
    logger.info("Creating new Qdrant collection")
    qdrant_client.create_collection(
        collection_name=os.getenv("QDRANT_COLLECTION"),
        vectors_config=VectorParams(
            size=1536,  # Vector dimension for text-embedding-3-large
            distance=Distance.COSINE  # Cosine similarity for text comparison
        )
    )

# Initialize vector store wrapper for LangChain integration
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=os.getenv("QDRANT_COLLECTION"),
    embedding=embedding_model,
)

# Configure retriever with MMR (Maximal Marginal Relevance) for diverse results
retriever = vector_store.as_retriever(
    search_type="mmr",  # Reduces redundancy in retrieved documents
    search_kwargs={
        "k": 2,          # Return top 2 most relevant documents
        "fetch_k": 20,   # Consider top 20 candidates before MMR filtering
        "lambda_mult": 0.7  # Balance between relevance and diversity
    }
)

# =============================================================================
# RAG WORKFLOW NODE FUNCTIONS
# =============================================================================

@traceable  # Enable tracing for debugging and monitoring
def query_rewriter(state: RAGState, **kwargs):
    """
    Rewrites user queries to improve document retrieval accuracy.
    
    Process:
    1. Load conversation history for context (now in Swedish!)
    2. Remove unnecessary prefixes and improve clarity
    3. Maintain original meaning while optimizing for search
    
    Args:
        state: Current workflow state containing the original query
        
    Returns:
        dict: Updated state with rewritten_query
    """
    # Retrieve conversation history for context (Swedish summaries from memory_store)
    memory_variables = memory.load_memory_variables({})
    history = memory_variables.get("history", "")

    # Create prompt for query rewriting (in Swedish)
    rewrite_prompt = f"""
    Du ska skriva om följande fråga för att förbättra träffsäkerheten vid dokumenthämtning.
    Behåll den ursprungliga betydelsen och nyckelorden, men:
    1. Ta bort onödiga prefix, t.ex. "kan du kolla upp", "skulle du kunna", etc.
    2. Omformulera den till en tydlig, direkt fråga.
    3. Lägg inte till ny information eller ändra betydelsen.
    4. Använd kontext från tidigare konversationer för att göra frågan tydligare.
    5. Svara endast med den omskrivna frågan, utan extra kommentarer.

    Tidigare konversation:
    {history}

    Original fråga: {state['query']}
    """
    
    # Generate rewritten query using LLM
    rewritten = llm.invoke(rewrite_prompt).content.strip()
    logger.info(f"Query rewritten from '{state['query']}' to '{rewritten}'")
    
    return {"rewritten_query": rewritten}

@traceable
def retrieve_documents(state: RAGState, **kwargs):
    """
    Retrieves relevant documents from the vector store using the rewritten query.
    
    Uses MMR retrieval to get diverse, relevant documents that can inform
    the answer generation process.
    
    Args:
        state: Current workflow state containing the rewritten query
        
    Returns:
        dict: Updated state with retrieved documents
    """
    # Retrieve documents using the optimized query
    docs = retriever.get_relevant_documents(state["rewritten_query"])
    logger.info(f"Retrieved {len(docs)} documents for query: {state['rewritten_query']}")
    
    return {"documents": docs}

@traceable
def generate_answer(state: RAGState, **kwargs):
    """
    Generates the final answer using retrieved documents and conversation history.
    
    Process:
    1. Load conversation history for context (Swedish summaries)
    2. Prepare document context and sources
    3. Create comprehensive system prompt in Swedish
    4. Generate response using LLM with all context
    5. Save conversation to memory for future context
    
    Args:
        state: Current workflow state with query, documents, and history
        
    Returns:
        dict: Updated state with the generated answer
    """
    # Load conversation history for context (now Swedish summaries!)
    memory_variables = memory.load_memory_variables({})
    history = memory_variables.get("history", "")

    # Prepare document context and source information
    if not state["documents"]:
        context = "[INGA DOKUMENT]"  # No documents found
        sources = "[INGA KÄLLOR]"   # No sources available
    else:
        # Combine all document content
        context = "\n\n".join(doc.page_content for doc in state["documents"])
        # Extract source metadata from documents
        sources = "\n\n".join(doc.metadata.get("source", "okänd källa") for doc in state["documents"])

    # Comprehensive system instructions in Swedish for the table tennis assistant
    system_instructions = f"""
Du är en assistent som heter Ping, med expertkunskap kring bordtennis (i synnerhet svensk bordtennis).
Du förlitar dig främst på dokumentation och information som tillhandahålls (se "Relevant dokumentation"),
men du kan i viss mån även svara på generella frågor om bordtennis.

1. Roll & Behörighet
- Du svarar som en kunnig men neutral assistent.
- Om juridiska frågor uppstår, ge allmänna råd men inkludera en ansvarsfriskrivning att du inte är en juridisk rådgivare.

2. Källor & Prioritering
- I första hand: Tidigare konversationer (history) för sammanhang.
- I andra hand: Relevant dokumentation (context) som tillhandahålls.
- Du kan citera ur den dokumentation du har tillgång till.
- Om informationen inte finns i ovanstående källor, meddela osäkerhet eller be om förtydligande från användaren.

3. Specifika Riktlinjer
- **Sammanfattningsfrågor**
  Om en fråga endast efterfrågar en kort sammanfattning av tidigare konversation (utan nya detaljer),
  *använd inte* innehåll från de relevanta dokumenten. Besvara uteslutande baserat på det som sagts tidigare.

- **Osäker eller Ofullständig information**
  Om det saknas data i dokumentation eller tidigare konversationer, informera användaren att informationen inte är tillgänglig.

- **Hänvisa källor**
  Hänvisa alltid till källor om tillgängliga.

4. Sammanfattning av indata
- Tidigare Konversationer (history): {history}
- Relevant Dokumentation (context): {context}
- Källor: {sources}
""".strip()

    # Prepare messages for the LLM
    messages = [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": state["query"]},
    ]

    # Generate response using the language model
    response = llm.invoke(messages)
    answer = response.content.strip()
    
    # IMPORTANT: Save this conversation exchange to memory
    # This will create/update Swedish summaries for future context
    memory.save_context(
        {"input": state["query"]},
        {"output": answer}
    )
    
    logger.info("Generated response for query: %s", state["query"])
    logger.info("Response from OpenAI: %s", answer)
    logger.info("Conversation saved to Swedish memory")
    
    return {"answer": answer}

# =============================================================================
# WORKFLOW CONSTRUCTION
# =============================================================================

# Build the RAG workflow graph
workflow = StateGraph(RAGState)

# Add workflow nodes
workflow.add_node("rewrite_query", query_rewriter)
workflow.add_node("retrieve_docs", retrieve_documents)
workflow.add_node("generate_answer", generate_answer)

# Define workflow flow
workflow.set_entry_point("rewrite_query")  # Start with query rewriting
workflow.add_edge("rewrite_query", "retrieve_docs")  # Then retrieve documents
workflow.add_edge("retrieve_docs", "generate_answer")  # Finally generate answer
workflow.set_finish_point("generate_answer")  # End after generating answer

# Compile the workflow for execution
compiled_workflow = workflow.compile()

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def health_check():
    """
    Health check endpoint to verify API is running.
    
    Returns:
        dict: Status message confirming API availability
    """
    return {"status": "OK", "message": "Pingis RAG API is running"}

@app.post("/api/query")
async def handle_query(request: Request):
    """
    Main API endpoint for processing user queries.
    
    Process:
    1. Extract query from request
    2. Validate input
    3. Execute RAG workflow (which now includes Swedish memory management)
    4. Return structured response
    
    Args:
        request: HTTP request containing the user query
        
    Returns:
        dict: Response containing answer, metadata, and status
        
    Raises:
        HTTPException: For invalid requests or processing errors
    """
    try:
        # Parse JSON request body
        data = await request.json()
        query = data.get("query")
        
        # Validate that query is provided
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        logger.info(f"Processing query: {query}")
        
        # Execute the RAG workflow (now with proper Swedish memory integration)
        result = compiled_workflow.invoke({"query": query})
        
        # Return structured response with metadata
        return {
            "status": "success",
            "answer": result["answer"],
            "rewritten_query": result.get("rewritten_query", ""),
            "documents": len(result.get("documents", []))  # Number of documents used
        }
        
    except Exception as e:
        # Log error and return HTTP error response
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# OPTIONAL: MEMORY MANAGEMENT ENDPOINTS
# =============================================================================

@app.get("/api/memory/status")
async def get_memory_status():
    """
    Get current memory status and summary.
    
    Returns:
        dict: Current conversation summary and memory statistics
    """
    try:
        memory_variables = memory.load_memory_variables({})
        return {
            "status": "success",
            "current_summary": memory_variables.get("history", "Ingen tidigare konversation"),
            "language": "Swedish"
        }
    except Exception as e:
        logger.error(f"Error getting memory status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/memory/clear")
async def clear_memory():
    """
    Clear conversation memory.
    
    Returns:
        dict: Confirmation of memory clearing
    """
    try:
        memory.clear()
        logger.info("Conversation memory cleared")
        return {
            "status": "success",
            "message": "Konversationsminne har rensats"
        }
    except Exception as e:
        logger.error(f"Error clearing memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# WORKFLOW VISUALIZATION (OPTIONAL)
# =============================================================================

def create_workflow_visualization():
    """
    Creates a visual representation of the RAG workflow using Graphviz.
    
    Generates a PNG image showing the flow from query rewriting through
    document retrieval to answer generation.
    """
    # Create directed graph
    dot = Digraph()
    
    # Add nodes for each workflow step
    dot.node("rewrite_query", "Rewrite Query\n(Swedish Context)")
    dot.node("retrieve_docs", "Retrieve Documents") 
    dot.node("generate_answer", "Generate Answer\n(Save to Swedish Memory)")
    
    # Add edges showing workflow flow
    dot.edge("rewrite_query", "retrieve_docs")
    dot.edge("retrieve_docs", "generate_answer")
    
    # Render graph to PNG file
    dot.render("workflow_graph", format="png")
    logger.info("Workflow visualization saved as workflow_graph.png")

# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Generate workflow visualization when running directly
    create_workflow_visualization()