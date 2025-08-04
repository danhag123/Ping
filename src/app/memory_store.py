# Memory Store for Table Tennis Chatbot (Ping)
# Handles conversation memory and summarization in Swedish

# Import LangChain components for conversation memory management
from langchain.memory import ConversationSummaryMemory  # Stores conversation summaries instead of full history
from langchain_openai import ChatOpenAI                 # OpenAI language model integration
from langchain.prompts import PromptTemplate           # Template for structured prompts

# Initialize dedicated LLM for memory summarization tasks
# Using temperature=0 for consistent, deterministic summaries across conversations
llm_memory = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Define a Swedish summarization prompt
# This template ensures all conversation summaries are generated in Swedish
# and follow a specific format for consistency and information preservation
swedish_summary_prompt = PromptTemplate(
input_variables=["summary", "new_lines"],  # Two inputs: existing summary + new conversation content
template=(
"Sammanfatta följande konversation kortfattat på svenska med den viktigaste informationen:\n\n"
# Translation: "Summarize the following conversation briefly in Swedish with the most important information:"
"Tidigare sammanfattning:\n{summary}\n\n"
# Translation: "Previous summary: {summary}"
"Nya rader:\n{new_lines}\n\n"
# Translation: "New lines: {new_lines}"
"Generera en uppdaterad kortfattad sammanfattning på svenska."
# Translation: "Generate an updated brief summary in Swedish."
"Kom ihåg alla frågor från användaren och alla dina svar. Spara även ner viktig information som organisationsnummer."
# Translation: "Remember all questions from the user and all your answers. Also save important information like organization numbers."
 ),
)

# Initialize memory with Swedish prompt
# This creates a conversation memory system that:
# - Uses the dedicated LLM for summarization
# - Returns string summaries (not message objects) for easier handling
# - Uses the custom Swedish prompt for all summarization operations
memory = ConversationSummaryMemory(
llm=llm_memory,                    # Use the dedicated summarization LLM
return_messages=False,             # Return strings instead of message objects for simpler integration
prompt=swedish_summary_prompt      # Use our custom Swedish summarization template
)