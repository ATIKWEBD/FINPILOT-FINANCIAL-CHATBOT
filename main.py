import os
import gradio as gr
from dotenv import load_dotenv

# --- LangChain & RAG Imports ---
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate # <-- ADD THIS IMPORT

# --- Load Environment Variables ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

# --- ADD THE SYSTEM PROMPT BACK IN ---
SYSTEM_PROMPT = """
You are 'FinPilot', an expert AI guide for financial literacy in India.

## Your Core Mission
Your primary goal is to empower users by providing clear, understandable, and responsible information about common financial products and concepts. You are an unbiased educational guide, not a salesperson for any specific company. You must never guarantee loan approval.

## Your Conversational Flow
When a user asks about needing a loan, follow this structured process:
1.  Acknowledge & Empathize: Start by acknowledging their need.
2.  Ask Clarifying Questions: If the query is vague, ask questions to understand their specific situation.
3.  Educate Proactively: Proactively offer simple explanations for key terms like EMI, APR, etc.
4.  Recommend & Justify: Suggest suitable *types* of loan products and explain why they fit.
5.  Manage Expectations: Remind the user that eligibility depends on the lender's policies, credit score, etc.
6.  Offer Next Steps: Conclude by suggesting a general next step, like contacting a reputable bank or NBFC.

## Your Knowledge Base & Rules
- You are an expert on the content provided in the knowledge base (financial concepts, RBI rules). Always prioritize this information in your answers.
- You have high-level knowledge of common loan types: Personal, Vehicle, Home, Business, and Agricultural Loans.
- CRITICAL SAFETY RULE: You must NEVER ask for any Personal Identifiable Information (PII).
- Handling Out-of-Scope Questions: If asked about topics outside your knowledge, politely state your expertise is limited to general financial literacy.
- Output Formatting: Use markdown (bolding, bullet points) to make answers clear and readable.
"""
# ------------------------------------

# --- 1. RAG PIPELINE SETUP ---
def setup_rag_pipeline():
    # Load documents
    loader = DirectoryLoader('./knowledge_base/', glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    if not documents:
        raise ValueError("No documents were loaded. Check that the files in knowledge_base have content.")

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    if not texts:
        raise ValueError("Document splitting resulted in 0 chunks. Check file content.")

    # Create embeddings and vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_documents(texts, embeddings)
    retriever = vector_store.as_retriever()

    # --- CREATE A CUSTOM PROMPT TEMPLATE ---
    prompt_template = SYSTEM_PROMPT + """

    CONTEXT:
    {context}

    QUESTION:
    {question}

    YOUR ANSWER:
    """
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    # ---------------------------------------

    # Create the RAG chain, now with the custom prompt
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key, temperature=0.7)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt} # <-- INJECT THE PROMPT HERE
    )
    return qa_chain

# --- Initialize the RAG chain ---
rag_chain = setup_rag_pipeline()

# --- 2. GRADIO UI & LOGIC ---
# ... (The rest of your Gradio UI code is correct and does not need to be changed) ...
custom_css = """
#main_block { background: #f9f9f9; }
.gradio-container { font-family: 'Helvetica Neue', sans-serif; }
#header { text-align: center; padding: 20px; background-color: #00796b; color: white; }
#header h1 { font-size: 3em; margin: 0; }
#header p { font-size: 1.2em; margin-top: 5px; }
"""

def finpilot_query(query):
    """The function that runs the RAG chain and formats the output."""
    try:
        result = rag_chain.invoke({"query": query})
        return result['result']
    except Exception as e:
        return f"An error occurred: {str(e)}"

with gr.Blocks(theme='gradio/soft', css=custom_css, elem_id="main_block") as demo:
    with gr.Row(elem_id="header"):
        gr.HTML("<h1>FinPilot 🤖</h1><p>Your Advanced AI Guide for Financial Products & Knowledge</p>") # Changed title to be general

    with gr.Tabs():
        with gr.TabItem("FinPilot Chat"):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=500,
                type="messages"
            )
            
            with gr.Row():
                with gr.Column(scale=4):
                    msg = gr.Textbox(
                        label="Ask your question here:",
                        placeholder="Type your query and press Enter...",
                        container=False
                    )
                with gr.Column(scale=1):
                    clear = gr.Button("🗑️ Clear Conversation")

            def respond(message, chat_history):
                chat_history.append({"role": "user", "content": message})
                bot_message = finpilot_query(message)
                chat_history.append({"role": "assistant", "content": bot_message})
                return None, chat_history

            msg.submit(respond, [msg, chatbot], [msg, chatbot])
            clear.click(lambda: [], None, chatbot, queue=False)

        with gr.TabItem("Financial Knowledge Center"):
            gr.Markdown("### Core Financial Concepts")
            # ... (Accordion content remains the same) ...
            with gr.Accordion("What is a Credit Score?", open=False):
                gr.Markdown(finpilot_query("Explain everything about a Credit Score based on my knowledge base."))
            with gr.Accordion("Explain Assets vs. Liabilities", open=False):
                gr.Markdown(finpilot_query("Explain Assets vs. Liabilities in detail based on my knowledge base."))
            
            gr.Markdown("### Key RBI Guidelines")
            with gr.Accordion("Rules for Digital Lending", open=False):
                gr.Markdown(finpilot_query("What are the key RBI guidelines for digital lending according to my documents?"))
            with gr.Accordion("What is the Cooling-Off Period?", open=False):
                gr.Markdown(finpilot_query("Explain the cooling-off period for digital loans."))

if __name__ == '__main__':
    demo.launch(share=True)