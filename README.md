# FinPilot 🤖 - Your AI Financial Guide

An AI-powered financial guide chatbot built with Retrieval-Augmented Generation (RAG), LangChain, and Google Gemini. FinPilot features an interactive Gradio UI and a custom knowledge base on Indian financial concepts and RBI regulations to provide expert, unbiased advice.

## 🚀 Demo

<img width="1810" height="931" alt="image" src="https://github.com/user-attachments/assets/4c1d4cc9-6b49-4ed9-b5d0-a4944a524560" />
<img width="1579" height="871" alt="image" src="https://github.com/user-attachments/assets/3eccd101-f057-4baf-bf5f-d83dd8d211e6" />



## ✨ Key Features
* **Conversational AI:** A natural language chat interface to ask complex financial questions.
* **Retrieval-Augmented Generation (RAG):** Answers are based on a custom, factual knowledge base, minimizing hallucinations and providing accurate information.
* **Expert Knowledge Base:** Loaded with detailed information on dozens of financial concepts (Credit Score, EMI, APR, LTV) and RBI guidelines (Fair Practices Code, Digital Lending Rules).
* **Advanced AI Persona:** Guided by a sophisticated system prompt to ensure responses are professional, empathetic, and follow a structured conversational flow.
* **Interactive UI:** A clean, tabbed Gradio interface that is intuitive and easy to use.

## 🛠️ Tech Stack
* **Backend:** Python
* **AI & ML:** LangChain, Google Gemini (`gemini-1.5-flash`), FAISS (for vector storage)
* **UI:** Gradio

## 🧠 Project Architecture
The application uses a RAG (Retrieval-Augmented Generation) pipeline to provide accurate, context-aware answers:

1.  **Knowledge Base:** Text files containing financial concepts and RBI rules are loaded and split into chunks.
2.  **Vectorization:** The text chunks are converted into numerical vectors (embeddings) using Google's embedding model and stored in a FAISS vector store.
3.  **User Query:** When a user asks a question, their query is also converted into a vector.
4.  **Retrieval:** The FAISS vector store performs a similarity search to find the most relevant text chunks from the knowledge base.
5.  **Generation:** The original question, the retrieved context, and the advanced system prompt are passed to the Gemini LLM, which generates the final, accurate, and well-formatted answer.

## ⚙️ Setup and Installation

Follow these steps to run the project locally:

**1. Clone the Repository**
```bash
git clone <https://github.com/ATIKWEBD/FINPILOT-FINANCIAL-CHATBOT>
cd <https://github.com/ATIKWEBD/FINPILOT-FINANCIAL-CHATBOT/edit/main/README.md>
```

**2. Create a Virtual Environment**
```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Set Up Environment Variables**
* Create a file named `.env` in the root of your project folder.
* Get your API key from [Google AI Studio](https://aistudio.google.com/).
* Add your key to the `.env` file:
    ```
    GOOGLE_API_KEY="PASTE_YOUR_API_KEY_HERE"
    ```

**5. Populate the Knowledge Base**
* Ensure your `knowledge_base` folder contains the `.txt` files with the financial information.

**6. Run the Application**
```bash
python main.py
```
The application will be available at a local URL like `http://127.0.0.1:7860`.
