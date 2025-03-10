import os
import fitz  # PyMuPDF
import json
from typing import List

# RAGAS Evaluation Imports
from ragas.metrics import (
    context_precision,  # Measures if retrieved text is relevant to the answer.
    context_recall  # Measures if all necessary information was retrieved.
)
from ragas import evaluate
from datasets import Dataset

# FAISS & LangChain Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# =============================================================================
# FAISS & Document Loading Setup
# =============================================================================
DB_FILE = "faiss_index"

llm = ChatOpenAI(model="gpt-4o")
embeddings = OpenAIEmbeddings()

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using PyMuPDF."""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
    return text

def load_documents(folder_path):
    """Loads all PDF and text documents from a folder."""
    documents = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        elif file_name.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            continue  

        if text.strip():
            documents.append(Document(page_content=text, metadata={"source": file_name}))
    return documents

def create_vector_db(documents):
    """Processes documents, creates FAISS vector DB, and saves it locally."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    vector_db = FAISS.from_documents(docs, embeddings)
    vector_db.save_local(DB_FILE)
    print(f"Vector DB created and saved at {DB_FILE}")

def load_vector_db():
    """Loads FAISS vector DB from local storage."""
    if os.path.exists(DB_FILE):
        return FAISS.load_local(DB_FILE, embeddings, allow_dangerous_deserialization=True)
    else:
        raise FileNotFoundError("No FAISS DB found! Create one first.")

def retrieve_relevant_docs(query, k=3):
    """Retrieves the top-k relevant document chunks from FAISS."""
    vector_db = load_vector_db()
    results = vector_db.similarity_search(query, k=k)
    return [doc.page_content for doc in results]

# =============================================================================
# RAGAS Evaluation (Retrieval and Answer Testing)
# =============================================================================
def evaluate_ragas(query: str, retrieved_docs: List[str], generated_answer: str, correct_answer: str):
    """Evaluates the RAG retrieval system and response using RAGAS metrics."""
    
    if not isinstance(retrieved_docs, list):
        retrieved_docs = [retrieved_docs]  

    # Prepare data for evaluation
    evaluation_data = {
        "question": [query],
        "contexts": [retrieved_docs],
        "answer": [generated_answer],
        "reference": [correct_answer]
    }
    dataset = Dataset.from_dict(evaluation_data)

    # Run RAGAS evaluation
    scores = evaluate(dataset, metrics=[
        context_precision,
        context_recall
    ])

    scores_dict = scores.__dict__
    if "evaluation_dataset" in scores_dict:
        scores_dict.pop("evaluation_dataset")
    only_scores = scores_dict.get("scores", {})

    print("\n📊 **RAGAS Evaluation Scores:**")
    print(json.dumps(only_scores, indent=2, default=str))

# =============================================================================
# Chat with Context and Evaluate Response
# =============================================================================
def chat_with_context(query, correct_answer):
    """Fetches relevant documents, queries ChatGPT, and evaluates response."""
    retrieved_docs = retrieve_relevant_docs(query)
    print("\n🔍 RAG Retrieved Contexts:\n")
    for idx, doc in enumerate(retrieved_docs, 1):
        print(f"[{idx}] {doc}\n")
    
    if not retrieved_docs:
        print("⚠ No relevant documents found in the database.")
        return
    
    prompt = f"Use the following context to answer the query:\n\n{retrieved_docs}\n\nQuery: {query}"
    response = llm.invoke(prompt)
    generated_answer = response.content
    print("\n💬 ChatGPT Response:\n", generated_answer)
    
    evaluate_ragas(query, retrieved_docs, generated_answer, correct_answer)

# =============================================================================
# Main Script Execution
# =============================================================================
if __name__ == "__main__":
    folder_path = "documents"  # Folder with PDFs and TXT files
    documents = load_documents(folder_path)
    create_vector_db(documents)

    while True:
        user_query = input("Enter your query (or 'exit' to stop): ")
        if user_query.lower() == "exit":
            break
        
        correct_answer = input("\n✅ Enter the expected correct answer for evaluation: ")
        chat_with_context(user_query, correct_answer)
