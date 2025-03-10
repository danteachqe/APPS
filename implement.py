import os
import fitz  # PyMuPDF
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Set API key or raise an error if not set
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

# Instantiate the OpenAI client
client = OpenAI(api_key=api_key)

# FAISS Database File
DB_FILE = "faiss_index"

# Define the GPT model class
class GPT35Model:
    def __init__(self, model_name="gpt-4"):
        self.model_name = model_name

    # Generates a response based on retrieved context
    def generate(self, prompt: str) -> str:
        completion = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content

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
            continue  # Skip unsupported files

        if text.strip():
            documents.append(Document(page_content=text, metadata={"source": file_name}))
    
    return documents

def create_vector_db(documents):
    """Processes documents, creates FAISS vector DB, and saves it locally."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vector_db = FAISS.from_documents(docs, embeddings)

    # Save FAISS index locally
    vector_db.save_local(DB_FILE)
    print(f"Vector DB created and saved at {DB_FILE}")

def load_vector_db():
    """Loads FAISS vector DB from local file (allowing safe deserialization)."""
    if os.path.exists(DB_FILE):
        return FAISS.load_local(DB_FILE, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    else:
        raise FileNotFoundError("No FAISS DB found! Create one first.")

def retrieve_relevant_docs(query, k=3):
    """Finds the top-k relevant chunks from FAISS DB."""
    vector_db = load_vector_db()
    results = vector_db.similarity_search(query, k=k)
    return "\n".join([doc.page_content for doc in results])

def chat_with_context(query):
    """Fetches relevant documents, displays them, and queries ChatGPT with context."""
    context = retrieve_relevant_docs(query)
    
    print("\n🔍 RAG Retrieved Context Before Sending to LLM:\n")
    print(context)  # Show retrieved context before sending to ChatGPT

    if not context.strip():
        return "⚠ No relevant documents found in the database."

    prompt = f"Use the following context to answer the query:\n\n{context}\n\nQuery: {query}"

    model = GPT35Model()
    response = model.generate(prompt)  # Use the GPT-4 model class
    return response

if __name__ == "__main__":
    folder_path = "documents"  # Change this if your folder is elsewhere
    documents = load_documents(folder_path)

    # Create FAISS database (run only once per dataset)
    create_vector_db(documents)

    # Run queries
    while True:
        user_query = input("Enter your query (or 'exit' to stop): ")
        if user_query.lower() == "exit":
            break
        response = chat_with_context(user_query)
        print("\n💬 ChatGPT Response:\n", response)
