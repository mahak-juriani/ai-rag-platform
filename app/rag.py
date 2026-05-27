from pypdf import PdfReader
import chromadb
import os
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

chroma_client = chromadb.PersistentClient(path="chroma_db")

collection = chroma_client.get_or_create_collection(
    name="documents"
)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("ChromaDB initialized")
print("Embedding model loaded")

def extract_text_from_pdf(pdf_path):

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text

    return text


def chunk_text(text):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_text(text)

    return chunks

def store_chunks(chunks):

    for index, chunk in enumerate(chunks):

        embedding = embedding_model.encode(chunk).tolist()

        collection.add(
            ids=[str(index)],
            documents=[chunk],
            embeddings=[embedding]
        )

    print(f"Stored {len(chunks)} chunks")

def retrieve_relevant_chunks(query, top_k=3):

    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results["documents"][0]

def generate_answer(query):

    relevant_chunks = retrieve_relevant_chunks(query)

    context = "\n\n".join(relevant_chunks)

    prompt = f"""
    Use ONLY the context below to answer the question.

    If the answer is not found in the context, say:
    "I could not find that information in the document."

    Context:
    {context}

    Question:
    {query}
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": relevant_chunks
    }