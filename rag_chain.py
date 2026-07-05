from dotenv import load_dotenv

# LLM
from langchain_groq import ChatGroq

# Prompt
from langchain_core.prompts import ChatPromptTemplate

# PDF Loading & Chunking
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings & Vector Store
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# RAG Chain builders
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

# warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

pdf_path = "TransformerAttentionMechanism.pdf"

print(f" Loading PDF: {pdf_path}")
loader = PyPDFLoader(pdf_path)
pages = loader.load()

print("Chunking the document...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(pages)
print(f"   → {len(chunks)} chunks created")

print(" Storing in ChromaDB...")
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="pdf_collection"
)

rag_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful assistant that answers questions based ONLY 
on the provided context from a PDF document.

RULES:
- Answer the question using ONLY the information in the context below
- Be concise and direct in your response
- If the context contains relevant information, provide a clear answer
- Use bullet points for listing multiple items

IMPORTANT: If the answer is NOT in the provided context, strictly say:
"I don't know based on the provided document."
Do NOT make up or assume any information.

CONTEXT:
{context}"""
    ),
    (
        "human",
        "{input}"
    )
])

document_chain = create_stuff_documents_chain(llm, rag_prompt)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

rag_chain = create_retrieval_chain(retriever, document_chain)

question1 = "What is the dimension of the embedding vector?"
response1 = rag_chain.invoke({"input": question1})
print(f" Answer: {response1['answer']}\n")

question2 = "What is the recipe for chocolate cake?"
response2 = rag_chain.invoke({"input": question2})
print(f" Answer: {response2['answer']}")