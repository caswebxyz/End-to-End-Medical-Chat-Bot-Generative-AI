from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate


from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)
load_dotenv()

# --- Load environment variables ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_REGION = "us-east-1"
INDEX_NAME = "medicalbot-index"

# --- Initialize Pinecone client (v3 syntax) ---
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if needed
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # all-MiniLM-L6-v2 embedding size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
    )

# --- Load embeddings ---
embeddings = download_hugging_face_embeddings()

# --- Connect to index ---
index = pc.Index(INDEX_NAME)

# ✅ Use the new LangChain PineconeVectorStore
docsearch = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text",
)

# --- Prompt template ---
system_prompt = (
    "You are an medical expert assistant for question-answering tasks"
    "Use the following pieces of retrieved context to answer the question at the end. "
    "If you don't know the answer, just say that you don't know, don't try to make up an answer."
    "Use three sentences maximum and be precise." 
    "\n\n"
    "{context}\n"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Answer the question: {input}")
])

llm = OpenAI()

# --- Load local Llama model ---
# llm = CTransformers(
#     model="TheBloke/Llama-2-7B-Chat-GGML",        # or your local path
#     model_type="llama",
#     config={'max_new_tokens':512, 'temperature':0.8}
# )

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
# --- Create RetrievalQA chain ---
question_answering_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

rag_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=question_answering_chain)


# --- Flask routes ---
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input=msg
    print(f"User Query: {msg}")
    response = rag_chain.invoke({"input": msg})
    print("Response:", response["answer"])
    return str(response["answer"])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
