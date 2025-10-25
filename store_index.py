from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec



load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("Data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


index_name="medicalbot-index"

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medicalbot-index"

pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ),
)

docsearch = PineconeVectorStore.from_documents(
    text_chunks,
    embeddings,
    index_name=index_name,
)