from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

raw_documents = TextLoader("combined_information.csv").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
documents = text_splitter.split_documents(raw_documents)


embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"

db_info = Chroma.from_documents(documents, embedding=HuggingFaceEmbeddings(model_name=embeddings_model_name))


template = """
You are a medical information specialist providing brief answers about brain tumors.

Use the following pieces of context to answer the question. The context contains important medical information
about symptoms, diagnoses, and treatments for different types of brain tumors.

Context: {context}

Question: {question}

Detailed Answer:
"""
custom_prompt = PromptTemplate.from_template(template)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
# Ensure API Key is loaded
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")

# Initialize Gemini-1.5-Flash Model with increased max_output_tokens
gemini_chat_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
    max_output_tokens=1024,  # Adjust this value as needed for longer responses
    api_key=api_key
)

# Create Retrieval-based Q&A Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=gemini_chat_model,
    retriever=db_info.as_retriever(search_kwargs={"k": 5}),  # Retrieve more documents
    return_source_documents=False,  # No need to return source documents
    chain_type="stuff",  # Use stuff chain to include all context
    chain_type_kwargs={
        "prompt": custom_prompt,
    }
)

# Function to Query Gemini AI with Retrieved Info
def query_gemini(question):
    # Add a system instruction to further enhance the response quality
    enhanced_question = question

    # Get response
    response = qa_chain.invoke(enhanced_question)

    # Just return the result directly
    return response["result"]


