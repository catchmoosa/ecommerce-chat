import os
import logging
from typing import Dict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from services.product_service import parse_product_data

# Set up logging
logger = logging.getLogger(__name__)

# Global variables
_VECTOR_STORE = None  # Cache for vector store

# Initialize vector store
def init_vector_store() -> FAISS:
    global _VECTOR_STORE
    
    # If vector store is already initialized, return it
    if _VECTOR_STORE is not None:
        logger.info("Using cached vector store")
        return _VECTOR_STORE
    
    try:
        # Load knowledge base from file
        kb_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "kb_content.txt")
        logger.info(f"Loading knowledge base from: {kb_path}")
        
        with open(kb_path, "r") as f:
            kb_content = f.read()
        
        # Parse product data from knowledge base
        parse_product_data(kb_content)
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        documents = text_splitter.create_documents([kb_content])
        logger.info(f"Created {len(documents)} document chunks")
        
        # Create embedding model
        embeddings = OpenAIEmbeddings()
        
        # Create and return vector store
        _VECTOR_STORE = FAISS.from_documents(documents, embeddings)
        logger.info("Vector store initialized successfully")
        return _VECTOR_STORE
    except Exception as e:
        logger.error(f"Error initializing vector store: {e}")
        raise

# Setup RAG chain
def setup_rag_chain(vector_store: FAISS):
    try:
        # Define prompt with conversation history context and product database access
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful customer support AI for an e-commerce store.

        Answer the customer's question based on the following context and product information:
        
        Retrieved context from knowledge base:
        {context}
        
        Prior conversation history:
        {conversation_history}
        
        Always check both the context AND the full product catalog before saying a product doesn't exist.
        If a specific product isn't found in the context but might exist in our catalog, mention similar products.
        
        When asked about specific product features (like screen size, price range, etc.):
        1. Check retrieved context first
        2. Use previous conversation context to understand user's preferences
        3. If the exact match isn't found, suggest the closest alternatives
        
        Be concise and friendly in your response.
        
        IMPORTANT: Our store has many products not fully described in the context, so never say we don't have a certain type of product if it's a common category.
        
        If the customer wants to add a product to their cart, extract the product name and quantity.
        If the customer asks about what's in their cart, explain that they can click the cart icon in the top right.
        
        Maintain a conversational tone and reference previous parts of the conversation when relevant.
        
        Customer question: {input}
        """)
        
        # Create LLM with higher temperature for more engaging responses
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)
        
        # Create document chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        
        # Create retrieval chain with more documents and higher similarity threshold
        # Increase k to retrieve more potential matches
        retriever = vector_store.as_retriever(search_kwargs={"k": 6, "fetch_k": 10, "search_type": "similarity"})
        rag_chain = create_retrieval_chain(retriever, document_chain)
        
        logger.info("RAG chain setup complete")
        return rag_chain
    except Exception as e:
        logger.error(f"Error setting up RAG chain: {e}")
        raise