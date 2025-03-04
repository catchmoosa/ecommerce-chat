import os
import logging
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import chainlit as cl

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global cache for vector store
_VECTOR_STORE = None

# Initialize vector store
def init_vector_store() -> FAISS:
    global _VECTOR_STORE
    
    # If vector store is already initialized, return it
    if _VECTOR_STORE is not None:
        logger.info("Using cached vector store")
        return _VECTOR_STORE
    
    try:
        # Load knowledge base from file
        kb_path = os.path.join(os.path.dirname(__file__), "kb_content.txt")
        logger.info(f"Loading knowledge base from: {kb_path}")
        
        with open(kb_path, "r") as f:
            kb_content = f.read()
        
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
        # Define prompt
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful customer support AI for an e-commerce store.
        Answer the customer's question based on the following context:
        
        {context}
        
        If the information isn't in the context, politely say you don't have that information.
        Be concise and friendly in your response.
        
        Customer question: {input}
        """)
        
        # Create LLM
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)
        
        # Create document chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        
        # Create retrieval chain
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        rag_chain = create_retrieval_chain(retriever, document_chain)
        
        logger.info("RAG chain setup complete")
        return rag_chain
    except Exception as e:
        logger.error(f"Error setting up RAG chain: {e}")
        raise

# This function has been removed as we're not using source document citations
# in this version of LangChain

# Initialize on app start
@cl.on_chat_start
async def on_chat_start():
    try:
        logger.info("New chat session started")
        
        # Initialize the vector store and RAG chain
        vector_store = init_vector_store()
        rag_chain = setup_rag_chain(vector_store)
        
        # Store in user session
        cl.user_session.set("rag_chain", rag_chain)
        
        await cl.Message(
            content="👋 Hello! I'm your e-commerce support assistant. How can I help you today?"
        ).send()
    except Exception as e:
        logger.error(f"Error in chat initialization: {e}")
        await cl.Message(
            content="Sorry, I'm having trouble starting up. Please try again in a moment."
        ).send()

# Handle incoming messages
@cl.on_message
async def on_message(message: cl.Message):
    try:
        logger.info(f"Received message: {message.content}")
        
        # Get RAG chain from user session
        rag_chain = cl.user_session.get("rag_chain")
        
        # Show typing indicator
        thinking_msg = cl.Message(content="")
        await thinking_msg.send()
        
        # Process the message with RAG chain
        response = rag_chain.invoke({"input": message.content})
        answer = response["answer"]
        
        # No source documents available in this version of LangChain
        # Just use the answer directly
        full_response = answer
        
        logger.info("Response processed successfully")
        
        # Update message with response
        thinking_msg.content = full_response
        await thinking_msg.update()
        
        logger.info(f"Sent response: {answer[:100]}...")
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        await cl.Message(
            content="Sorry, I encountered an error while processing your request. Please try again."
        ).send()