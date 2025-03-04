# E-Commerce Customer Support AI

A customer support AI agent for e-commerce using Retrieval Augmented Generation (RAG) with LangChain and Chainlit.

## Features

- Answers common e-commerce questions using a knowledge base
- Uses vector embeddings for accurate information retrieval
- Cites information sources in responses
- Beautiful web interface with Chainlit
- Handles questions about products, shipping, returns, warranties, and more

## Requirements

- Python 3.10+
- OpenAI API key

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set your OpenAI API key:
   ```
   export OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Run the application:
   ```
   python customer_support_agent.py
   ```
2. Open your browser at http://localhost:8000

## Knowledge Base

The knowledge base is stored in `kb_content.txt`. You can modify this file to include your own product information, policies, and FAQs.

## Implementation Details

- Uses FAISS for vector storage and similarity search
- Implements RAG pattern with LangChain
- Caches vector store to improve response time
- Provides source citations for transparency