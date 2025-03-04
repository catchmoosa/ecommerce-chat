# E-Commerce Customer Support Bot

A customer support AI agent for e-commerce using Retrieval Augmented Generation (RAG) with LangChain and Chainlit.

## Features

- Product search and recommendations
- Shopping cart functionality
- Product catalog with structured data
- Conversation history for context awareness
- RAG-powered product knowledge
- Beautiful web interface with Chainlit

## Project Structure

The code has been modularized into the following structure:

```
├── app.py                     # Main application entry point
├── services/                  # Core services
│   ├── __init__.py
│   ├── rag_service.py         # RAG and vector store functionality
│   └── product_service.py     # Product catalog and search functionality
├── utils/                     # Utility modules
│   ├── __init__.py
│   ├── history_manager.py     # Conversation history management
│   └── cart_manager.py        # Shopping cart display and management
├── kb_content.txt             # Knowledge base content (product catalog)
├── user_history/              # User session history storage
└── requirements.txt           # Project dependencies
```

## Requirements

- Python 3.10+
- OpenAI API key

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv astudio && source astudio/bin/activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Set your OpenAI API key:
   ```
   export OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Run the application:
   ```
   python app.py
   ```
2. Open your browser at http://localhost:8000

The chatbot provides the following functionality:

- Search for products by attributes (screen size, price, etc.)
- Add products to cart
- View cart contents
- Checkout
- Clear cart
- Ask questions about products

## Knowledge Base

The knowledge base is stored in `kb_content.txt`. You can modify this file to include your own product information, policies, and FAQs. The current format organizes products by categories (laptops, phones, watches, earphones, speakers).

## Implementation Details

- Uses FAISS for vector storage and similarity search
- Implements RAG pattern with LangChain
- Caches vector store to improve response time 
- Uses Chainlit for UI components like buttons and actions
- Maintains conversation history for better context awareness