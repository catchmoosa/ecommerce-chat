import os
import logging
import json
from typing import List, Dict, Optional, Union
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import chainlit as cl
from chainlit.element import Image
from chainlit.action import Action

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
_VECTOR_STORE = None  # Cache for vector store
PRODUCTS = {}  # Dictionary to store all product information
CONVERSATION_HISTORY = {}  # Dictionary to store conversation history by session_id

# Initialize vector store
def init_vector_store() -> FAISS:
    global _VECTOR_STORE, PRODUCTS
    
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

def parse_product_data(kb_content: str):
    """Parse product data from kb_content to store structured product information"""
    global PRODUCTS
    
    # Initialize categories
    PRODUCTS = {
        "laptops": {},
        "phones": {},
        "watches": {},
        "earphones": {},
        "speakers": {}
    }
    
    lines = kb_content.split('\n')
    current_category = None
    current_product = None
    product_details = {}
    
    for line in lines:
        line = line.strip()
        
        # Check for category headers
        if line.startswith('## LAPTOPS'):
            current_category = "laptops"
        elif line.startswith('## PHONES'):
            current_category = "phones"
        elif line.startswith('## WATCHES'):
            current_category = "watches"
        elif line.startswith('## EARPHONES'):
            current_category = "earphones"
        elif line.startswith('## SPEAKERS'):
            current_category = "speakers"
        
        # Check for product names
        elif line.startswith('### ') and current_category:
            if current_product and product_details:
                PRODUCTS[current_category][current_product] = product_details
            
            current_product = line.replace('### ', '')
            product_details = {"name": current_product}
        
        # Parse product details
        elif line.startswith('- **') and current_product:
            try:
                key_value = line.replace('- **', '').split(':** ')
                key = key_value[0].lower().replace(' ', '_')
                value = key_value[1]
                product_details[key] = value
            except Exception:
                pass
    
    # Add the last product
    if current_product and product_details and current_category:
        PRODUCTS[current_category][current_product] = product_details
    
    logger.info(f"Parsed {sum(len(category) for category in PRODUCTS.values())} products from knowledge base")

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

def extract_product_request(message: str) -> Optional[Dict[str, Union[str, int]]]:
    """Extract product name and quantity from a user message"""
    # Simple checks for common shopping phrases
    add_phrases = ["add", "buy", "purchase", "get", "want", "put in cart", "add to cart"]
    
    for phrase in add_phrases:
        if phrase in message.lower():
            # Now look through our products to see if any are mentioned
            for category in PRODUCTS.values():
                for product_name, details in category.items():
                    if product_name.lower() in message.lower():
                        # Try to extract quantity - default to 1
                        quantity = 1
                        # Simple number extraction
                        for word in message.lower().split():
                            if word.isdigit():
                                quantity = int(word)
                                break
                        
                        return {
                            "name": product_name,
                            "details": details,
                            "quantity": quantity
                        }
    
    return None

# Format conversation history to be used in prompts
def format_conversation_history(history):
    formatted = ""
    for item in history:
        role = item.get("role", "")
        content = item.get("content", "")
        if role and content:
            formatted += f"{role.capitalize()}: {content}\n"
    return formatted

async def show_cart(user_cart: List[Dict[str, Union[str, int, Dict]]]):
    """Display the cart contents in a message"""
    if not user_cart:
        await cl.Message(content="Your shopping cart is empty.").send()
        return
    
    # Create a cart display message
    total_price = 0
    cart_content = "# Shopping Cart\n\n"
    
    for item in user_cart:
        try:
            price_str = item['details'].get('price', '$0')
            # Remove $ and any other non-numeric characters (like 'pair')
            price = float(''.join([c for c in price_str if c.isdigit() or c == '.']))
            item_total = price * item['quantity']
            total_price += item_total
            
            cart_content += f"- {item['quantity']}x **{item['name']}** - {price_str} (Item total: ${item_total:.2f})\n"
        except Exception as e:
            logger.error(f"Error calculating price for {item['name']}: {e}")
            cart_content += f"- {item['quantity']}x **{item['name']}** - Price calculation error\n"
    
    cart_content += f"\n**Total: ${total_price:.2f}**"
    
    # Add Checkout button
    checkout_action = Action(
        name="checkout",
        value="checkout",
        label="Proceed to Checkout",
        description="Complete your purchase",
        payload={}
    )
    
    # Add Clear Cart button
    clear_action = Action(
        name="clear_cart",
        value="clear",
        label="Clear Cart",
        description="Remove all items from your cart",
        payload={}
    )
    
    await cl.Message(content=cart_content, actions=[checkout_action, clear_action]).send()

# Initialize on app start
@cl.on_chat_start
async def on_chat_start():
    try:
        logger.info("New chat session started")
        session_id = cl.user_session.get("session_id")
        
        # Generate a unique session ID if one doesn't exist
        if not session_id:
            session_id = str(hash(str(cl.user_session)))
            cl.user_session.set("session_id", session_id)
            
        # Initialize conversation history for this session
        if session_id not in CONVERSATION_HISTORY:
            CONVERSATION_HISTORY[session_id] = []
        
        # Initialize the vector store and RAG chain
        vector_store = init_vector_store()
        rag_chain = setup_rag_chain(vector_store)
        
        # Initialize cart
        cl.user_session.set("cart", [])
        
        # Store in user session
        cl.user_session.set("rag_chain", rag_chain)
        
        # Add shopping cart button to sidebar
        cart_action = Action(
            name="view_cart",
            value="view",
            label="🛒 Cart",
            description="View your shopping cart",
            payload={}
        )
        
        greeting = "👋 Hello! I'm your e-commerce support assistant. How can I help you today?"
        
        # Add greeting to conversation history
        CONVERSATION_HISTORY[session_id].append({"role": "assistant", "content": greeting})
        
        # Store conversation history in session
        cl.user_session.set("conversation_history", CONVERSATION_HISTORY[session_id])
        
        await cl.Message(
            content=greeting,
            actions=[cart_action]
        ).send()
    except Exception as e:
        logger.error(f"Error in chat initialization: {e}")
        await cl.Message(
            content="Sorry, I'm having trouble starting up. Please try again in a moment."
        ).send()

@cl.action_callback("view_cart")
async def on_view_cart(action):
    """Handle view cart button click"""
    session_id = cl.user_session.get("session_id")
    
    # Get conversation history
    if session_id and session_id in CONVERSATION_HISTORY:
        conversation_history = CONVERSATION_HISTORY[session_id]
    else:
        # Create new session ID if needed
        session_id = str(hash(str(cl.user_session)))
        cl.user_session.set("session_id", session_id)
        CONVERSATION_HISTORY[session_id] = []
        conversation_history = CONVERSATION_HISTORY[session_id]
    
    cart = cl.user_session.get("cart", [])
    await show_cart(cart)
    
    # Add system message to history
    conversation_history.append({"role": "system", "content": "User viewed their cart"})
    CONVERSATION_HISTORY[session_id] = conversation_history
    cl.user_session.set("conversation_history", conversation_history)

@cl.action_callback("checkout")
async def on_checkout(action):
    """Handle checkout button click"""
    session_id = cl.user_session.get("session_id")
    
    # Get conversation history
    if session_id and session_id in CONVERSATION_HISTORY:
        conversation_history = CONVERSATION_HISTORY[session_id]
    else:
        session_id = str(hash(str(cl.user_session)))
        cl.user_session.set("session_id", session_id)
        CONVERSATION_HISTORY[session_id] = []
        conversation_history = CONVERSATION_HISTORY[session_id]
    
    # In a real app, this would redirect to a checkout page
    checkout_message = "Thank you for your order! In a real application, you would now be redirected to a payment page."
    await cl.Message(content=checkout_message).send()
    
    # Add system and assistant messages to history
    conversation_history.append({"role": "system", "content": "User completed checkout"})
    conversation_history.append({"role": "assistant", "content": checkout_message})
    
    # Clear the cart after checkout
    cl.user_session.set("cart", [])
    
    # Update conversation history
    CONVERSATION_HISTORY[session_id] = conversation_history
    cl.user_session.set("conversation_history", conversation_history)

@cl.action_callback("clear_cart")
async def on_clear_cart(action):
    """Handle clear cart button click"""
    session_id = cl.user_session.get("session_id")
    
    # Get conversation history
    if session_id and session_id in CONVERSATION_HISTORY:
        conversation_history = CONVERSATION_HISTORY[session_id]
    else:
        session_id = str(hash(str(cl.user_session)))
        cl.user_session.set("session_id", session_id)
        CONVERSATION_HISTORY[session_id] = []
        conversation_history = CONVERSATION_HISTORY[session_id]
    
    cl.user_session.set("cart", [])
    clear_message = "Your cart has been cleared."
    await cl.Message(content=clear_message).send()
    
    # Add system and assistant messages to history
    conversation_history.append({"role": "system", "content": "User cleared their cart"})
    conversation_history.append({"role": "assistant", "content": clear_message})
    
    # Update conversation history
    CONVERSATION_HISTORY[session_id] = conversation_history
    cl.user_session.set("conversation_history", conversation_history)


# Helper function to search products by attributes
def search_products_by_attribute(query_text: str) -> List[Dict]:
    """
    Search for products by various attributes like size, price, brand, etc.
    Returns a list of matching product details.
    """
    query = query_text.lower()
    results = []
    
    # Extract potential search attributes
    screen_size_match = None
    price_range = None
    
    # Check for screen size
    import re
    size_patterns = [
        r'(\d+\.?\d*)\s*inch', 
        r'(\d+\.?\d*)"',
        r'(\d+\.?\d*)-inch'
    ]
    
    for pattern in size_patterns:
        match = re.search(pattern, query)
        if match:
            screen_size_match = float(match.group(1))
            break
    
    # Check for price range
    price_under_match = re.search(r'under\s*\$?(\d+)', query)
    price_over_match = re.search(r'over\s*\$?(\d+)', query)
    price_between_match = re.search(r'\$?(\d+)\s*-\s*\$?(\d+)', query)
    
    if price_under_match:
        price_range = (0, int(price_under_match.group(1)))
    elif price_over_match:
        price_range = (int(price_over_match.group(1)), float('inf'))
    elif price_between_match:
        price_range = (int(price_between_match.group(1)), int(price_between_match.group(2)))
    
    # Determine category
    category_keywords = {
        "laptops": ["laptop", "notebook", "macbook", "gaming laptop", "ultrabook"],
        "phones": ["phone", "smartphone", "mobile", "iphone", "android"],
        "watches": ["watch", "smartwatch", "fitness tracker"],
        "earphones": ["earphone", "headphone", "earbud", "airpod", "earbuds"],
        "speakers": ["speaker", "sound bar", "audio", "soundbar"]
    }
    
    target_categories = []
    for category, keywords in category_keywords.items():
        if any(keyword in query for keyword in keywords):
            target_categories.append(category)
    
    # If no specific category is mentioned, search all
    if not target_categories:
        target_categories = list(PRODUCTS.keys())
    
    # Search through products
    for category in target_categories:
        for product_name, details in PRODUCTS[category].items():
            match = True
            
            # Check screen size if specified
            if screen_size_match and category in ["laptops", "phones"]:
                try:
                    size_str = details.get("display", "")
                    size_match = re.search(r'(\d+\.?\d*)"', size_str) or re.search(r'(\d+\.?\d*) inch', size_str)
                    if size_match:
                        product_size = float(size_match.group(1))
                        # Allow some flexibility (±1 inch)
                        if abs(product_size - screen_size_match) > 1:
                            match = False
                    else:
                        match = False
                except (ValueError, AttributeError):
                    match = False
            
            # Check price range if specified
            if price_range and match:
                try:
                    price_str = details.get("price", "$0")
                    # Extract numeric part of price
                    price = float(''.join([c for c in price_str if c.isdigit() or c == '.']))
                    if price < price_range[0] or price > price_range[1]:
                        match = False
                except (ValueError, TypeError):
                    match = False
            
            # If all criteria match, add to results
            if match:
                results.append({
                    "name": product_name,
                    "details": details,
                    "category": category
                })
    
    return results

# Handle incoming messages
@cl.on_message
async def on_message(message: cl.Message):
    try:
        logger.info(f"Received message: {message.content}")
        
        # Get session ID and conversation history
        session_id = cl.user_session.get("session_id")
        if not session_id:
            session_id = str(hash(str(cl.user_session)))
            cl.user_session.set("session_id", session_id)
        
        # Initialize conversation history if needed
        if session_id not in CONVERSATION_HISTORY:
            CONVERSATION_HISTORY[session_id] = []
        
        # Get history from global store
        conversation_history = CONVERSATION_HISTORY[session_id]
        
        # Get RAG chain and cart from user session
        rag_chain = cl.user_session.get("rag_chain")
        cart = cl.user_session.get("cart", [])
        
        # Add user message to conversation history
        conversation_history.append({"role": "user", "content": message.content})
        
        # Check if the message is about the cart
        show_cart_phrases = ["show cart", "view cart", "my cart", "what's in my cart", "checkout"]
        if any(phrase in message.content.lower() for phrase in show_cart_phrases):
            await show_cart(cart)
            # Add system message to conversation history
            conversation_history.append({"role": "system", "content": "Cart contents displayed"})
            return
        
        # Check if the message is a product request
        product_request = extract_product_request(message.content)
        
        # Show typing indicator
        thinking_msg = cl.Message(content="")
        await thinking_msg.send()
        
        if product_request:
            # Add to cart
            cart.append(product_request)
            cl.user_session.set("cart", cart)
            
            # Respond with confirmation
            response = f"Added {product_request['quantity']}x {product_request['name']} to your cart."
            
            # Add to conversation history
            conversation_history.append({"role": "assistant", "content": response})
            
            # Add cart view button
            cart_action = Action(
                name="view_cart",
                value="view",
                label="View Cart",
                description="View your shopping cart",
                payload={}
            )
            
            thinking_msg.content = response
            thinking_msg.actions = [cart_action]
            await thinking_msg.update()
            
            logger.info(f"Added product to cart: {product_request['name']}")
        else:
            # Format conversation history for the prompt
            formatted_history = format_conversation_history(conversation_history[-10:] if len(conversation_history) > 10 else conversation_history)
            
            # Check if it's a product search query first
            additional_context = ""
            
            # Check for specific product attribute search
            search_phrases = ["looking for", "find", "search", "show", "display", "inch", "price", "under", "over"]
            attribute_search = any(phrase in message.content.lower() for phrase in search_phrases)
            
            if attribute_search:
                # Search for products by attributes (size, price, etc.)
                product_results = search_products_by_attribute(message.content)
                
                if product_results:
                    # Format matched products as additional context
                    additional_context = "Found matching products based on your query:\n\n"
                    for i, product in enumerate(product_results[:5], 1):  # Limit to top 5
                        additional_context += f"{i}. {product['name']} ({product['category']}): "
                        for key, value in product['details'].items():
                            if key in ['price', 'display', 'processor', 'ram', 'storage', 'graphics', 'battery']:
                                additional_context += f"{key.replace('_', ' ').title()}: {value}, "
                        additional_context = additional_context.rstrip(", ") + "\n"
            
            # Process the message with RAG chain, including any extra product search results
            enriched_input = message.content
            if additional_context:
                enriched_input = f"{message.content}\n\nAdditional product information:\n{additional_context}"
                
            response = rag_chain.invoke({
                "input": enriched_input,
                "conversation_history": formatted_history
            })
            answer = response["answer"]
            
            # Add to conversation history
            conversation_history.append({"role": "assistant", "content": answer})
            
            # Add cart button
            cart_action = Action(
                name="view_cart",
                value="view",
                label="🛒 Cart",
                description="View your shopping cart",
                payload={}
            )
            
            thinking_msg.content = answer
            thinking_msg.actions = [cart_action]
            await thinking_msg.update()
            
            logger.info(f"Sent response: {answer[:100]}...")
        
        # Update conversation history in global store and session
        CONVERSATION_HISTORY[session_id] = conversation_history
        cl.user_session.set("conversation_history", conversation_history)
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        await cl.Message(
            content="Sorry, I encountered an error while processing your request. Please try again."
        ).send()