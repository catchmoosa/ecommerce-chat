import logging
import chainlit as cl
from chainlit.element import Image
from chainlit.action import Action

from services.rag_service import init_vector_store, setup_rag_chain
from services.product_service import parse_product_data, search_products_by_attribute, extract_product_request
from utils.history_manager import get_or_create_history, format_conversation_history, CONVERSATION_HISTORY
from utils.cart_manager import show_cart

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize on app start
@cl.on_chat_start
async def on_chat_start():
    try:
        logger.info("============= NEW CHAT SESSION STARTED =============")
        session_id = cl.user_session.get("session_id")
        
        # Generate a unique session ID if one doesn't exist
        if not session_id:
            session_id = str(hash(str(cl.user_session)))
            cl.user_session.set("session_id", session_id)
            logger.info(f"Generated new session ID: {session_id}")
        else:
            logger.info(f"Using existing session ID: {session_id}")
            
        # Initialize conversation history for this session
        if session_id not in CONVERSATION_HISTORY:
            CONVERSATION_HISTORY[session_id] = []
            logger.info(f"Initialized new conversation history for session {session_id}")
        else:
            logger.info(f"Using existing conversation history for session {session_id}")
        
        # Initialize the vector store and RAG chain
        vector_store = init_vector_store()
        rag_chain = setup_rag_chain(vector_store)
        logger.info("Vector store and RAG chain initialized")
        
        # Initialize cart
        cart = []
        cl.user_session.set("cart", cart)
        
        # Log empty cart to terminal
        from utils.cart_manager import log_cart_contents
        logger.info("Initializing new empty cart")
        log_cart_contents(cart)
        
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
        
        # Verify cart was properly initialized
        session_cart = cl.user_session.get("cart", None)
        logger.info(f"Cart after initialization: {session_cart}, Type: {type(session_cart)}")
        
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
    conversation_history = get_or_create_history(session_id)
    
    # Get cart with debug info
    cart = cl.user_session.get("cart", [])
    logger.info(f"VIEW CART: Retrieved cart from session: {cart}")
    
    # Log view cart action to terminal
    logger.info("User clicked View Cart button")
    from utils.cart_manager import log_cart_contents
    log_cart_contents(cart)
    
    # Use a copy of the cart to avoid reference issues
    await show_cart(cart.copy() if cart else [])
    
    # Add system message to history
    conversation_history.append({"role": "system", "content": "User viewed their cart"})
    CONVERSATION_HISTORY[session_id] = conversation_history
    cl.user_session.set("conversation_history", conversation_history)

@cl.action_callback("checkout")
async def on_checkout(action):
    """Handle checkout button click"""
    session_id = cl.user_session.get("session_id")
    
    # Get conversation history
    conversation_history = get_or_create_history(session_id)
    
    # Get cart before clearing it
    cart = cl.user_session.get("cart", [])
    
    # Log checkout to terminal
    logger.info("User clicked Checkout button")
    from utils.cart_manager import log_cart_contents
    logger.info("CHECKOUT - Processing cart:")
    log_cart_contents(cart)
    
    # In a real app, this would redirect to a checkout page
    checkout_message = "Thank you for your order! In a real application, you would now be redirected to a payment page."
    await cl.Message(content=checkout_message).send()
    
    # Add system and assistant messages to history
    conversation_history.append({"role": "system", "content": "User completed checkout"})
    conversation_history.append({"role": "assistant", "content": checkout_message})
    
    # Clear the cart after checkout
    cl.user_session.set("cart", [])
    logger.info("Cart cleared after checkout")
    
    # Update conversation history
    CONVERSATION_HISTORY[session_id] = conversation_history
    cl.user_session.set("conversation_history", conversation_history)

@cl.action_callback("clear_cart")
async def on_clear_cart(action):
    """Handle clear cart button click"""
    session_id = cl.user_session.get("session_id")
    
    # Get conversation history
    conversation_history = get_or_create_history(session_id)
    
    # Get cart contents before clearing
    cart = cl.user_session.get("cart", [])
    
    # Log clear cart action to terminal
    logger.info("User clicked Clear Cart button")
    from utils.cart_manager import log_cart_contents
    logger.info("Clearing cart with contents:")
    log_cart_contents(cart)
    
    # Clear the cart
    cl.user_session.set("cart", [])
    clear_message = "Your cart has been cleared."
    await cl.Message(content=clear_message).send()
    
    # Add system and assistant messages to history
    conversation_history.append({"role": "system", "content": "User cleared their cart"})
    conversation_history.append({"role": "assistant", "content": clear_message})
    
    # Update conversation history
    CONVERSATION_HISTORY[session_id] = conversation_history
    cl.user_session.set("conversation_history", conversation_history)
    
    # Log empty cart
    logger.info("Cart now empty")

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
        
        # Get history from global store
        conversation_history = get_or_create_history(session_id)
        
        # Get RAG chain and cart from user session
        rag_chain = cl.user_session.get("rag_chain")
        cart = cl.user_session.get("cart", [])
        
        # Log cart contents to terminal
        from utils.cart_manager import log_cart_contents
        log_cart_contents(cart)
        
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
            # Log product request before adding to cart
            logger.info(f"Product request found: {product_request}")
            
            # Debug cart before adding
            logger.info(f"Cart before adding: {cart}")
            
            # Add to cart - create a new list to ensure we're not dealing with references
            cart = cart.copy()
            cart.append(product_request)
            
            # Debug cart after adding
            logger.info(f"Cart after adding: {cart}")
            
            # Update session cart
            cl.user_session.set("cart", cart)
            
            # Log updated cart contents to terminal
            logger.info("Logging updated cart contents after adding product:")
            log_cart_contents(cart)
            
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