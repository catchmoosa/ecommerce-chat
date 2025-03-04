import logging
from typing import Dict, List, Union
import chainlit as cl
from chainlit.action import Action

# Set up logging
logger = logging.getLogger(__name__)

def log_cart_contents(user_cart: List[Dict[str, Union[str, int, Dict]]]):
    """Log cart contents to terminal"""
    # Debug cart reference and type
    logger.info(f"Cart type: {type(user_cart)}, Cart ID: {id(user_cart)}")
    logger.info(f"Raw cart contents: {user_cart}")
    
    if not user_cart:
        logger.info("🛒 CART: Empty")
        return
    
    # Create a cart display message for terminal
    total_price = 0
    cart_lines = ["🛒 CART CONTENTS:"]
    
    for i, item in enumerate(user_cart):
        try:
            # Debug each item
            logger.info(f"Processing cart item {i}: {item}")
            
            price_str = item['details'].get('price', '$0')
            # Remove $ and any other non-numeric characters (like 'pair')
            price = float(''.join([c for c in price_str if c.isdigit() or c == '.']))
            item_total = price * item['quantity']
            total_price += item_total
            
            cart_lines.append(f"  - {item['quantity']}x {item['name']} - {price_str} (Item total: ${item_total:.2f})")
        except Exception as e:
            logger.error(f"Error calculating price for item {i}: {e}")
            logger.error(f"Item that caused error: {item}")
            cart_lines.append(f"  - Error with item {i} - Price calculation error")
    
    cart_lines.append(f"  TOTAL: ${total_price:.2f}")
    cart_lines.append("-" * 40)  # Add a separator line
    
    # Log to terminal
    for line in cart_lines:
        logger.info(line)

async def show_cart(user_cart: List[Dict[str, Union[str, int, Dict]]]):
    """Display the cart contents in a message"""
    # Debug cart reference and content
    logger.info(f"show_cart received cart: ID={id(user_cart)}, content={user_cart}")
    
    if not user_cart:
        logger.info("show_cart detected empty cart")
        await cl.Message(content="Your shopping cart is empty.").send()
        return
    
    # Create a cart display message
    total_price = 0
    cart_content = "# Shopping Cart\n\n"
    
    for i, item in enumerate(user_cart):
        try:
            logger.info(f"show_cart processing item {i}: {item}")
            
            price_str = item['details'].get('price', '$0')
            # Remove $ and any other non-numeric characters (like 'pair')
            price = float(''.join([c for c in price_str if c.isdigit() or c == '.']))
            item_total = price * item['quantity']
            total_price += item_total
            
            cart_content += f"- {item['quantity']}x **{item['name']}** - {price_str} (Item total: ${item_total:.2f})\n"
        except Exception as e:
            logger.error(f"Error in show_cart calculating price for item {i}: {e}")
            logger.error(f"Item that caused error: {item}")
            cart_content += f"- Error with item {i} - Price calculation error\n"
    
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
    
    # Also log to terminal
    logger.info("Logging cart from show_cart function:")
    log_cart_contents(user_cart)
    
    logger.info(f"Sending cart message with content: {cart_content}")
    await cl.Message(content=cart_content, actions=[checkout_action, clear_action]).send()