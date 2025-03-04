import logging
import re
from typing import Dict, List, Optional, Union

# Set up logging
logger = logging.getLogger(__name__)

# Global variables
PRODUCTS = {}  # Dictionary to store all product information

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

def extract_product_request(message: str) -> Optional[Dict[str, Union[str, int]]]:
    """Extract product name and quantity from a user message"""
    # Simple checks for common shopping phrases
    add_phrases = ["add", "buy", "purchase", "get", "want", "put in cart", "add to cart"]
    
    # Log for debugging
    logger.info(f"Checking if message contains a product request: '{message}'")
    logger.info(f"PRODUCTS dictionary contains {sum(len(category) for category in PRODUCTS.values())} products")
    
    # Log the available product names
    product_names = []
    for category, products in PRODUCTS.items():
        for product_name in products.keys():
            product_names.append(f"{product_name} ({category})")
    logger.info(f"Available products: {product_names}")
    
    for phrase in add_phrases:
        if phrase in message.lower():
            logger.info(f"Found shopping phrase: '{phrase}'")
            # Now look through our products to see if any are mentioned
            for category, products in PRODUCTS.items():
                for product_name, details in products.items():
                    product_lower = product_name.lower()
                    message_lower = message.lower()
                    
                    if product_lower in message_lower:
                        # Try to extract quantity - default to 1
                        quantity = 1
                        # Simple number extraction
                        for word in message_lower.split():
                            if word.isdigit():
                                quantity = int(word)
                                break
                        
                        logger.info(f"Found product match: {product_name}, quantity: {quantity}")
                        
                        return {
                            "name": product_name,
                            "details": details,
                            "quantity": quantity
                        }
    
    logger.info("No product match found in message")
    return None

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