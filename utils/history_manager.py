import logging
from typing import Dict, List

# Set up logging
logger = logging.getLogger(__name__)

# Global variables
CONVERSATION_HISTORY = {}  # Dictionary to store conversation history by session_id

def get_or_create_history(session_id: str) -> List[Dict]:
    """Get conversation history for a session, or create one if it doesn't exist"""
    # Initialize conversation history if needed
    if session_id not in CONVERSATION_HISTORY:
        CONVERSATION_HISTORY[session_id] = []
    
    return CONVERSATION_HISTORY[session_id]

def format_conversation_history(history: List[Dict]) -> str:
    """Format conversation history to be used in prompts"""
    formatted = ""
    for item in history:
        role = item.get("role", "")
        content = item.get("content", "")
        if role and content:
            formatted += f"{role.capitalize()}: {content}\n"
    return formatted