from datetime import date

# Get today's date
today = date.today()

# Format the date as desired, here we use the same format as in the original string
formatted_date = today.strftime("%B %d, %Y")

# Modify the prompt to include the dynamic date
eodhd_and_financial_toolset_prompt = f"""
You are an advanced chatbot called Shopper. Your role is to ubnderstand what the customer wants to purchase, and seklect whatever he wants to purchase.
"""
