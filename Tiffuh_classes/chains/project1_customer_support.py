from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()


from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch,RunnableLambda
prompt = ChatPromptTemplate.from_template(
    """
    You are a customer support classifier.

    Classify the customer ticket into one category:

    - product info
    - return
    - cancellation
    - refund
    - technical issue

    Only return the category name.

    Customer Ticket:
    {ticket}
    """
)



llm= ChatOpenAI(
    model="openai/gpt-oss-120b",
    api_key=os.getenv('GROQ_API_KEY'),
    base_url= "https://api.groq.com/openai/v1",
    max_tokens= 1000
)

# Output Parser
parser = StrOutputParser()

# Chain
classification_chain = prompt | llm | parser

# Routing Function

def route_ticket(category):
    category = category.strip().lower()

    routes = {
        "product info": "Send to Product Information Team",
        "return": "Send to Returns Department",
        "cancellation": "Send to Cancellation Team",
        "refund": "Send to Refund Processing Team",
        "technical issue": "Send to Technical Support"
    }

    return routes.get(category, "Send to General Support")

# Runnable
router = RunnableLambda(route_ticket)

# Final Chain
final_chain = classification_chain | router

# Example Tickets

tickets = [
    "I want to return the shoes I ordered last week.",
    "Can you give more details about the laptop specifications?",
    "Please cancel my subscription immediately.",
    "My payment was deducted twice, I need a refund.",
    "The app crashes whenever I open it."
]

# Run
for ticket in tickets:
    result = final_chain.invoke({"ticket": ticket})

    print("Ticket:", ticket)
    print("Routing:", result)
    print("-" * 50)