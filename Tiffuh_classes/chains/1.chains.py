from dotenv import load_dotenv
import os
load_dotenv()


from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm= ChatOpenAI(
    model="openai/gpt-oss-120b",
    api_key=os.getenv('GROQ_API_KEY'),
    base_url= "https://api.groq.com/openai/v1",
    max_tokens= 1000
)

prompt= PromptTemplate(
    template="Generate 5 interseting facts about {topic}",
    input_variables=['topic']
)
# Generate 5 interseting facts about {quantum physics}--- python language(transalators)(parser- its convert the row text to python)
parser= StrOutputParser()
chain= prompt|llm|parser
result= chain.invoke({"topic":"football"})

print(result)
# chain.get_graph().print_ascii()