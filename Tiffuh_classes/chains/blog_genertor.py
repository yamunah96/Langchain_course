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

prompt=PromptTemplate.from_template(
    "Create a blog about the topic {topic}, and blog suppose to be short for an instgram post with 10 lines, make intersting and viral"
)
# blog suppose to be short for an instgram post with 10 lines, make intersting and viral
parser=  StrOutputParser()
chain= prompt | llm| parser
result= chain.invoke({"topic":"AI agents in medical field "})
print(result)
