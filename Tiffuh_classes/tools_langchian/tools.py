from dotenv import load_dotenv
import os
load_dotenv()
#  uv add -U ddgs

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain_core.runnables import RunnableLambda,RunnableParallel
from langchain.tools import tool
from langchain_core.messages import HumanMessage


from langchain_community.tools import DuckDuckGoSearchRun,ShellTool

llm= ChatOpenAI(
    model="openai/gpt-oss-120b",
    api_key=os.getenv('GROQ_API_KEY'),
    base_url= "https://api.groq.com/openai/v1",
    max_tokens= 1000
)

# # Initialize the tool
# search_tool = DuckDuckGoSearchRun()

# # Use the tool directly
# results= search_tool.invoke("Top news in USA today")
# print(results)
# print(search_tool.name)
# print(search_tool.description )
# print(search_tool.args)


# custom_tools
@tool
def multiply(a:int,b:int)->int:
    """Take any two numbers and multiply"""
    return a*b

@tool
def percentage(number:float,per:float):
    """Take number,and percentage value calculate its amount after
      implementation of percentage Formula: Part = Percentage\100 * Whole"""
    return (per/100)*number
# result= multiply.invoke({"a":3,"b":10})
# print(result)
# print(multiply.name)
# print(multiply.description)
# print(multiply.args)

llm_with_tools= llm.bind_tools([multiply,percentage])
query= HumanMessage("Can you multiply 3 with 1000, and also calculate 20%70")

result= llm_with_tools.invoke([query])
print(result.content)