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
import requests
from typing import Annotated

from langchain_community.tools import DuckDuckGoSearchRun,ShellTool
from langchain_core.tools import InjectedToolArg

llm= ChatOpenAI(
    model="openai/gpt-oss-120b",
    api_key=os.getenv('GROQ_API_KEY'),
    base_url= "https://api.groq.com/openai/v1",
    max_tokens= 1000
)

@tool
def get_conversion_factor(base_cur:str,target_cur:str)->float:
    '''This function fetches the currency converstions factor btw a given base and target_cur'''
    url=f'https://v6.exchangerate-api.com/v6/00a881590c1276a60f550e8b/pair/{base_cur}/{target_cur}'
    # respones= requests.get(url)
    return requests.get(url).json()   # returns full dict; we'll pull the rate below
# print(get_conversion_factor.invoke({'base_cur':'USD','target_cur':'INR'}))

@tool
def convert(base_cur:int,conversion_rate:Annotated[float,InjectedToolArg])->float:
    '''Given a currency conversion rate this function calculates the target currency value'''
    return base_cur*conversion_rate
# print(convert.invoke({"base_cur":10,"conversion_rate":95.2}))


# tool blindig
llm_with_tools= llm.bind_tools([get_conversion_factor,convert])
messages=[HumanMessage("What is the conversion factor btw INR and USD,and based on thate can you convert 10 USD to INR")]



ai_message = llm_with_tools.invoke(messages)
# print(ai_message.content)
messages.append(ai_message)   # IMPORTANT: keep the AIMessage in history

conversion_rate = None
#  Step 2: execute each requested tool
for call in ai_message.tool_calls:
    if call["name"] == "get_conversion_factor":
        result = get_conversion_factor.invoke(call)        # returns a ToolMessage
        conversion_rate = result.content if isinstance(result.content, (int, float)) \
                          else eval(result.content)["conversion_rate"]  # dict-as-str
        messages.append(result)

    elif call["name"] == "convert":
        # inject the rate the LLM was forbidden from supplying
        call["args"]["conversion_rate"] = conversion_rate
        result = convert.invoke(call)
        messages.append(result)

# Step 3: send results back; NOW content is populated
final = llm_with_tools.invoke(messages)
print(final.content)

# Memory, Youtube transcipt agent,chatbot ()