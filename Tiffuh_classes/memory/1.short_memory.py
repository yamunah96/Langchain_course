# runnablewithMessagehistory
# 1. the messages gets stored in list formate- chatPrompatTemplate+ Messageholder


from dotenv import load_dotenv
import os
load_dotenv()
#  uv add -U ddgs

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain_core.runnables import RunnableLambda,RunnableParallel
from langchain.tools import tool
from langchain_core.messages import HumanMessage


from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory


from langchain_community.tools import DuckDuckGoSearchRun,ShellTool
parser=  StrOutputParser()


llm= ChatOpenAI(
    model="openai/gpt-oss-120b",
    api_key=os.getenv('GROQ_API_KEY'),
    base_url= "https://api.groq.com/openai/v1",
    max_tokens= 1000
)

prompt= ChatPromptTemplate.from_messages([
    ("system","Your are frienly coding tutor,Keep answers in shot form"),
    MessagesPlaceholder(variable_name="history"),
    ("human","{input}")
])
chain= prompt | llm 
store={}
def get_history(session_id:str)->InMemoryChatMessageHistory:
    # if session_id not in store:
    #     store[session_id]=InMemoryChatMessageHistory()
    # return store[session_id]
    return FileChatMessageHistory(f"chat_{session_id}.json")

# wrap the chain to a history +saved automatically

chat= RunnableWithMessageHistory(
    chain,get_session_history=get_history,input_messages_key="input",history_messages_key="history"
)

yamuna= {"configurable":{"session_id":"yamuna"}}
tiffuh= {"configurable":{"session_id":"tiffuh"}}
# result1= chat.invoke({"input":"hi my name is yamuna"},config=cfg)
# print(result1)


# result2= chat.invoke({"input":"Whats is my name"},config=cfg)
# print(result2)

chat.invoke({"input":"My fav langauge is python"},config=yamuna)
chat.invoke({"input":"My fav langauge is Java"},config=tiffuh)

print(chat.invoke({"input":"whats my fav langauge"},config=tiffuh).content)

# print(store["yamuna"])

# SqlIte_backeen,
