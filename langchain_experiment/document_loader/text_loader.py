from dotenv import load_dotenv
import os
load_dotenv()


from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_core.runnables import RunnableLambda
parser= StrOutputParser()

llm= ChatOpenAI(
    model="openai/gpt-oss-120b",
    api_key=os.getenv('GROQ_API_KEY'),
    base_url= "https://api.groq.com/openai/v1",
    max_tokens= 1000
)
promot1=  PromptTemplate(
    template="write short poem using above text file -\n {poem}",
    input_variables=['poem']
)
prompt2= PromptTemplate(
    template='Convert the following poem to {language}:\n{poem}',
    input_variables=['poem','language']
)
loader= TextLoader("data.txt",encoding='utf-8')
# print(loader)

docs= loader.load()
# print(docs)
# print(type(docs))
# print(docs[0].page_content)
print(docs[0].metadata)

format_input= RunnableLambda(
    lambda x:{
        "poem":x,
        "language": "Hindi"
    }
)
chain= promot1| llm | parser | format_input| prompt2 | llm | parser
result= chain.invoke({
    "poem":docs[0].page_content
})
print(result)