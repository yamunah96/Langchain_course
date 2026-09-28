from dotenv import load_dotenv
import os
load_dotenv()


from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain_core.runnables import RunnableLambda,RunnableParallel

llm= ChatOpenAI(
    model="openai/gpt-oss-120b",
    api_key=os.getenv('GROQ_API_KEY'),
    base_url= "https://api.groq.com/openai/v1",
    max_tokens= 1000
)
loader= PyPDFLoader('./Expense_Tracker_Coding_Guide.pdf')
docs= loader.load()

# print(len(docs))
# print(docs[11].page_content)
# print(docs[11].metadata)

prompt1= PromptTemplate(
    template="summary of the page in 1 line {pagecontent}",
    input_variables=['pagecontent']
)
parser= StrOutputParser()
chain= prompt1 | llm | parser

result= chain.invoke({
    'pagecontent':docs[11].page_content,
})
print(result)