from dotenv import load_dotenv
import os
load_dotenv()


from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader,PyPDFLoader,WebBaseLoader
from langchain_core.runnables import RunnableLambda,RunnableParallel

llm= ChatOpenAI(
    model="openai/gpt-oss-120b",
    api_key=os.getenv('GROQ_API_KEY'),
    base_url= "https://api.groq.com/openai/v1",
    max_tokens= 1000
)
url='https://www.amazon.in/boAt-Airdopes-Alpha-Wireless-Earbuds/dp/B0C3ZYFZ77/?_encoding=UTF8&pd_rd_w=Vtv5m&content-id=amzn1.sym.14ff1b43-540a-4bfd-a18c-350bea29dfa2&pf_rd_p=14ff1b43-540a-4bfd-a18c-350bea29dfa2&pf_rd_r=71J1JGTXNXGXBTFZ7589&pd_rd_wg=C26bJ&pd_rd_r=b39614e5-a2b1-462c-bb44-ee166fdb3058&ref_=pd_hp_d_atf_Headphones'
loader= WebBaseLoader(url)
docs= loader.load()

# print(len(docs))
# print(docs[11].page_content)
# print(docs[11].metadata)

prompt1= PromptTemplate(
    template=" Answer the following question \n {question} from the following text -\n {text}",
    input_variables=['question','text']
)
parser= StrOutputParser()
chain= prompt1 | llm | parser

result= chain.invoke({
    'question':"what is the product  that we are talking", "text":f"product features and brand name in the give doc {docs} "
})
print(result)