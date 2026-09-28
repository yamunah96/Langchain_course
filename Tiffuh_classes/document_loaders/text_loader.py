from dotenv import load_dotenv
import os
load_dotenv()


from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_core.runnables import RunnableLambda,RunnableParallel

llm= ChatOpenAI(
    model="openai/gpt-oss-120b",
    api_key=os.getenv('GROQ_API_KEY'),
    base_url= "https://api.groq.com/openai/v1",
    max_tokens= 1000
)
prompt1= PromptTemplate(
    template="write a 2 line poem using the above  {poem}",
    input_variables=['poem']
)
prompt2= PromptTemplate(
    template="convert the above {poem} into this {language}",
    input_variables=['poem','language']
)

parser= StrOutputParser()

# mandtory steps convert the promp1  output proper format
format_map= RunnableLambda(
    lambda x:{"poem":x, "language":"spanish"}
)

loader= TextLoader("./data.txt",encoding="utf-8")
docs= loader.load()


# chain= prompt1 | llm | parser | format_map| prompt2 |llm | parser

final_chain= RunnableParallel({
    "short_poem": prompt1 |llm | parser,
    "translated_poem": format_map| prompt2 | llm | parser
})

# mistakes proper formatting 

result= final_chain.invoke({
    "poem":docs[0].page_content,
})

print(result)

# i want original 2 line  poem from created p1, with translated output from p2
# print(docs)
# print(type(docs))
# print(docs[0].page_content)
# print(docs[0].metadata)