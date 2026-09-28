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

p1= PromptTemplate(
    template='Generate short and simple notes from the  following text  \n{text}',
    input_variables=['text']
)
p2= PromptTemplate(
    template='Generate  5 short question answers from the following text  \n{text}',
    input_variables=['text']
)
p3= PromptTemplate(
    template='merge the provided notes and quiz into a single documenet  \n{text}',
    input_variables=['text']
)

parser=  StrOutputParser()

chain= p1 | llm | parser| p2 | llm | parser | p3 | llm |parser
result= chain.invoke({"text":"Unemployemnet rate in World"})
print(result)

# chain.get_graph().print_ascii()