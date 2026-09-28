from dotenv import load_dotenv
import os
load_dotenv()


from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch

llm= ChatOpenAI(
    model="openai/gpt-oss-120b",
    api_key=os.getenv('GROQ_API_KEY'),
    base_url= "https://api.groq.com/openai/v1",
    max_tokens= 1000
)
parser=  StrOutputParser()
# student1  (math question)   # student2 (coding question)
student1_prompt= PromptTemplate.from_template(
    'Solve this math problem :{query} with an example'
)

student2_prompt= PromptTemplate.from_template(
    'Solve this coding problem :{query} with an example'
)

math_chain= student1_prompt | llm | parser
coding_chain= student2_prompt | llm | parser

router= RunnableBranch(
   (lambda x: "math" in x["query"].lower() ,math_chain), coding_chain
)

result= router.invoke({
    "query":" Solve this Math equation 10+20"
})
print(result)


# Customer support ticket routing.  (1. customer ticket number, "product info", "return", "cancellation" etc)

# transcript--- chatbot # toolcalling, memmory(llM), LSEL, 2 project - 