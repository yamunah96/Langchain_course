from dotenv import load_dotenv
import os
load_dotenv()


from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate



llm= ChatOpenAI(
    model="openai/gpt-oss-120b",
    api_key=os.getenv('GROQ_API_KEY'),
    base_url= "https://api.groq.com/openai/v1",
    max_tokens= 1000
)
# print(llm.invoke("whats the captial of USA").content)




prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write 5 interview questions about {topic}"
)
chain = prompt | llm


result = chain.invoke({
    "topic": "Machine Learning"
})

print(result.content)