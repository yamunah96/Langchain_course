from dotenv import load_dotenv
import os
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate

chat_prompt= ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are helpful assitany that provides information about {subject}")
    
])