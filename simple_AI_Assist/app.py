from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.7
)


prompt = PromptTemplate(
    input_variable=["user_input"],
    template="""
    Your are a helpfull AI assitant.
    User says = {user_input} 
    Your response:
    """
)

chain = prompt | llm

if __name__ == "__main__":
    user = input("Ask me anything:")
    response = chain.invoke({"user_input": user})
    print(response.content)

