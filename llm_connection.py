from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatGroq(
    model = "openai/gpt-oss-120b",
    temperature = 0
)

ques = input('Enter your question: ')

response = llm.invoke([HumanMessage(content=ques)]) 

print(f'AI Response: {response.content}')