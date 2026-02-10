import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

# -----------------------------
# Load Environment
# -----------------------------
load_dotenv()

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="AI Coding Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Coding Assistant")
st.caption("Powered by Groq LLaMA 3.1")

# -----------------------------
# Load Groq Model
# -----------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3
)

# -----------------------------
# Prompt
# -----------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert AI coding assistant.\n"
     "ALWAYS return code inside triple backticks with language name.\n"
     "Example:\n"
     "```python\nprint('Hello')\n```\n"
     "Explain only if user asks.\n"
     "Be concise and accurate."
    ),
    ("human", "{question}")
])

chain = prompt | llm

# -----------------------------
# Chat Memory
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# Show Chat History
# -----------------------------
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# -----------------------------
# User Input
# -----------------------------
user_input = st.chat_input("Ask your coding question...")

if user_input:
    st.session_state.messages.append(
        HumanMessage(content=user_input)
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.invoke(
                {"question": user_input}
            )
            answer = response.content
            st.markdown(answer)

    st.session_state.messages.append(
        AIMessage(content=answer)
    )
