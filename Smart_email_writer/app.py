import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# -----------------------
# Load environment
# -----------------------
load_dotenv()

# -----------------------
# Page config
# -----------------------
st.set_page_config(
    page_title="Smart Email Writer",
    page_icon="✉️",
    layout="centered"
)

st.title("✉️ Smart Email Writer Assistant")
st.caption("Write professional emails in seconds (Powered by Groq)")

# -----------------------
# Load LLM (Groq)
# -----------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.4
)

# -----------------------
# Sidebar Controls
# -----------------------
with st.sidebar:
    st.header("✍️ Email Settings")

    email_type = st.selectbox(
        "Email Type",
        ["Professional", "Formal", "Friendly", "Cold Email", "Follow-up"]
    )

    tone = st.selectbox(
        "Tone",
        ["Polite", "Confident", "Friendly", "Persuasive", "Apologetic"]
    )

    length = st.selectbox(
        "Email Length",
        ["Short", "Medium", "Detailed"]
    )

# -----------------------
# Prompt Template
# -----------------------
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a smart professional email writing assistant.\n"
     "Write clear, well-structured, and natural-sounding emails.\n"
     "Return ONLY the email content.\n"
     "Do NOT add explanations.\n"
     "Format the email properly with subject, greeting, body, and closing.\n"
    ),
    ("human",
     "Write a {email_type} email.\n"
     "Tone: {tone}\n"
     "Length: {length}\n"
     "Context: {context}")
])

chain = prompt | llm

# -----------------------
# User Input
# -----------------------
context = st.text_area(
    "📌 What is this email about?",
    placeholder="Example: Requesting leave for 2 days due to personal reasons",
    height=150
)

generate = st.button("✉️ Generate Email")

# -----------------------
# Generate Email
# -----------------------
if generate and context.strip():
    with st.spinner("Writing your email..."):
        response = chain.invoke({
            "email_type": email_type,
            "tone": tone,
            "length": length,
            "context": context
        })

        email_text = response.content

    st.subheader("✅ Generated Email")
    st.markdown(
        f"""
```text
{email_text}
"""
)