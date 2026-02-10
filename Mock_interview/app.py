import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# ----------------------
# Load env
# ----------------------
load_dotenv()

# ----------------------
# Page config
# ----------------------
st.set_page_config(
    page_title="Mock Interviewer AI",
    page_icon="🎤",
    layout="centered"
)

st.title("🎤 Mock Interviewer – GenAI Coach")
st.caption("Practice → See Ideal Answer → Get Feedback")

# ----------------------
# Load Groq Model
# ----------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.4
)

# ----------------------
# Sidebar
# ----------------------
with st.sidebar:
    st.header("Interview Setup")

    role = st.text_input("Target Role", "Python Developer")
    level = st.selectbox("Experience Level", ["Fresher", "Junior", "Mid", "Senior"])
    interview_type = st.selectbox("Interview Type", ["Technical", "HR", "Behavioral", "Mixed"])
    start = st.button("🚀 Start Interview")

# ----------------------
# Session State
# ----------------------
if "question" not in st.session_state:
    st.session_state.question = None

# ----------------------
# Prompts
# ----------------------
question_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a professional interviewer.\n"
     "Generate ONE interview question.\n"
     "Role: {role}\n"
     "Level: {level}\n"
     "Type: {type}"
    ),
    ("human", "Ask a question.")
])

ideal_answer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert interviewer.\n"
     "Provide an ideal high-quality answer to the question."
    ),
    ("human", "{question}")
])

feedback_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Evaluate the candidate answer.\n"
     "Give score out of 10.\n"
     "Give strengths and improvements."
    ),
    ("human",
     "Question: {question}\n"
     "User Answer: {answer}")
])

question_chain = question_prompt | llm
ideal_chain = ideal_answer_prompt | llm
feedback_chain = feedback_prompt | llm

# ----------------------
# Start Interview
# ----------------------
if start:
    q = question_chain.invoke({
        "role": role,
        "level": level,
        "type": interview_type
    })
    st.session_state.question = q.content

# ----------------------
# Display Question
# ----------------------
if st.session_state.question:
    st.subheader("🧠 Interview Question")
    st.write(st.session_state.question)

    user_answer = st.text_area("Your Answer", height=150)

    if st.button("Submit Answer"):
        with st.spinner("Evaluating..."):

            ideal = ideal_chain.invoke(
                {"question": st.session_state.question}
            )

            feedback = feedback_chain.invoke({
                "question": st.session_state.question,
                "answer": user_answer
            })

        st.subheader("✅ Ideal Answer")
        st.write(ideal.content)

        st.subheader("📊 Feedback & Score")
        st.write(feedback.content)

        if st.button("➡️ Next Question"):
            q = question_chain.invoke({
                "role": role,
                "level": level,
                "type": interview_type
            })
            st.session_state.question = q.content
