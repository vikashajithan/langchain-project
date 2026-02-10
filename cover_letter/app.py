import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# -----------------------
# Load environment
# -----------------------
load_dotenv()

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="Cover Letter Generator",
    page_icon="📝",
    layout="centered"
)

st.title("📝 AI Cover Letter Generator")
st.caption("Upload your resume → Get a personalized cover letter")

# -----------------------
# Load LLM (Groq)
# -----------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.4
)

# -----------------------
# Helper: Read PDF
# -----------------------
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# -----------------------
# Upload Resume
# -----------------------
resume_file = st.file_uploader(
    "📄 Upload your Resume (PDF only)",
    type=["pdf"]
)

job_role = st.text_input(
    "🎯 Target Job Role (optional)",
    placeholder="Example: Junior Python Developer"
)

company_name = st.text_input(
    "🏢 Company Name (optional)",
    placeholder="Example: Google"
)

# -----------------------
# Prompt Template
# -----------------------
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert career assistant.\n"
     "Generate a professional cover letter based on the resume content.\n"
     "Customize it for the job role and company if provided.\n"
     "Return ONLY the cover letter.\n"
     "Do not include explanations."
    ),
    ("human",
     "Resume:\n{resume}\n\n"
     "Job Role: {role}\n"
     "Company: {company}")
])

chain = prompt | llm

# -----------------------
# Generate Button
# -----------------------
if st.button("✨ Generate Cover Letter"):
    if resume_file is None:
        st.warning("Please upload your resume.")
    else:
        with st.spinner("Reading resume and generating cover letter..."):
            resume_text = extract_text_from_pdf(resume_file)

            response = chain.invoke({
                "resume": resume_text,
                "role": job_role,
                "company": company_name
            })

            cover_letter = response.content

        st.subheader("✅ Your Cover Letter")
        st.markdown(
            f"""
```text
{cover_letter}
"""
)