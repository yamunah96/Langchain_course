import streamlit as st
from dotenv import load_dotenv
import os
import tempfile
import tiktoken

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Checker",
    page_icon="📄",
    layout="wide"
)

# ── LLM setup ────────────────────────────────────────────────────────────────
@st.cache_resource
def get_llm():
    return ChatOpenAI(
        model="openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        max_tokens=2000
    )

llm = get_llm()
parser = StrOutputParser()

# ── Prompts ───────────────────────────────────────────────────────────────────
JOB_MATCH_PROMPT = PromptTemplate(
    template="""You are an expert resume reviewer. Analyze this resume against the job description.

**Score**: X/100
**Overall Match**: X%
**Keywords Matched**: • ...
**Missing Keywords**: • ...
**Readability Score**: X/100
**ATS Compatibility Score**: X/100
**2-liner Summary**: ...
**Skill Gap Analysis**: • ...
**Improvement Suggestions**: • ...
**Industry Specific Feedback**: • ...

Job Description: {job_description}
Resume: {resume_text}

Be honest and structured. Use the exact format above.
""",
    input_variables=["resume_text", "job_description"]
)

STANDALONE_PROMPT = PromptTemplate(
    template="""You are an expert resume reviewer. Score this resume on its own merit.

1. **Score**: X/100
2. **Strengths**: • ...
3. **Weaknesses**: • ...
4. **Skills Mentioned**: • ...
5. **Recommended Skills to Add**: • ...
6. **Next Career Steps**: • ...
7. **ATS Compatibility**: X/100
8. **Overall Verdict**: ...

Resume: {resume_text}

Be honest and use the exact structure above.
""",
    input_variables=["resume_text"]
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def truncate_text(text: str, max_tokens: int = 6000) -> str:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])


def load_uploaded_file(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if suffix == ".pdf":
        loader = PyPDFLoader(tmp_path)
    elif suffix == ".docx":
        loader = Docx2txtLoader(tmp_path)
    else:
        os.unlink(tmp_path)
        raise ValueError(f"Unsupported file type: {suffix}")

    docs = loader.load()
    os.unlink(tmp_path)
    full_text = "\n\n".join(d.page_content for d in docs)
    return truncate_text(full_text)


def run_chain(prompt, **kwargs) -> str:
    return (prompt | llm | parser).invoke(kwargs)

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("📄 Resume Checker")
st.caption("Upload your resume and get instant AI-powered feedback.")

st.divider()

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Upload Resume")
    uploaded_file = st.file_uploader(
        "Supported formats: PDF, DOCX",
        type=["pdf", "docx"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        st.success(f"Uploaded: **{uploaded_file.name}**")

    st.subheader("Analysis Mode")
    mode = st.radio(
        "Choose how to analyze:",
        options=["Job-Specific Analysis", "Standalone Analysis"],
        help="Job-Specific matches your resume against a job description. Standalone scores it on its own."
    )

    job_description = ""
    if mode == "Job-Specific Analysis":
        st.subheader("Job Description")
        job_description = st.text_area(
            "Paste the job description here",
            height=180,
            placeholder="e.g. We are looking for a Data Analyst with experience in SQL, Python, Power BI..."
        )

    analyze_btn = st.button("Analyze Resume", type="primary", use_container_width=True)

with col2:
    st.subheader("Analysis Result")

    if analyze_btn:
        if not uploaded_file:
            st.warning("Please upload a resume first.")
        elif mode == "Job-Specific Analysis" and not job_description.strip():
            st.warning("Please enter a job description.")
        else:
            with st.spinner("Analyzing your resume..."):
                try:
                    resume_text = load_uploaded_file(uploaded_file)

                    if mode == "Job-Specific Analysis":
                        result = run_chain(
                            JOB_MATCH_PROMPT,
                            resume_text=resume_text,
                            job_description=job_description
                        )
                    else:
                        result = run_chain(
                            STANDALONE_PROMPT,
                            resume_text=resume_text
                        )

                    st.markdown(result)

                except Exception as e:
                    st.error(f"Something went wrong: {e}")
    else:
        st.info("Upload a resume and click **Analyze Resume** to see results here.")
