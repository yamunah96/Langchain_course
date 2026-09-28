import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

import tiktoken
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, WebBaseLoader

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(page_title="Resume Analyser", page_icon="📄", layout="wide")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Remove default top padding */
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

    /* App background */
    .stApp { background: #f0f2f6; }

    /* Header banner */
    .app-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 12px;
        padding: 1.4rem 2rem;
        margin-bottom: 1.5rem;
        color: white;
    }
    .app-header h1 { margin: 0; font-size: 1.8rem; font-weight: 700; color: white; }
    .app-header p  { margin: 0.3rem 0 0; font-size: 0.9rem; opacity: 0.75; color: white; }

    /* Left panel card */
    .left-card {
        background: white;
        border-radius: 12px;
        padding: 1.6rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        min-height: 600px;
    }

    /* Right panel card */
    .right-card {
        background: white;
        border-radius: 12px;
        padding: 1.8rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        min-height: 600px;
    }

    /* Section labels inside panels */
    .panel-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #6b7280;
        margin-bottom: 0.4rem;
    }

    /* Result heading */
    .result-placeholder {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 400px;
        color: #9ca3af;
        text-align: center;
    }
    .result-placeholder .icon { font-size: 3rem; margin-bottom: 1rem; }
    .result-placeholder p { font-size: 0.95rem; }

    /* Tool badge */
    .tool-badge {
        display: inline-block;
        background: #e0e7ff;
        color: #3730a3;
        border-radius: 999px;
        padding: 0.2rem 0.8rem;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    /* Radio buttons */
    div[data-testid="stRadio"] label { font-size: 0.9rem; }

    /* Analyse button */
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1rem;
        color: white;
    }
    .stButton button[kind="primary"]:hover { opacity: 0.9; }

    /* Thin separator */
    hr { border: none; border-top: 1px solid #e5e7eb; margin: 1rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── LLM ──────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_llm():
    return ChatOpenAI(
        model="openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        max_tokens=2000,
    )

# ── Helpers ───────────────────────────────────────────────────────────────────

def count_tokens(text: str, max_tokens: int = 6000) -> str:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])


def load_resume(source: str) -> str:
    src = source.strip()
    if src.lower().endswith(".pdf"):
        loader = PyPDFLoader(src)
    elif src.lower().endswith(".docx"):
        loader = Docx2txtLoader(src)
    elif src.startswith(("http://", "https://")):
        loader = WebBaseLoader(src)
    else:
        raise ValueError("Unsupported format. Provide a PDF, DOCX file, or a URL.")
    docs = loader.load()
    return count_tokens("\n".join(d.page_content for d in docs))

# ── Prompts ───────────────────────────────────────────────────────────────────

_prompt_jd = PromptTemplate(
    template="""You are a professional resume analyser. Analyse the resume against the job description and return in EXACT format:

**Score**: X/100
**Overall Match**: X%
Keywords matched: • ...
Missing keywords: • ...
Readability Score: X/100
ATS Compatibility Score: X/100

2-liner summary: ...
Skill gap analysis: • ...
Overall improvement suggestions: • ...
Industry specific feedback: • ...

Job Description: {job_description}
Resume: {resume_text}
Be honest, use rubrics.""",
    input_variables=["job_description", "resume_text"],
)

_prompt_standalone = PromptTemplate.from_template(
    """Score this resume standalone (clarity, format, ATS, skills). Use EXACT structure:

1. **Score**: X/100
2. **Strengths**: • ...
3. **Weaknesses**: • ...
4. **Skills Mentioned**: • ...
5. **Recommended Skills**: • ...
6. **Next Career Steps**: • ...

Resume: {context}"""
)

_parser = StrOutputParser()


def _run_chain(prompt, **kwargs) -> str:
    return (prompt | get_llm() | _parser).invoke(kwargs)

# ── Core tool functions (called directly — no agent ambiguity) ────────────────

def run_tool1(resume_source: str, job_description: str) -> str:
    """Tool 1: Match resume against a job description."""
    resume_text = load_resume(resume_source)
    return _run_chain(_prompt_jd, resume_text=resume_text, job_description=job_description)


def run_tool2(resume_source: str) -> str:
    """Tool 2: Standalone resume analysis."""
    resume_text = load_resume(resume_source)
    return _run_chain(_prompt_standalone, context=resume_text)

# ── LangChain tool definitions (for agent / programmatic use) ─────────────────

@tool
def analyse_with_job_description(resume: str, job_description: str) -> str:
    """Analyse a resume against a specific job description for match score, keywords, and skill gaps."""
    return run_tool1(resume, job_description)


@tool
def standalone_resume_analyser(resume: str) -> str:
    """Analyse a resume standalone for clarity, format, ATS compatibility, and career growth."""
    return run_tool2(resume)

# ── Session state ─────────────────────────────────────────────────────────────

if "result" not in st.session_state:
    st.session_state.result = None
if "error" not in st.session_state:
    st.session_state.error = None
if "active_tool" not in st.session_state:
    st.session_state.active_tool = None

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="app-header">
        <h1>📄 AI Resume Analyser</h1>
        <p>Upload your resume and get instant AI-powered feedback — match it against a job or get a standalone review.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Two-column layout ─────────────────────────────────────────────────────────

left, right = st.columns([1, 1.6], gap="large")

# ─────────────────── LEFT PANEL ───────────────────────────────────────────────
with left:
    st.markdown('<div class="left-card">', unsafe_allow_html=True)

    # ── Resume input ──────────────────────────────────────────────────────────
    st.markdown('<div class="panel-label">Resume Source</div>', unsafe_allow_html=True)
    input_mode = st.radio(
        "input_mode",
        ["📎 Upload File (PDF / DOCX)", "🔗 Portfolio / Website URL"],
        label_visibility="collapsed",
    )

    resume_source: str | None = None
    tmp_path: str | None = None

    if "Upload File" in input_mode:
        uploaded = st.file_uploader(
            "Drop your resume here",
            type=["pdf", "docx"],
            help="Supported: PDF and DOCX",
        )
        if uploaded:
            suffix = ".pdf" if uploaded.name.lower().endswith(".pdf") else ".docx"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded.read())
            tmp.close()
            tmp_path = tmp.name
            resume_source = tmp_path
            st.success(f"**{uploaded.name}** ready")
    else:
        url = st.text_input(
            "URL",
            placeholder="https://yourportfolio.com",
            label_visibility="collapsed",
        )
        if url.strip():
            resume_source = url.strip()
            st.success("URL set")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Analysis mode ─────────────────────────────────────────────────────────
    st.markdown('<div class="panel-label">Analysis Mode</div>', unsafe_allow_html=True)
    tool_choice = st.radio(
        "tool_choice",
        [
            "🎯 Match against Job Description",
            "🔍 Standalone Resume Analysis",
        ],
        label_visibility="collapsed",
    )

    job_desc = ""
    if "Match" in tool_choice:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="panel-label">Job Description</div>', unsafe_allow_html=True)
        job_desc = st.text_area(
            "jd",
            height=200,
            placeholder="Paste the full job description here…",
            label_visibility="collapsed",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Run button ────────────────────────────────────────────────────────────
    analyse_clicked = st.button(
        "Analyse Resume",
        type="primary",
        use_container_width=True,
        disabled=(resume_source is None),
    )

    if analyse_clicked:
        if "Match" in tool_choice and not job_desc.strip():
            st.warning("Please paste a job description first.")
        else:
            with st.spinner("Agent is analysing your resume…"):
                try:
                    if "Match" in tool_choice:
                        st.session_state.result = run_tool1(resume_source, job_desc)
                        st.session_state.active_tool = "Tool 1 — Job Description Match"
                    else:
                        st.session_state.result = run_tool2(resume_source)
                        st.session_state.active_tool = "Tool 2 — Standalone Analysis"
                    st.session_state.error = None
                except Exception as exc:
                    st.session_state.error = str(exc)
                    st.session_state.result = None
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)

    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────── RIGHT PANEL ──────────────────────────────────────────────
with right:
    st.markdown('<div class="right-card">', unsafe_allow_html=True)

    if st.session_state.result:
        st.markdown(
            f'<div class="tool-badge">✅ {st.session_state.active_tool}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(st.session_state.result)

    elif st.session_state.error:
        st.error(f"**Analysis failed:** {st.session_state.error}")

    else:
        st.markdown(
            """
            <div class="result-placeholder">
                <div class="icon">📊</div>
                <p><strong>Your analysis will appear here</strong></p>
                <p>Upload a resume on the left, choose an analysis mode,<br>and click <em>Analyse Resume</em> to get started.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)
