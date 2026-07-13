import streamlit as st
import json
import os
import sys
import asyncio
from dotenv import load_dotenv

load_dotenv(override=True)

# Fix asyncio "Event loop is closed" error on Windows
if sys.platform == 'win32':
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Setup page
st.set_page_config(page_title="AI Hiring Assistant", layout="wide")

# Inject Premium Custom CSS Theme
st.markdown("""
<style>
/* Modern Glassmorphism & Cyber Aesthetics */
.stButton>button {
    background: linear-gradient(135deg, #7A3CFF 0%, #4A00E0 100%);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(122, 60, 255, 0.3);
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(122, 60, 255, 0.6);
    color: white;
    border: none;
}
.stTextInput>div>div>input {
    border-radius: 10px;
    border: 1px solid #2D2B4A;
    background-color: #13111C;
    transition: border-color 0.3s ease;
}
.stTextInput>div>div>input:focus {
    border-color: #7A3CFF;
}
.stTextArea>div>div>textarea {
    border-radius: 10px;
    border: 1px solid #2D2B4A;
    background-color: #13111C;
}
.stTextArea>div>div>textarea:focus {
    border-color: #7A3CFF;
}
/* Enhanced Chat Messages */
[data-testid="stChatMessage"] {
    background-color: #161521;
    border-radius: 12px;
    padding: 15px;
    border: 1px solid #2D2B4A;
    margin-bottom: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}
/* Tab Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
}
.stTabs [aria-selected="true"] {
    border-bottom: 3px solid #7A3CFF !important;
    background: linear-gradient(to top, rgba(122, 60, 255, 0.1), transparent);
    border-radius: 8px 8px 0px 0px;
}
</style>
""", unsafe_allow_html=True)

# Add src to path
sys.path.append('src')

# Import modules
try:
    # pyrefly: ignore [missing-import]
    from generative_ai_engine import ExplainableAI, RAGPipeline, ConversationalAgent
    # pyrefly: ignore [missing-import]
    from resume_parser import parse_resume
    MODULES_LOADED = True
except ImportError as e:
    MODULES_LOADED = False
    st.error(f"Failed to load modules: {e}")

st.title("AI-Powered Intelligent Hiring Assistant", anchor=False)
st.markdown("An end-to-end AI system that evaluates candidate resumes against job requirements and generates personalized, explainable feedback.")

if not MODULES_LOADED:
    st.warning("Core modules are missing. The app might not function correctly.")

# Setup state
if "rag" not in st.session_state:
    try:
        st.session_state.rag = RAGPipeline('vector_store/faiss_index/hr_knowledge.index', 'vector_store/faiss_index/chunks.json')
    except:
        st.session_state.rag = None

if "xai" not in st.session_state:
    st.session_state.xai = ExplainableAI()

tab1, tab2 = st.tabs(["Candidate View", "Recruiter View"])

with tab1:
    st.header("Candidate Portal", anchor=False)
    st.markdown("Upload your resume and the target Job Description to get instant feedback and chat with our AI Mentor.")
    
    col_jd, col_res = st.columns(2)
    
    with col_jd:
        st.subheader("1. Job Description", anchor=False)
        st.markdown("Paste the raw Job Description text below.")
        jd_raw_text = st.text_area("Job Description Paragraph", value="We are looking for a Software Engineer with at least 2 years of experience. You must be proficient in Python, React, AWS, SQL, and Git.", height=150)
        
    with col_res:
        st.subheader("2. Your Resume", anchor=False)
        uploaded_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=['pdf', 'docx'])
        
    if uploaded_file and st.button("Analyze My Resume"):
        st.success("Resume uploaded successfully!")
        
        # Save temp file
        temp_dir = "data/temp"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner('Parsing resume and analyzing match...'):
            parsed_data = parse_resume(file_path)
            
            import re
            
            api_key = os.getenv("GROQ_API_KEY")
            jd = None
            if api_key:
                from groq import Groq
                client = Groq(api_key=api_key)
                prompt = f"""Extract information from this job description and return ONLY a JSON object:
{{
  "title": "string",
  "required_skills": ["list", "of", "skills"],
  "min_experience_years": int
}}
Job Description: {jd_raw_text}"""
                try:
                    res = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1
                    )
                    content = res.choices[0].message.content
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        jd = json.loads(json_match.group(0))
                except Exception as e:
                    pass
            
            if not jd:
                title_match = re.search(r'(?i)(.*? engineer|.*? developer|.*? manager|.*? analyst|.*? scientist)', jd_raw_text)
                title = title_match.group(1).strip() if title_match else "Professional"
                
                exp_match = re.search(r'(\d+)\+?\s*years?', jd_raw_text, re.IGNORECASE)
                exp = int(exp_match.group(1)) if exp_match else 0
                
                # Enhanced Local NLP Extractor
                common_skills = ['python', 'java', 'react', 'aws', 'sql', 'git', 'machine learning', 'pandas', 'c++', 'docker', 'kubernetes', 'node', 'javascript', 'html', 'css', 'azure', 'gcp', 'scikit-learn', 'numpy', 'statistics', 'predictive modeling', 'data analysis', 'data visualization', 'spring', 'hibernate', 'linux', 'devops', 'golang', 'go', 'mlops']
                skills = []
                for s in common_skills:
                    if re.search(r'\b' + re.escape(s) + r'\b', jd_raw_text, re.IGNORECASE):
                        skills.append(s.title())
                
                jd = {
                    "title": title.title(),
                    "required_skills": skills if skills else ['Python', 'SQL'],
                    "min_experience_years": exp
                }
            
            # Create candidate
            # Parse experience heuristic
            exp_text = parsed_data.get("experience", "")
            exp_years = len(exp_text.split('\\n')) // 3 if exp_text else 0 # Dummy heuristic
            
            candidate = {
                "name": parsed_data.get("name") or "Candidate",
                "skills": parsed_data.get("raw_text", ""),
                "experience_years": exp_years,
                "final_score": 0 # Will be calculated by XAI logic implicitly
            }
            
            # Generate feedback
            feedback = st.session_state.xai.generate_feedback(candidate, jd)
            
            # Calculate score
            total_req = len(jd['required_skills'])
            matched = len(feedback['matched_skills'])
            score = int((matched / total_req) * 100) if total_req > 0 else 100
            feedback['overall_match_score'] = score
            
            st.session_state.current_feedback = feedback
            st.session_state.messages = [] # Reset chat
            
            # Cleanup
            try:
                os.remove(file_path)
            except:
                pass
            
    if "current_feedback" in st.session_state:
        feedback = st.session_state.current_feedback
        
        st.divider()
        st.subheader(f"Feedback Report for {feedback.get('candidate_name', 'Candidate')}", anchor=False)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Overall Match Score", f"{feedback['overall_match_score']}%")
            st.write("**Matched Skills:**", ", ".join(feedback['matched_skills']) if feedback['matched_skills'] else "None")
            st.write("**Missing Skills:**", ", ".join(feedback['missing_skills']) if feedback['missing_skills'] else "None")
            
        with col2:
            st.write("**Strengths:**")
            for s in feedback['strengths']:
                st.write(f"- {s}")
            st.write("**Weaknesses:**")
            for w in feedback['weaknesses']:
                st.write(f"- {w}")
                
        st.info(f"**Learning Roadmap:**\\n{feedback['learning_roadmap']}")
        
        st.divider()
        st.subheader("Chat with AI Mentor", anchor=False)
        
        # Start agent
        if "agent" not in st.session_state or st.session_state.get("agent_feedback") != feedback:
            st.session_state.agent = ConversationalAgent(feedback, st.session_state.rag)
            st.session_state.agent_feedback = feedback
            
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
        if prompt := st.chat_input("Ask a question about your feedback... (e.g., 'Why is my score low?')"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    reply = st.session_state.agent.ask(prompt)
                st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

with tab2:
    st.header("Recruiter Dashboard", anchor=False)
    st.markdown("Upload Job Descriptions and view ranked candidates in batch.")
    
    recruiter_jd = st.text_area("Paste Job Description for Batch Ranking", height=150, placeholder="We are looking for a Data Scientist with...")
    
    st.markdown("### Upload Candidate Dataset")
    st.markdown("Upload a CSV file containing candidate resumes. The CSV must have at least one column named `clean_text`, `Skills`, or `Text` containing the resume content.")
    uploaded_csv = st.file_uploader("Upload Candidates (CSV)", type=["csv"])
    
    if st.button("Rank Candidates against JD"):
        if recruiter_jd.strip():
            with st.spinner("Extracting requirements and ranking candidates..."):
                try:
                    # Extract skills with Groq
                    prompt = f"Extract a comma-separated list of ONLY the core technical skills from this Job Description:\\n{recruiter_jd}\\nReturn ONLY the comma-separated list, nothing else."
                    
                    api_key = os.getenv("GROQ_API_KEY")
                    if api_key:
                        from groq import Groq
                        client = Groq(api_key=api_key)
                        res = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.1
                        )
                        jd_skills_str = res.choices[0].message.content
                    else:
                        import re
                        common_tech_skills = ['python', 'java', 'react', 'aws', 'sql', 'machine learning', 'pandas', 'c++', 'docker', 'kubernetes', 'node', 'javascript', 'html', 'css', 'azure', 'gcp', 'scikit-learn', 'numpy', 'statistics', 'predictive modeling', 'data analysis', 'data visualization', 'spring', 'hibernate', 'git', 'linux', 'devops']
                        found_skills = []
                        for skill in common_tech_skills:
                            if re.search(r'\b' + re.escape(skill) + r'\b', recruiter_jd, re.IGNORECASE):
                                found_skills.append(skill.title())
                        
                        if found_skills:
                            jd_skills_str = ", ".join(found_skills)
                        else:
                            jd_skills_str = "Communication, Problem Solving"
                            
                    jd_skills = [s.strip() for s in jd_skills_str.split(",") if s.strip()]
                    
                    st.success(f"Extracted Skills: {', '.join(jd_skills)}")
                    
                    # Rank candidates
                    import ranking_engine
                    import importlib
                    importlib.reload(ranking_engine)
                    ranker = ranking_engine.CandidateRanker()
                    
                    if uploaded_csv is not None:
                        import pandas as pd
                        custom_df = pd.read_csv(uploaded_csv)
                        success = ranker.load_custom_data(custom_df)
                        if not success:
                            st.error("Failed to process uploaded CSV. Falling back to default database.")
                            ranker.load_data()
                    else:
                        ranker.load_data()
                        
                    ranked_df = ranker.rank_candidates(required_skills=jd_skills, recruiter_jd_text=recruiter_jd)
                    
                    display_cols = []
                    for col in ["Name", "Candidate Name", "CandidateName", "Email", "Phone", "ResumeID", "Category"]:
                        if col in ranked_df.columns:
                            display_cols.append(col)
                            
                    display_cols.extend(["SkillScore", "SimilarityScore", "FinalScore"])
                    if "Skills_List" in ranked_df.columns: display_cols.append("Skills_List")
                    
                    st.dataframe(ranked_df[display_cols].head(20), use_container_width=True)
                    
                    # Show plot
                    ranker.visualize_ranking(top_n=10)
                    plot_path = 'reports/3_Semantic_Ranking/candidate_ranking.png'
                    if os.path.exists(plot_path):
                        st.image(plot_path, caption="Candidate Score Breakdown")
                        
                except Exception as e:
                    st.error(f"Error ranking candidates: {e}")
        else:
            st.warning("Please paste a Job Description first.")
