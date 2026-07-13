import os
import json
import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer

class ExplainableAI:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            self.client = Groq(api_key=api_key)
        else:
            self.client = None

    def generate_feedback(self, candidate, jd):
        api_key = os.getenv("GROQ_API_KEY")
        if api_key and not self.client:
            self.client = Groq(api_key=api_key)

        feedback = {
            "overall_match_score": candidate.get("final_score", 0),
            "matched_skills": [],
            "missing_skills": [],
            "strengths": [],
            "weaknesses": [],
            "recommendations": [],
            "learning_roadmap": ""
        }

        import re
        resume_text = candidate.get("skills", "").lower()
        jd_skills = [s.lower() for s in jd.get("required_skills", []) if s.strip()]

        for skill in jd_skills:
            escaped = re.escape(skill)
            prefix = r'(?<!\w)' if re.match(r'^\w', skill) else r''
            suffix = r'(?!\w)' if re.search(r'\w$', skill) else r''
            skill_pattern = prefix + escaped + suffix
            
            if re.search(skill_pattern, resume_text):
                feedback["matched_skills"].append(skill)
            else:
                feedback["missing_skills"].append(skill)

        if len(feedback["matched_skills"]) / max(1, len(jd_skills)) > 0.7:
            feedback["strengths"].append("Strong alignment with core technical skills.")
        else:
            feedback["weaknesses"].append("Lacks several core technical skills required for this role.")

        if candidate.get("experience_years", 0) >= jd.get("min_experience_years", 0):
            feedback["strengths"].append(f"Meets the required experience of {jd.get('min_experience_years', 0)} years.")
        else:
            feedback["weaknesses"].append(f"Falls short of the required {jd.get('min_experience_years', 0)} years of experience.")

        if feedback["missing_skills"]:
            feedback["recommendations"].append(f"Consider acquiring certifications or building projects using: {', '.join(feedback['missing_skills'][:3])}.")

        roadmap_prompt = f"Generate a short, 3-step learning roadmap for a candidate trying to become a {jd.get('title', 'Professional')}. They are missing the following skills: {', '.join(feedback['missing_skills'])}. Keep it concise."

        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": roadmap_prompt}]
                )
                feedback["learning_roadmap"] = response.choices[0].message.content
            except Exception as e:
                feedback["learning_roadmap"] = f"Error generating roadmap: {e}"
        else:
            feedback["learning_roadmap"] = f"1. Take a course on {', '.join(feedback['missing_skills'][:2])}.\n2. Build a project.\n3. Practice interview questions."

        return feedback

class RAGPipeline:
    def __init__(self, index_path, chunks_path):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.read_index(index_path)
        with open(chunks_path, 'r') as f:
            self.chunks = json.load(f)
            
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            self.client = Groq(api_key=api_key)
        else:
            self.client = None
            
    def retrieve(self, query, top_k=2):
        query_emb = self.embedding_model.encode([query], show_progress_bar=False)
        distances, indices = self.index.search(query_emb, top_k)
        return "\n\n".join([self.chunks[i] for i in indices[0]])
        
    def generate_response(self, query):
        context = self.retrieve(query)
        prompt = f"""You are an expert AI Hiring Assistant. Use the context to answer the candidate's question.
Context: {context}
Question: {query}
Answer:"""
        if not self.client:
            return f"Context retrieved:\n{context}\n\n(Groq API key missing, cannot generate final response)"
            
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {e}"

class ConversationalAgent:
    def __init__(self, candidate_feedback, rag_pipeline):
        self.feedback = candidate_feedback
        self.rag = rag_pipeline
        self.history = []

        self.system_prompt = f"You are an AI Hiring Mentor chatting with {self.feedback.get('candidate_name', 'the candidate')} applying for {self.feedback.get('job_title', 'a job')}. Their score is {self.feedback.get('overall_match_score', 0)}%. Missing skills: {', '.join(self.feedback.get('missing_skills', []))}."

        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            self.client = Groq(api_key=api_key)
        else:
            self.client = None

    def ask(self, user_input):
        api_key = os.getenv("GROQ_API_KEY")
        if api_key and not self.client:
            self.client = Groq(api_key=api_key)

        self.history.append({"role": "user", "content": user_input})
        
        if not self.client:
            return "I'm offline (no API key). You should learn " + ", ".join(self.feedback.get('missing_skills', []))
        else:
            try:
                messages = [{"role": "system", "content": self.system_prompt}]
                for msg in self.history[:-1]:
                    messages.append(msg)
                
                hr_context = self.rag.retrieve(user_input)
                augmented_prompt = f"HR Context:\n{hr_context}\n\nQuestion: {user_input}"
                messages.append({"role": "user", "content": augmented_prompt})

                response = self.client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages
                )
                reply = response.choices[0].message.content
            except Exception as e:
                reply = f"Error: {e}"

        self.history.append({"role": "assistant", "content": reply})
        return reply
