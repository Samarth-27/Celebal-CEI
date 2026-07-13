import os
import re
import spacy
import fitz  # PyMuPDF
import pdfplumber
from docx import Document

try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    # Fallback model so the app still runs
    nlp = spacy.blank("en")

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    if ext == '.pdf':
        try:
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text() + "\n"
        except Exception:
            try:
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        text += page.extract_text() + "\n"
            except Exception:
                pass
    elif ext == '.docx':
        try:
            doc = Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception:
            pass
    return text.strip()

def extract_email(text):
    match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    return match.group(0) if match else None

def extract_phone(text):
    match = re.search(r'\+?\d{1,3}[-.\s]?\(?\d{1,4}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}', text)
    return match.group(0) if match else None

def extract_sections(text):
    lines = text.split('\n')
    sections = {'Skills': [], 'Experience': [], 'Education': [], 'Projects': [], 'Certifications': []}
    current_section = None
    section_keywords = {
        'Skills': ['skills', 'technologies'],
        'Experience': ['experience', 'work history'],
        'Education': ['education'],
        'Projects': ['projects'],
        'Certifications': ['certifications']
    }
    for line in lines:
        line_clean = line.strip().lower()
        if not line_clean: continue
        found_header = False
        for sec, keywords in section_keywords.items():
            if any(line_clean.startswith(kw) for kw in keywords) and len(line_clean.split()) < 5:
                current_section = sec
                found_header = True
                break
        if not found_header and current_section:
            sections[current_section].append(line.strip())

    for sec in sections:
        sections[sec] = '\n'.join(sections[sec]).strip()
    return sections

def parse_resume(file_path):
    raw_text = extract_text(file_path)
    if not raw_text: return {}
    doc = nlp(raw_text)
    name = None
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text
            break
    sections = extract_sections(raw_text)
    return {
        "filename": os.path.basename(file_path),
        "name": name,
        "email": extract_email(raw_text),
        "phone": extract_phone(raw_text),
        "skills": sections.get("Skills", ""),
        "experience": sections.get("Experience", ""),
        "education": sections.get("Education", ""),
        "projects": sections.get("Projects", ""),
        "certifications": sections.get("Certifications", ""),
        "raw_text": raw_text
    }
