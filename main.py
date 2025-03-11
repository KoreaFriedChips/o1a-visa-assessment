import os
from typing import List, Dict, Any, Optional
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import PyPDF2
import docx
import re
from enum import Enum
import io
import json
import numpy as np
import joblib
import nltk
from nltk.tokenize import sent_tokenize
from fastapi.middleware.cors import CORSMiddleware

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

app = FastAPI(title="ML-Enhanced O-1A Visa Qualification Assessment API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QualificationLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class CriterionMatch(BaseModel):
    criterion: str
    matches: List[str]
    confidence: float

class ImpactAssessment(BaseModel):
    description: str
    significance: str
    scale: str

class CriterionDetail(BaseModel):
    criterion: str
    matches: List[str]
    confidence: float
    impact: Optional[ImpactAssessment] = None

class AssessmentResponse(BaseModel):
    matches_by_criterion: List[CriterionDetail]
    qualification_rating: QualificationLevel
    justification: str
    strongest_criteria: List[str]
    improvement_suggestions: List[str]

O1A_CRITERIA = {
    "Awards": [
        r"award", r"prize", r"medal", r"distinction", r"recognition", 
        r"honor", r"scholarship", r"grant", r"fellowship", r"recipient",
        r"won", r"winning", r"received", r"honored", r"recognized"
    ],
    "Membership": [
        r"member", r"association", r"society", r"organization", 
        r"committee", r"board", r"fellow", r"elected", r"admitted",
        r"invited", r"invitee", r"exclusive", r"prestigious", r"appointment"
    ],
    "Press": [
        r"press", r"media", r"news", r"article", r"feature", 
        r"interview", r"publication", r"spotlight", r"covered", r"journal",
        r"magazine", r"newspaper", r"documentary", r"profile", r"broadcast"
    ],
    "Judging": [
        r"judge", r"reviewer", r"panel", r"evaluation", r"assess", 
        r"peer review", r"referee", r"competition", r"adjudicate", r"examiner",
        r"selection committee", r"editorial board", r"evaluate", r"critique"
    ],
    "Original contribution": [
        r"contribution", r"innovation", r"invention", r"patent", 
        r"novel", r"groundbreaking", r"pioneering", r"developed", r"created",
        r"discovery", r"breakthrough", r"revolutionary", r"transformative",
        r"first-of-its-kind", r"unprecedented", r"established new", r"initiated"
    ],
    "Scholarly articles": [
        r"scholar", r"article", r"publication", r"journal", r"paper", 
        r"author", r"research", r"publish", r"conference proceedings",
        r"cited", r"citation", r"impact factor", r"peer-reviewed", r"h-index",
        r"thesis", r"dissertation", r"scientific", r"academic"
    ],
    "Critical employment": [
        r"critical", r"essential", r"lead", r"director", r"key role", 
        r"pivotal", r"senior", r"distinguished", r"principal", r"chief",
        r"executive", r"founder", r"co-founder", r"head", r"chair", 
        r"president", r"vice president", r"C-suite", r"VP", r"CTO", r"CEO", r"COO"
    ],
    "High remuneration": [
        r"salary", r"compensation", r"remuneration", r"income", r"wage", 
        r"earnings", r"stipend", r"pay", r"bonus", r"stock option",
        r"equity", r"profit sharing", r"high paying", r"top", r"percentile",
        r"competitive", r"lucrative", r"substantial"
    ]
}

IMPACT_MARKERS = {
    "scale": [
        r"global", r"international", r"worldwide", r"national", r"industry-wide",
        r"market-leading", r"millions", r"billions", r"thousands", r"large-scale",
        r"extensive", r"widespread", r"substantial", r"significant"
    ],
    "significance": [
        r"first", r"only", r"groundbreaking", r"revolutionary", r"transformative",
        r"influential", r"pioneering", r"leading", r"major", r"substantial",
        r"critical", r"essential", r"unique", r"unprecedented", r"rare"
    ],
    "metrics": [
        r"\d+%", r"\d+ percent", r"doubled", r"tripled", r"increased by", 
        r"reduced by", r"improved", r"enhanced", r"saved", r"generated",
        r"worth \$", r"valued at", r"revenue", r"funding", r"raised", r"budget"
    ]
}

use_ml_model = False
try:
    text_model = joblib.load("models/o1a_text_model.joblib")
    enhanced_model = joblib.load("models/o1a_enhanced_model.joblib")
    tfidf_vectorizer = joblib.load("models/o1a_tfidf_vectorizer.joblib")
    use_ml_model = True
    print("ML models loaded successfully!")
except Exception as e:
    print(f"ML models not found or error loading them: {str(e)}")
    print("Falling back to rule-based approach.")

def extract_text_from_pdf(file_content):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(file_content)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")

def extract_text_from_docx(file_content):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(file_content)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing DOCX: {str(e)}")

def extract_text_from_cv(file_content, file_extension):
    """Extract text based on file type"""
    if file_extension.lower() == "pdf":
        return extract_text_from_pdf(file_content)
    elif file_extension.lower() in ["docx", "doc"]:
        return extract_text_from_docx(file_content)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload PDF or DOCX.")

def extract_sentences(text):
    """Split text into sentences using NLTK"""
    try:
        sentences = sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]
    except:
        text = re.sub(r'\n+', '. ', text)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

# Text based model - Find matches for each O-1A criterion in the CV text
def find_criterion_matches(text, sentences):
    matches_by_criterion = []
    
    for criterion, patterns in O1A_CRITERIA.items():
        matched_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for pattern in patterns:
                if re.search(pattern.lower(), sentence_lower):
                    if sentence not in matched_sentences:
                        matched_sentences.append(sentence)
                    break
        
        confidence = min(0.9, len(matched_sentences) / 10.0 + 0.3) if matched_sentences else 0.0
        
        impact = None
        if matched_sentences and confidence > 0.5:
            impact = assess_impact(matched_sentences)
        
        matches_by_criterion.append(
            CriterionDetail(
                criterion=criterion, 
                matches=matched_sentences,
                confidence=confidence,
                impact=impact
            )
        )
    
    return matches_by_criterion

def assess_impact(sentences):
    scale = "Not specified"
    significance = "Not specified"
    metrics_found = []
    
    combined_text = " ".join(sentences).lower()
    
    for scale_marker in IMPACT_MARKERS["scale"]:
        if re.search(scale_marker.lower(), combined_text):
            scale = "Global/Large-scale" if any(x in combined_text for x in ["global", "world", "international", "million", "billion"]) else "Significant"
            break
    
    for sig_marker in IMPACT_MARKERS["significance"]:
        if re.search(sig_marker.lower(), combined_text):
            significance = "Exceptional/Revolutionary" if any(x in combined_text for x in ["first", "only", "revolutionary", "groundbreaking"]) else "Substantial"
            break
    
    for metric_pattern in IMPACT_MARKERS["metrics"]:
        matches = re.findall(metric_pattern, combined_text)
        metrics_found.extend(matches)
    
    if metrics_found:
        description = f"Quantifiable impact: {', '.join(metrics_found)}"
    else:
        description = "Impact described in qualitative terms"
        
    return ImpactAssessment(
        description=description,
        significance=significance,
        scale=scale
    )

def predict_with_ml_model(cv_text, criterion_matches):
    """Use trained ML model to predict qualification rating"""
    try:
        criteria_count = sum(1 for c in criterion_matches if len(c.matches) > 0)
        strong_criteria_count = sum(1 for c in criterion_matches if len(c.matches) > 0 and c.confidence >= 0.7)
        
        text_features = tfidf_vectorizer.transform([cv_text])
        
        additional_features = np.array([[criteria_count, strong_criteria_count]])
        combined_features = np.hstack((text_features.toarray(), additional_features))
        
        prediction = enhanced_model.predict(combined_features)[0]
        
        probabilities = enhanced_model.predict_proba(combined_features)[0]
        confidence = max(probabilities)
        
        return prediction, confidence
    except Exception as e:
        print(f"Error in ML prediction: {str(e)}")
        return calculate_qualification_rating(criterion_matches), 0.7

def calculate_qualification_rating(matches_by_criterion):
    """Determine qualification rating based on criteria matches (rule-based)"""
    strong_criteria = sum(1 for criterion in matches_by_criterion 
                         if len(criterion.matches) > 0 and criterion.confidence >= 0.7)
    
    medium_criteria = sum(1 for criterion in matches_by_criterion 
                         if len(criterion.matches) > 0 and 0.4 <= criterion.confidence < 0.7)
    
    weak_criteria = sum(1 for criterion in matches_by_criterion 
                        if len(criterion.matches) > 0 and criterion.confidence < 0.4)
    
    # Calculate a weighted score
    weighted_score = (strong_criteria * 1.0) + (medium_criteria * 0.5) + (weak_criteria * 0.2)
    
    # Determine a qualification level based on previous year's approval rate (i.e. training data matches a similar rate)
    # 2021: 91%, 2022: 94%, 2023: 92%
    if (strong_criteria >= 3 or weighted_score >= 3.5):
        return QualificationLevel.HIGH
    elif (strong_criteria >= 2 or weighted_score >= 2.0):
        return QualificationLevel.MEDIUM
    else:
        return QualificationLevel.LOW

def generate_justification(matches_by_criterion, qualification_rating, ml_confidence=None):
    """Generate detailed justification text for the qualification rating"""
    strong_criteria = [c.criterion for c in matches_by_criterion if len(c.matches) > 0 and c.confidence >= 0.7]
    medium_criteria = [c.criterion for c in matches_by_criterion if len(c.matches) > 0 and 0.4 <= c.confidence < 0.7]
    
    strong_count = len(strong_criteria)
    medium_count = len(medium_criteria)
    
    confidence_text = ""
    if ml_confidence is not None:
        confidence_level = "high" if ml_confidence > 0.8 else "moderate" if ml_confidence > 0.6 else "limited"
        confidence_text = f" The system has {confidence_level} confidence in this assessment."
    
    if qualification_rating == QualificationLevel.HIGH:
        return (
            f"The candidate demonstrates strong evidence for {strong_count} criteria "
            f"({', '.join(strong_criteria)}) and moderate evidence for {medium_count} more. "
            f"Since O-1A visa requires meeting at least 3 criteria with substantial evidence, "
            f"and the candidate shows compelling evidence across multiple criteria, "
            f"there appears to be a high likelihood of qualification.{confidence_text}"
        )
    elif qualification_rating == QualificationLevel.MEDIUM:
        return (
            f"The candidate shows strong evidence for {strong_count} criteria "
            f"({', '.join(strong_criteria) if strong_criteria else 'none'}) and moderate evidence "
            f"for {medium_count} more ({', '.join(medium_criteria) if medium_criteria else 'none'}). "
            f"While the minimum requirement is 3 criteria, the strength of evidence "
            f"suggests a moderate chance of qualification, though additional documentation "
            f"may be beneficial to strengthen the application.{confidence_text}"
        )
    else:
        return (
            f"The candidate demonstrates limited strong evidence across the required criteria, "
            f"with only {strong_count} strong criteria. "
            f"Since O-1A visa requires meeting at least 3 criteria with substantial evidence, "
            f"the candidate may face challenges qualifying without additional accomplishments "
            f"or more compelling documentation.{confidence_text}"
        )

def generate_improvement_suggestions(matches_by_criterion):
    """Generate suggestions for improving O-1A qualification"""
    suggestions = []
    
    # Check criteria with no or weak matches
    weak_or_missing = [c.criterion for c in matches_by_criterion if len(c.matches) == 0 or c.confidence < 0.4]
    
    if "Awards" in weak_or_missing:
        suggestions.append("Consider highlighting any recognition or awards you've received in your field, even smaller or regional ones.")
    
    if "Membership" in weak_or_missing:
        suggestions.append("Join or apply to prestigious professional associations in your field that require nomination or selective admission.")
    
    if "Press" in weak_or_missing:
        suggestions.append("Seek opportunities for media coverage of your work, or publish articles about your expertise in industry publications.")
    
    if "Judging" in weak_or_missing:
        suggestions.append("Look for opportunities to serve as a reviewer for conferences, journals, or competitions in your field.")
    
    if "Original contribution" in weak_or_missing:
        suggestions.append("Document how your work has created new processes, techniques, or approaches that others in your field have adopted.")
    
    if "Scholarly articles" in weak_or_missing:
        suggestions.append("Consider publishing your expertise in academic or industry journals, even if as a co-author.")
    
    if "Critical employment" in weak_or_missing:
        suggestions.append("Emphasize leadership roles or instances where you've been essential to projects or organizations.")
    
    if "High remuneration" in weak_or_missing:
        suggestions.append("If applicable, document how your compensation reflects your exceptional ability (comparative data can help).")
    
    suggestions.append("Obtain letters of recommendation from experts in your field that specifically address how you meet O-1A criteria.")
    suggestions.append("Quantify your impact wherever possible (revenue generated, users impacted, efficiency improved, etc.).")
    
    return suggestions

@app.post("/assess-o1a-qualification", response_model=AssessmentResponse)
async def assess_o1a_qualification(file: UploadFile = File(...)):
    filename = file.filename
    file_extension = filename.split(".")[-1] if "." in filename else ""
    
    contents = await file.read()
    file_like_object = io.BytesIO(contents)
    
    cv_text = extract_text_from_cv(file_like_object, file_extension)
    
    sentences = extract_sentences(cv_text)
    
    matches_by_criterion = find_criterion_matches(cv_text, sentences)
    
    ml_confidence = None
    if use_ml_model:
        qualification_rating_value, ml_confidence = predict_with_ml_model(cv_text, matches_by_criterion)
        qualification_rating = QualificationLevel(qualification_rating_value)
    else:
        qualification_rating = calculate_qualification_rating(matches_by_criterion)
    
    justification = generate_justification(matches_by_criterion, qualification_rating, ml_confidence)
    
    strongest_criteria = [c.criterion for c in matches_by_criterion 
                          if len(c.matches) > 0 and c.confidence >= 0.6]
    
    improvement_suggestions = generate_improvement_suggestions(matches_by_criterion)
    
    # Return assessment response
    return AssessmentResponse(
        matches_by_criterion=matches_by_criterion,
        qualification_rating=qualification_rating,
        justification=justification,
        strongest_criteria=strongest_criteria,
        improvement_suggestions=improvement_suggestions
    )

@app.get("/")
async def root():
    return {"message": "ML-Enhanced O-1A Visa Qualification Assessment API. Go to /docs for API documentation."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)