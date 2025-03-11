import pandas as pd
import io
import re
import nltk
from tqdm import tqdm
import json
from enum import Enum
from typing import List, Dict, Any, Optional

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class QualificationLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

O1A_CRITERIA = {
    "Awards": [
        r"award", r"prize", r"medal", r"distinction", r"recognition", 
        r"honor", r"scholarship", r"grant", r"fellowship", r"recipient",
        r"won", r"winning", r"received", r"honored", r"recognized"
    ],
    "Membership": [
        r"member", r"association", r"society", r"organization", 
        r"committee", r"board", r"fellow", r"elected", r"admitted",
        r"invited", r"exclusive", r"prestigious", r"appointment"
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

def extract_sentences(text):
    """Split text into sentences using NLTK"""
    try:
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]
    except:
        text = re.sub(r'\n+', '. ', text)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

def find_criterion_matches(text, sentences):
    matches_by_criterion = {}
    
    # Process each criterion
    for criterion, patterns in O1A_CRITERIA.items():
        matched_sentences = []
        
        # Check each sentence for potential matches
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for pattern in patterns:
                if re.search(pattern.lower(), sentence_lower):
                    # Avoid duplicate sentences
                    if sentence not in matched_sentences:
                        matched_sentences.append(sentence)
                    break
        
        # Calculate a simple confidence score based on number of matches
        confidence = min(0.9, len(matched_sentences) / 10.0 + 0.3) if matched_sentences else 0.0
        
        # Store matches and confidence
        matches_by_criterion[criterion] = {
            "matches": matched_sentences,
            "confidence": confidence
        }
    
    return matches_by_criterion
    
# Determine qualification rating based on criteria matches and confidence"""
def calculate_qualification_rating(matches_by_criterion):
    # Count criteria with at least one match and good confidence
    strong_criteria = sum(1 for criterion, data in matches_by_criterion.items() 
                         if len(data["matches"]) > 0 and data["confidence"] >= 0.7)
    
    medium_criteria = sum(1 for criterion, data in matches_by_criterion.items() 
                         if len(data["matches"]) > 0 and 0.4 <= data["confidence"] < 0.7)
    
    weak_criteria = sum(1 for criterion, data in matches_by_criterion.items() 
                        if len(data["matches"]) > 0 and data["confidence"] < 0.4)
    
    weighted_score = (strong_criteria * 1.0) + (medium_criteria * 0.5) + (weak_criteria * 0.2)
    
    if (strong_criteria >= 3 or weighted_score >= 3.5):
        return QualificationLevel.HIGH.value
    elif (strong_criteria >= 2 or weighted_score >= 2.0):
        return QualificationLevel.MEDIUM.value
    else:
        return QualificationLevel.LOW.value

def process_cv(resume_text):
    sentences = extract_sentences(resume_text)
    
    matches_by_criterion = find_criterion_matches(resume_text, sentences)
    
    qualification_rating = calculate_qualification_rating(matches_by_criterion)
    
    return {
        "matches_by_criterion": matches_by_criterion,
        "qualification_rating": qualification_rating
    }

def main():
    """Process the CSV dataset and add assessment columns"""
    print("Loading dataset...")
    try:
        df = pd.read_csv("data/cv_dataset.csv")
        print(f"Loaded {len(df)} CVs.")
    except Exception as e:
        print(f"Error loading CSV: {str(e)}")
        return
    
    print("Processing CVs...")
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            resume_text = row["Resume_str"]
            
            # Error handling: skip empty resumes
            if pd.isna(resume_text) or resume_text.strip() == "":
                results.append({
                    "criteria_matches": json.dumps({}),
                    "qualification_rating": QualificationLevel.LOW.value
                })
                continue
            
            assessment = process_cv(resume_text)
            
            results.append({
                "criteria_matches": json.dumps(assessment["matches_by_criterion"]),
                "qualification_rating": assessment["qualification_rating"]
            })
        except Exception as e:
            print(f"Error processing CV {idx}: {str(e)}")
            results.append({
                "criteria_matches": json.dumps({}),
                "qualification_rating": QualificationLevel.LOW.value
            })
    
    print("Adding results to dataframe...")
    result_df = pd.DataFrame(results)
    df["criteria_matches"] = result_df["criteria_matches"]
    df["qualification_rating"] = result_df["qualification_rating"]
    
    print("Saving enhanced dataset...")
    df.to_csv("data/cv_dataset_enhanced.csv", index=False)
    print("Processing complete!")

if __name__ == "__main__":
    main()