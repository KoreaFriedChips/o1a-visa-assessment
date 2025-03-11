# AI-Enhanced O-1A Visa Qualification Assessment Tool

An advanced AI/ML system that analyzes CVs to assess qualification potential for O-1A extraordinary ability visas, based on official USCIS guidelines.

## Project Overview

This tool uses natural language processing and machine learning to analyze resumes/CVs and determine an applicant's potential qualification for the O-1A visa category. It extracts evidence for each of the 8 O-1A criteria, provides a qualification rating, and offers personalized suggestions for improving chances of approval.

## Data Source

The system is trained using a dataset of 2400+ CVs from Kaggle that were taken from livecareer.com:

- **Dataset**: Resume Dataset
- **URL**: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset
- **Format**: CSV with columns for ID, Resume_str, Resume_html, and Category

This dataset is processed to extract O-1A criteria matches and generate initial qualification ratings, creating a labeled dataset for machine learning.

## Project Architecture

This project consists of three main components:

1. **Data Processing Pipeline**: Processes the Kaggle CV dataset to extract O-1A criteria matches and generate initial qualification ratings

- At first I wanted to create a synthetic dataset of CVs and give each a label and list of things that the person has done. Obviously this would be very time consuming and slow. Given that Kaggle has a dataset of CVs already we just need to create 2 more columns (rating and reasoning).

2. **Machine Learning Model**: Trains on the processed data to create a predictive model for O-1A qualification assessment

- I chose to use a hybrid approach with an enhanced Gradient Boosting Classifier that combines TF-IDF text features with structured criteria features. The Gradient Boosting handles the imbalanced distribution of qualification ratings better than alternatives, provides feature importance rankings that help explain classifications, and delivers strong performance with reasonable computational requirements, making it suitable for real-time API deployment.

3. **FastAPI Application**: Serves the trained model via a REST API that accepts CV uploads and returns detailed assessments

## O-1A Visa Criteria (Based on USCIS Guidelines)

The system analyzes CVs for evidence supporting these 8 criteria as defined by USCIS in [8 CFR 214.2(o)(3)(iii)](https://www.uscis.gov/policy-manual/volume-2-part-m-chapter-4):

1. **Awards**: Receipt of nationally or internationally recognized prizes/awards for excellence in the field
2. **Membership**: Membership in associations requiring outstanding achievement, as judged by recognized experts in the field
3. **Press**: Published material in professional or major trade publications or major media about the person and their work
4. **Judging**: Participation as a judge (individually or on a panel) of the work of others in the same or allied field
5. **Original contribution**: Original scientific, scholarly, or business-related contributions of major significance in the field
6. **Scholarly articles**: Authorship of scholarly articles in professional journals or other major media in the field
7. **Critical employment**: Employment in a critical or essential capacity at organizations with distinguished reputations
8. **High remuneration**: Evidence of commanding a high salary or other significantly high remuneration for services

According to USCIS guidelines, applicants must satisfy at least 3 of these 8 criteria to qualify for O-1A status. However, simply "checking the boxes" is not sufficient - the evidence must cumulatively establish the applicant's sustained national or international acclaim and recognition for achievements.

## Design Decisions Based on USCIS Guidelines

Several key design decisions in this application directly reflect USCIS's official guidance:

1. **Weighted Scoring System**: Rather than simply counting criteria, the system uses a weighted approach that considers both quantity and quality of evidence. This reflects USCIS's "totality of the evidence" approach described in the [Policy Manual Volume 2, Part M, Chapter 4](https://www.uscis.gov/policy-manual/volume-2-part-m-chapter-4).

2. **Impact Assessment**: The tool analyzes the significance and scale of achievements, not just their presence. This aligns with USCIS's guidance that evidence must demonstrate "a level of expertise indicating that the person is one of the small percentage who have risen to the very top of the field of endeavor."

3. **Field-Specific Patterns**: The keyword patterns incorporate field-specific terminology, acknowledging that USCIS evaluates achievement "relative to the specific field in which the beneficiary has been demonstrated to have extraordinary ability."

4. **Extraordinary Ability Standard**: The qualification rating system is calibrated to reflect USCIS's high standard that O-1A applicants must demonstrate "sustained national or international acclaim" and be among "the small percentage who have risen to the very top of their field."

5. **Comparative Evidence**: The system looks for comparative indicators (e.g., "top", "first", "best") that help establish an applicant is at the top of their field, as required by USCIS.

## Implementation Details

### Technology Stack

- **FastAPI**: Modern web framework for building APIs
- **scikit-learn**: Machine learning for classification
- **NLTK**: Natural language processing for text analysis
- **PyPDF2/python-docx**: Document parsing
- **pandas**: Data manipulation for training datasets

### Key Features

- **Dual-approach assessment**: Uses both rule-based pattern matching and machine learning classification
- **Impact analysis**: Evaluates the significance and scale of achievements
- **Personalized recommendations**: Generates specific suggestions for improving qualification chances
- **Confidence scoring**: Provides confidence levels for assessment results
- **Semi-supervised learning**: Uses rule-based system to generate training data for ML model

## Installation & Setup

### Prerequisites

- Python 3.9+
- pip (Python package manager)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/KoreaFriedChips/o1a-visa-assessment
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

### Processing the CV Dataset

If you have the Kaggle CSV dataset of CVs and want to process it to create training data:

```bash
python process_csv_dataset.py
```

This will:

1. Read the Kaggle CSV with CV data
2. Process each CV through the rule-based assessment system
3. Generate qualification ratings and criteria matches
4. Save results to a new enhanced CSV file

### Training the ML Model

After processing the dataset:

```bash
python train_ml_model.py
```

This will:

1. Load the enhanced dataset
2. Train a text-based model and an enhanced feature-based model
3. Evaluate model performance
4. Save the trained models for use in the API

### Running the API

```bash
uvicorn main:app --reload
```

The API will be available at http://localhost:8000, with interactive documentation at http://localhost:8000/docs.

## Usage Guide

### API Endpoints

The API exposes a single endpoint for CV assessment:

```
POST /assess-o1a-qualification
```

This endpoint accepts a file upload (PDF or DOCX CV) and returns a detailed qualification assessment.

### Example Request (using curl)

```bash
curl -X 'POST' \
  'http://localhost:8000/assess-o1a-qualification' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@resume.pdf'
```

### Example Response

```json
{
  "matches_by_criterion": [
    {
      "criterion": "Awards",
      "matches": [
        "Recipient of the Distinguished Researcher Award in Artificial Intelligence, 2022",
        "Awarded $500,000 research grant from National Science Foundation"
      ],
      "confidence": 0.7,
      "impact": {
        "description": "Quantifiable impact: $500,000",
        "significance": "Substantial",
        "scale": "Significant"
      }
    },
    {
      "criterion": "Membership",
      "matches": ["Elected member of the IEEE AI Ethics Committee"],
      "confidence": 0.5,
      "impact": {
        "description": "Impact described in qualitative terms",
        "significance": "Substantial",
        "scale": "Significant"
      }
    }
  ],
  "qualification_rating": "medium",
  "justification": "The candidate shows strong evidence for 2 criteria (Awards, Scholarly articles) and moderate evidence for 3 more. While the minimum requirement is 3 criteria, the strength of evidence suggests a moderate chance of qualification, though additional documentation may be beneficial to strengthen the application. The system has moderate confidence in this assessment.",
  "strongest_criteria": ["Awards", "Scholarly articles"],
  "improvement_suggestions": [
    "Consider joining prestigious professional associations in your field that require nomination or selective admission.",
    "Seek opportunities for media coverage of your work in industry publications.",
    "Document how your work has created new processes or approaches that others have adopted.",
    "Emphasize leadership roles where you've been essential to projects or organizations.",
    "Quantify your impact with specific metrics (users affected, efficiency improved, etc.)."
  ]
}
```

## Development Process

### 1. Initial Rule-Based System

The initial system uses pattern matching to:

- Detect keywords associated with each O-1A criterion
- Extract matching sentences from the CV
- Calculate confidence scores based on match quantity and quality
- Determine qualification rating using weighted criteria

### 2. Dataset Enhancement

The rule-based system processes the Kaggle CV dataset to:

- Extract matched criteria for each CV
- Generate qualification ratings
- Create a labeled dataset for supervised learning

### 3. ML Model Training

Two models are trained:

- A basic text-classification model using TF-IDF and Random Forest
- An enhanced model that combines text features with extracted criteria features

### 4. Final API Implementation

The API:

- Parses uploaded CV documents
- Uses both rule-based and ML approaches
- Falls back to rule-based assessment if ML models aren't available
- Returns detailed assessment with confidence scores
- Provides actionable improvement suggestions

## Alignment with O-1A Adjudication Process

This tool is designed to mirror the actual USCIS adjudication process as described in the [USCIS Policy Manual](https://www.uscis.gov/policy-manual/volume-2-part-m-chapter-4):

1. **Two-Part Analysis**: The system first identifies which criteria are met, then evaluates whether the totality of evidence demonstrates extraordinary ability, similar to the USCIS two-part analysis approach.

2. **Comparative Evidence**: Following USCIS guidance that "evidence must be indicative of a level of expertise indicating that the person is one of the small percentage who have risen to the very top of the field of endeavor," the system looks for indicators of exceptional standing.

3. **Field-Specific Context**: The system considers field-specific terminology and achievements, reflecting USCIS's approach to evaluate "the beneficiary in relation to others in the field."

4. **Beyond Minimum Criteria**: Simply meeting 3 criteria may not be sufficient for approval if the evidence doesn't demonstrate national or international acclaim. The system's weighted scoring and impact analysis reflect this nuanced approach.

## Evaluation Metrics

The ML model's performance can be evaluated using:

- Accuracy
- Precision, recall, and F1 score for each qualification level
- Confusion matrix analysis

## Future Improvements

1. **UI Development**: Create a frontend interface for non-technical users
2. **Expert Validation**: Compare system assessments with immigration attorney evaluations
3. **Additional Features**: Extract more nuanced features like institution prestige and role impact
4. **Fine-tuned LLM**: Implement a fine-tuned language model for more sophisticated text understanding
5. **Feedback Loop**: Add a mechanism for users to provide feedback on assessments to improve the system
6. **Field-Specific Models**: Train separate models for different fields (sciences, arts, business, etc.) to reflect field-specific standards of achievement

## Project Structure

```
o1a-visa-assessment/
├── main.py                   # FastAPI application
├── process_csv_dataset.py    # Script to process CV dataset
├── train_ml_model.py         # ML training script
├── requirements.txt          # Python dependencies
├── models/                   # Saved ML models
│   ├── o1a_text_model.joblib
│   ├── o1a_enhanced_model.joblib
│   └── o1a_tfidf_vectorizer.joblib
├── data/                     # Data files
│   ├── cv_dataset.csv        # Original Kaggle dataset
│   └── cv_dataset_enhanced.csv  # Processed dataset with criteria matches
└── README.md
```

## Disclaimer

This tool provides only a preliminary assessment and should not replace professional legal advice. According to [USCIS guidelines](https://www.uscis.gov/working-in-the-united-states/temporary-workers/o-1-visa-individuals-with-extraordinary-ability-or-achievement), O-1A visa applications require substantial documentation and expert guidance. The actual O-1A visa application is complex and involves detailed documentation beyond what a CV alone can indicate.

## References

1. [USCIS: O-1 Visa: Individuals with Extraordinary Ability or Achievement](https://www.uscis.gov/working-in-the-united-states/temporary-workers/o-1-visa-individuals-with-extraordinary-ability-or-achievement)
2. [USCIS Policy Manual, Volume 2, Part M](https://www.uscis.gov/policy-manual/volume-2-part-m)
3. [Alma O-1A Visa Guide](https://www.tryalma.com/o-1a-visa-guide)
4. [8 CFR 214.2(o)(3)(iii)](https://www.uscis.gov/policy-manual/volume-2-part-m-chapter-4) - Official regulatory criteria for O-1A classification

## License

MIT

## Acknowledgments

- Dataset provided by Kaggle user snehaanbhawal: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset
- O-1A visa criteria information based on official USCIS guidelines
