import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib

def main():
    print("Loading enhanced dataset...")
    df = pd.read_csv("data/cv_dataset_enhanced.csv")
    print(f"Loaded {len(df)} CVs with ratings and criteria matches.")
    
    # Ensure we have no missing values
    df = df.dropna(subset=["Resume_str", "qualification_rating"])
    
    X = df["Resume_str"]
    y = df["qualification_rating"]
    
    df["criteria_count"] = df["criteria_matches"].apply(
        lambda x: len([c for c in json.loads(x).items() if len(c[1]["matches"]) > 0]) if x and x != '{}' else 0
    )
    
    df["strong_criteria_count"] = df["criteria_matches"].apply(
        lambda x: len([c for c in json.loads(x).items() 
                      if len(c[1]["matches"]) > 0 and c[1]["confidence"] >= 0.7]) if x and x != '{}' else 0
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    print("Class distribution:")
    print(y.value_counts())
    
    print("Training text-based model...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    print("\nModel performance on test set:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    
    print("\nTraining enhanced model with criteria features...")
    
    feature_cols = ["criteria_count", "strong_criteria_count"]
    X_features_train = df.loc[X_train.index, feature_cols]
    X_features_test = df.loc[X_test.index, feature_cols]
    
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_tfidf_train = tfidf.fit_transform(X_train)
    X_tfidf_test = tfidf.transform(X_test)
    
    X_combined_train = np.hstack((X_tfidf_train.toarray(), X_features_train.values))
    X_combined_test = np.hstack((X_tfidf_test.toarray(), X_features_test.values))
    
    enhanced_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    enhanced_model.fit(X_combined_train, y_train)
    
    y_pred_enhanced = enhanced_model.predict(X_combined_test)
    print("\nEnhanced model performance on test set:")
    print(classification_report(y_test, y_pred_enhanced))
    print(f"Accuracy: {accuracy_score(y_test, y_pred_enhanced):.2f}")
    
    print("\nSaving models...")
    joblib.dump(pipeline, "models/o1a_text_model.joblib")
    joblib.dump(enhanced_model, "models/o1a_enhanced_model.joblib")
    joblib.dump(tfidf, "models/o1a_tfidf_vectorizer.joblib")
    
    print("Training complete! Models saved.")

if __name__ == "__main__":
    main()