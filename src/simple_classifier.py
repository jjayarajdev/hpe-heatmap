"""
Simple working classifier for demonstration purposes.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
from pathlib import Path
import logging

from .utils import config, logger
from .io_loader import load_processed_data

def create_simple_classifier():
    """Create a simple working classifier for the dashboard."""
    logger.info("Creating simple classifier for demonstration")
    
    # Load data
    data = load_processed_data()
    
    # Find opportunity data
    opportunity_df = None
    for name, df in data.items():
        if 'opportunity' in name.lower() and 'clean' not in name.lower():
            opportunity_df = df
            break
    
    if opportunity_df is None or len(opportunity_df) < 10:
        logger.warning("Not enough opportunity data for training. Creating dummy classifier.")
        
        # Create a dummy classifier with predefined classes
        dummy_classifier = {
            "type": "dummy",
            "classes": ["Cloud Services", "Infrastructure Services", "Security Services", 
                       "Development Services", "Network Services"],
            "vectorizer": TfidfVectorizer(max_features=1000, stop_words='english')
        }
        
        # Fit vectorizer on dummy data
        dummy_texts = [
            "cloud aws azure deployment",
            "infrastructure server management", 
            "security vulnerability assessment",
            "software development programming",
            "network firewall configuration"
        ]
        dummy_classifier["vectorizer"].fit(dummy_texts)
        
        # Save dummy classifier
        model_path = Path("models/service_classifier.pkl")
        model_path.parent.mkdir(exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(dummy_classifier, f)
        
        logger.info("Dummy classifier created and saved")
        return dummy_classifier
    
    # Use actual data for training
    logger.info(f"Training on {len(opportunity_df)} opportunities")
    
    # Prepare text data
    text_columns = ['RR name', 'RR Project Name', 'Skill_Certification_Name']
    available_text_cols = [col for col in text_columns if col in opportunity_df.columns]
    
    if not available_text_cols:
        logger.error("No suitable text columns found")
        return None
    
    # Combine text
    combined_text = []
    for _, row in opportunity_df.iterrows():
        text_parts = []
        for col in available_text_cols:
            if pd.notna(row[col]):
                text_parts.append(str(row[col]))
        combined_text.append(' '.join(text_parts))
    
    # Use practice name as labels
    if 'RR Practice Name' in opportunity_df.columns:
        labels = opportunity_df['RR Practice Name'].fillna('Other')
    else:
        logger.error("No suitable label column found")
        return None
    
    # Create simple pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
        ('classifier', MultinomialNB())
    ])
    
    # Train
    try:
        # Simple train without test split for small data
        pipeline.fit(combined_text, labels)
        
        # Test on training data (for demonstration)
        y_pred = pipeline.predict(combined_text)
        accuracy = accuracy_score(labels, y_pred)
        
        logger.info(f"Simple classifier trained with accuracy: {accuracy:.3f}")
        
        # Save classifier
        model_path = Path("models/service_classifier.pkl")
        model_path.parent.mkdir(exist_ok=True)
        
        classifier_data = {
            "type": "pipeline",
            "pipeline": pipeline,
            "classes": list(pipeline.classes_),
            "accuracy": accuracy
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(classifier_data, f)
        
        logger.info("Simple classifier saved successfully")
        
        return classifier_data
        
    except Exception as e:
        logger.error(f"Simple classifier training failed: {e}")
        return None

def predict_with_simple_classifier(text: str, top_k: int = 5):
    """Predict using the simple classifier."""
    model_path = Path("models/service_classifier.pkl")
    
    if not model_path.exists():
        return []
    
    try:
        with open(model_path, 'rb') as f:
            classifier_data = pickle.load(f)
        
        if classifier_data["type"] == "dummy":
            # Dummy predictions
            classes = classifier_data["classes"]
            # Simple keyword-based predictions
            text_lower = text.lower()
            scores = []
            
            for cls in classes:
                score = 0.1  # Base score
                cls_lower = cls.lower()
                
                if 'cloud' in cls_lower and 'cloud' in text_lower:
                    score += 0.8
                elif 'security' in cls_lower and any(term in text_lower for term in ['security', 'vulnerability', 'firewall']):
                    score += 0.8
                elif 'infrastructure' in cls_lower and any(term in text_lower for term in ['infrastructure', 'server', 'system']):
                    score += 0.8
                elif 'development' in cls_lower and any(term in text_lower for term in ['development', 'programming', 'software']):
                    score += 0.8
                elif 'network' in cls_lower and 'network' in text_lower:
                    score += 0.8
                
                scores.append((cls, score))
            
            # Sort and return top k
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:top_k]
        
        elif classifier_data["type"] == "pipeline":
            # Real pipeline predictions
            pipeline = classifier_data["pipeline"]
            
            # Get prediction probabilities
            try:
                proba = pipeline.predict_proba([text])[0]
                classes = pipeline.classes_
                
                # Combine and sort
                predictions = list(zip(classes, proba))
                predictions.sort(key=lambda x: x[1], reverse=True)
                
                return predictions[:top_k]
                
            except:
                # Fallback to simple prediction
                pred = pipeline.predict([text])[0]
                return [(pred, 0.8)]
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return []

if __name__ == "__main__":
    create_simple_classifier()
