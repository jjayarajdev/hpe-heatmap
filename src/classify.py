"""
Service classification models for opportunity categorization.

This module provides baseline and advanced models to classify opportunities
into service types with confidence scores and evaluation metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
import pickle
from pathlib import Path
import json
import re
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, 
    confusion_matrix, roc_auc_score, precision_recall_curve
)
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import xgboost as xgb
from collections import Counter

from .utils import config, logger, Timer, save_metrics
from .features import FeatureEngineer, TextVectorizer
from .focus_area_integration import FocusAreaIntegrator


class FocusAreaClassifier:
    """Classifier for Focus Area prediction."""
    
    def __init__(self):
        self.integrator = FocusAreaIntegrator()
        self.model = None
        self.vectorizer = None
        
    def predict(self, text: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Predict Focus Areas for given text."""
        return self.integrator.classify_text_to_focus_area(text, top_k)
    
    def predict_batch(self, texts: List[str]) -> List[List[Tuple[str, float]]]:
        """Predict Focus Areas for multiple texts."""
        return [self.predict(text) for text in texts]


class ServiceClassifier:
    """Multi-class and multi-label service classification."""
    
    def __init__(self, classifier_type: str = "xgboost"):
        self.classifier_type = classifier_type
        self.model = None
        self.feature_engineer = None
        self.label_encoder = None
        self.multilabel_binarizer = None
        self.is_multilabel = False
        self.classes_ = None
        self.feature_names = []
        
    def train(self, df_opps: pd.DataFrame, 
              text_columns: List[str],
              label_column: str,
              test_size: float = 0.2) -> Tuple['ServiceClassifier', Dict[str, Any]]:
        """
        Train service classifier on opportunity data.
        
        Args:
            df_opps: DataFrame with opportunities
            text_columns: List of text columns to use as features
            label_column: Column containing service labels
            test_size: Fraction of data for testing
            
        Returns:
            Tuple of (fitted classifier, metrics dict)
        """
        logger.info(f"Training {self.classifier_type} service classifier")
        
        # Prepare data
        X, y, metrics = self._prepare_training_data(df_opps, text_columns, label_column)
        
        if X is None:
            raise ValueError("Failed to prepare training data")
        
        # Split data
        # Check if we can use stratification
        use_stratify = False
        if not self.is_multilabel:
            # Check if each class has at least 2 samples
            unique, counts = np.unique(y, return_counts=True)
            if np.min(counts) >= 2:
                use_stratify = True
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=config.random_seed,
            stratify=y if use_stratify else None
        )
        
        logger.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
        
        # Train model
        with Timer(logger, f"{self.classifier_type} training"):
            self.model = self._create_model()
            self.model.fit(X_train, y_train)
        
        # Evaluate model
        evaluation_metrics = self._evaluate_model(X_test, y_test)
        
        # Cross-validation
        cv_scores = self._cross_validate(X, y)
        
        # Combine all metrics
        final_metrics = {
            **metrics,
            **evaluation_metrics,
            "cross_validation": cv_scores,
            "model_type": self.classifier_type,
            "training_samples": X_train.shape[0],
            "test_samples": X_test.shape[0],
            "n_features": X.shape[1]
        }
        
        logger.info(f"Model trained successfully. Test F1: {evaluation_metrics.get('macro_f1', 0):.3f}")
        
        return self, final_metrics
    
    def _prepare_training_data(self, df: pd.DataFrame, 
                             text_columns: List[str],
                             label_column: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:
        """Prepare training data from DataFrame."""
        logger.info("Preparing training data")
        
        # Check if label column exists
        if label_column not in df.columns:
            logger.error(f"Label column '{label_column}' not found")
            return None, None, {}
        
        # Remove rows with missing labels
        df_clean = df.dropna(subset=[label_column]).copy()
        
        if len(df_clean) == 0:
            logger.error("No rows with valid labels found")
            return None, None, {}
        
        # Analyze labels to determine if multi-label
        labels = df_clean[label_column]
        self.is_multilabel = self._detect_multilabel(labels)
        
        logger.info(f"Detected {'multi-label' if self.is_multilabel else 'multi-class'} problem")
        
        # Prepare features
        self.feature_engineer = FeatureEngineer()
        
        # Check if text columns exist
        available_text_columns = [col for col in text_columns if col in df_clean.columns]
        if not available_text_columns:
            logger.error(f"No text columns found: {text_columns}")
            return None, None, {}
        
        X, feature_names = self.feature_engineer.engineer_features(
            df_clean,
            text_columns=available_text_columns,
            categorical_columns=[],  # Focus on text for classification
            numeric_columns=[],
            create_interactions=False
        )
        
        self.feature_names = feature_names
        
        # Prepare labels
        if self.is_multilabel:
            # Multi-label encoding
            self.multilabel_binarizer = MultiLabelBinarizer()
            y = self.multilabel_binarizer.fit_transform(labels)
            self.classes_ = self.multilabel_binarizer.classes_
        else:
            # Multi-class encoding
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(labels.astype(str))
            self.classes_ = self.label_encoder.classes_
        
        # Prepare metrics
        prep_metrics = {
            "original_samples": len(df),
            "clean_samples": len(df_clean),
            "n_classes": len(self.classes_),
            "is_multilabel": self.is_multilabel,
            "class_distribution": self._get_class_distribution(labels),
            "feature_stats": {
                "n_features": X.shape[1],
                "feature_sparsity": np.count_nonzero(X == 0) / X.size,
                "feature_variance": np.var(X, axis=0).mean()
            }
        }
        
        return X, y, prep_metrics
    
    def _detect_multilabel(self, labels: pd.Series) -> bool:
        """Detect if this is a multi-label problem."""
        # Check if labels contain delimiters indicating multiple labels
        sample_labels = labels.dropna().astype(str).head(100)
        
        delimiters = [',', ';', '|', '&', 'and']
        multilabel_indicators = 0
        
        for label in sample_labels:
            for delimiter in delimiters:
                if delimiter in label:
                    multilabel_indicators += 1
                    break
        
        # If >20% of samples have delimiters, likely multi-label
        return (multilabel_indicators / len(sample_labels)) > 0.2
    
    def _get_class_distribution(self, labels: pd.Series) -> Dict[str, Any]:
        """Get class distribution statistics."""
        if self.is_multilabel:
            # For multi-label, split on common delimiters
            all_labels = []
            for label in labels.dropna().astype(str):
                label_parts = re.split(r'[,;|&]|and', label)
                all_labels.extend([part.strip() for part in label_parts if part.strip()])
            
            label_counts = Counter(all_labels)
            total_labels = sum(label_counts.values())
            
            return {
                "unique_labels": len(label_counts),
                "total_label_instances": total_labels,
                "avg_labels_per_sample": total_labels / len(labels),
                "most_common": label_counts.most_common(10),
                "imbalance_ratio": label_counts.most_common(1)[0][1] / max(label_counts.most_common()[-1][1], 1)
            }
        else:
            # Single-label distribution
            label_counts = labels.value_counts()
            
            return {
                "unique_classes": len(label_counts),
                "most_common_class": label_counts.index[0],
                "least_common_class": label_counts.index[-1],
                "class_counts": label_counts.to_dict(),
                "imbalance_ratio": label_counts.iloc[0] / label_counts.iloc[-1]
            }
    
    def _create_model(self):
        """Create classifier model based on type."""
        if self.classifier_type == "logistic":
            return LogisticRegression(
                random_state=config.random_seed,
                max_iter=1000,
                multi_class='ovr' if self.is_multilabel else 'auto'
            )
        
        elif self.classifier_type == "svm":
            return LinearSVC(
                random_state=config.random_seed,
                max_iter=2000
            )
        
        elif self.classifier_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                random_state=config.random_seed,
                n_jobs=config.n_jobs
            )
        
        elif self.classifier_type == "xgboost":
            if self.is_multilabel:
                return xgb.XGBClassifier(
                    objective='multi:softprob',
                    random_state=config.random_seed,
                    n_jobs=config.n_jobs
                )
            else:
                return xgb.XGBClassifier(
                    objective='multi:softprob',
                    random_state=config.random_seed,
                    n_jobs=config.n_jobs
                )
        
        else:
            raise ValueError(f"Unsupported classifier type: {self.classifier_type}")
    
    def _evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate trained model on test data."""
        logger.info("Evaluating model performance")
        
        # Get predictions
        y_pred = self.model.predict(X_test)
        
        # Get prediction probabilities
        try:
            y_pred_proba = self.model.predict_proba(X_test)
        except:
            y_pred_proba = None
        
        metrics = {}
        
        if self.is_multilabel:
            # Multi-label metrics
            metrics["accuracy"] = accuracy_score(y_test, y_pred)
            metrics["macro_f1"] = f1_score(y_test, y_pred, average='macro')
            metrics["micro_f1"] = f1_score(y_test, y_pred, average='micro')
            metrics["weighted_f1"] = f1_score(y_test, y_pred, average='weighted')
            
            # Per-class metrics
            class_report = classification_report(y_test, y_pred, 
                                               target_names=self.classes_,
                                               output_dict=True, zero_division=0)
            metrics["per_class_metrics"] = class_report
            
        else:
            # Multi-class metrics
            metrics["accuracy"] = accuracy_score(y_test, y_pred)
            metrics["macro_f1"] = f1_score(y_test, y_pred, average='macro')
            metrics["weighted_f1"] = f1_score(y_test, y_pred, average='weighted')
            
            # Classification report
            class_report = classification_report(y_test, y_pred, 
                                               target_names=self.classes_,
                                               output_dict=True, zero_division=0)
            metrics["per_class_metrics"] = class_report
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            metrics["confusion_matrix"] = cm.tolist()
            
            # ROC-AUC if binary or if we have probabilities
            if len(self.classes_) == 2 and y_pred_proba is not None:
                try:
                    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                    metrics["roc_auc"] = auc
                except:
                    pass
        
        return metrics
    
    def _cross_validate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Perform cross-validation."""
        logger.info("Performing cross-validation")
        
        try:
            cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, 
                               random_state=config.random_seed)
            
            # Calculate CV scores
            cv_scores = cross_val_score(self.model, X, y, cv=cv, 
                                      scoring='f1_macro', n_jobs=config.n_jobs)
            
            return {
                "cv_scores": cv_scores.tolist(),
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "cv_folds": config.cv_folds
            }
            
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            return {"error": str(e)}
    
    def predict_services(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Predict services for given text.
        
        Args:
            text: Input text to classify
            top_k: Number of top predictions to return
            
        Returns:
            List of (service_name, confidence_score) tuples
        """
        if not self.model or not self.feature_engineer:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create dummy DataFrame for feature engineering
        dummy_df = pd.DataFrame({col: [text] for col in self.feature_engineer.feature_names[:1]})
        
        # Engineer features (this is simplified - in practice, we'd need the original text columns)
        try:
            # Use the text vectorizer directly
            if hasattr(self.feature_engineer, 'text_vectorizer') and self.feature_engineer.text_vectorizer:
                X = self.feature_engineer.text_vectorizer.transform([text])
            else:
                # Fallback to simple TF-IDF
                from sklearn.feature_extraction.text import TfidfVectorizer
                vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                X = vectorizer.fit_transform([text, text]).toarray()[:1]  # Fit needs >1 sample
        
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            return []
        
        # Get predictions
        try:
            if self.is_multilabel:
                # Multi-label prediction
                y_pred_proba = self.model.predict_proba(X)[0]
                
                # Get top-k predictions
                top_indices = np.argsort(y_pred_proba)[-top_k:][::-1]
                predictions = [(self.classes_[i], y_pred_proba[i]) for i in top_indices]
                
            else:
                # Multi-class prediction
                y_pred_proba = self.model.predict_proba(X)[0]
                
                # Get top-k predictions
                top_indices = np.argsort(y_pred_proba)[-top_k:][::-1]
                predictions = [(self.classes_[i], y_pred_proba[i]) for i in top_indices]
            
            # Filter out low-confidence predictions
            predictions = [(name, score) for name, score in predictions if score > 0.1]
            
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return []
    
    def save_model(self, filepath: str):
        """Save trained classifier."""
        model_data = {
            "model": self.model,
            "feature_engineer": self.feature_engineer,
            "label_encoder": self.label_encoder,
            "multilabel_binarizer": self.multilabel_binarizer,
            "is_multilabel": self.is_multilabel,
            "classes_": self.classes_,
            "feature_names": self.feature_names,
            "classifier_type": self.classifier_type
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained classifier."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data["model"]
        self.feature_engineer = model_data["feature_engineer"]
        self.label_encoder = model_data.get("label_encoder")
        self.multilabel_binarizer = model_data.get("multilabel_binarizer")
        self.is_multilabel = model_data["is_multilabel"]
        self.classes_ = model_data["classes_"]
        self.feature_names = model_data["feature_names"]
        self.classifier_type = model_data["classifier_type"]
        
        logger.info(f"Model loaded from {filepath}")


def train_service_classifier(df_opps: pd.DataFrame, 
                           text_columns: List[str] = None,
                           label_column: str = None) -> Tuple[ServiceClassifier, Dict[str, Any]]:
    """
    Train and return a fitted classifier and metrics.
    
    Args:
        df_opps: DataFrame with opportunity data
        text_columns: List of text columns to use (auto-detected if None)
        label_column: Label column name (auto-detected if None)
        
    Returns:
        Tuple of (fitted classifier, metrics dict)
    """
    logger.info("Training service classifier")
    
    # Auto-detect columns if not provided
    if text_columns is None:
        text_columns = [col for col in df_opps.select_dtypes(include=['object']).columns
                       if df_opps[col].astype(str).str.len().mean() > 10]
        logger.info(f"Auto-detected text columns: {text_columns}")
    
    if label_column is None:
        # Look for columns that might contain service labels
        potential_label_cols = [col for col in df_opps.columns
                              if any(term in col.lower() for term in 
                                   ['service', 'category', 'type', 'class', 'label', 'practice', 'domain'])]
        
        if potential_label_cols:
            label_column = potential_label_cols[0]
            logger.info(f"Auto-detected label column: {label_column}")
        else:
            # Fallback to any column that might represent groupings
            fallback_cols = [col for col in df_opps.columns
                           if df_opps[col].nunique() < len(df_opps) * 0.5 and df_opps[col].nunique() > 1]
            
            if fallback_cols:
                label_column = fallback_cols[0]
                logger.warning(f"Using fallback label column: {label_column}")
            else:
                raise ValueError("No suitable label column found. Please specify label_column.")
    
    # Train baseline models
    results = {}
    best_classifier = None
    best_f1 = 0
    
    classifier_types = ["logistic", "xgboost"]
    
    for clf_type in classifier_types:
        try:
            logger.info(f"Training {clf_type} classifier")
            
            classifier = ServiceClassifier(classifier_type=clf_type)
            trained_classifier, metrics = classifier.train(
                df_opps, text_columns, label_column
            )
            
            results[clf_type] = metrics
            
            # Track best model
            f1_score = metrics.get("macro_f1", 0)
            if f1_score > best_f1:
                best_f1 = f1_score
                best_classifier = trained_classifier
            
        except Exception as e:
            logger.error(f"Training {clf_type} failed: {e}")
            results[clf_type] = {"error": str(e)}
    
    if best_classifier is None:
        raise ValueError("All classifier training failed")
    
    # Save best model
    model_path = Path("models") / "service_classifier.pkl"
    model_path.parent.mkdir(exist_ok=True)
    best_classifier.save_model(str(model_path))
    
    # Save label space
    label_space = {
        "classes": best_classifier.classes_.tolist() if hasattr(best_classifier.classes_, 'tolist') else list(best_classifier.classes_),
        "is_multilabel": best_classifier.is_multilabel,
        "n_classes": len(best_classifier.classes_)
    }
    
    label_path = Path("artifacts") / "label_space.json"
    with open(label_path, 'w') as f:
        json.dump(label_space, f, indent=2)
    
    # Save training results
    training_results = {
        "best_model": best_classifier.classifier_type,
        "best_f1": best_f1,
        "all_results": results,
        "model_path": str(model_path),
        "label_space_path": str(label_path)
    }
    
    save_metrics(training_results, "classification_training.json")
    
    logger.info(f"Best classifier: {best_classifier.classifier_type} (F1: {best_f1:.3f})")
    
    return best_classifier, training_results


def predict_services(text: str, top_k: int = 5, 
                    model_path: str = "models/service_classifier.pkl") -> List[Tuple[str, float]]:
    """
    Predict services for given text using trained model.
    
    Args:
        text: Input text to classify
        top_k: Number of top predictions to return
        model_path: Path to trained model
        
    Returns:
        List of (service_name, confidence_score) tuples
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return []
    
    try:
        # Load model (it's a simple dictionary with pipeline)
        import joblib
        classifier = joblib.load(model_path)
        
        if classifier.get('type') == 'pipeline':
            # Use the pipeline directly
            pipeline = classifier['pipeline']
            classes = classifier.get('classes', [])
            
            # Make prediction
            prediction = pipeline.predict([text])[0]
            probabilities = pipeline.predict_proba([text])[0]
            
            # Get top k predictions
            results = []
            class_probs = list(zip(classes, probabilities))
            class_probs.sort(key=lambda x: x[1], reverse=True)
            
            for class_name, prob in class_probs[:top_k]:
                results.append((str(class_name), float(prob)))
            
            return results
        else:
            logger.error(f"Unknown model type: {classifier.get('type', 'unknown')}")
            return []
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return []


def evaluate_classification_quality(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate if classification meets quality gates."""
    quality_assessment = {
        "passes_quality_gates": True,
        "quality_issues": [],
        "recommendations": []
    }
    
    # Check macro F1 threshold
    macro_f1 = metrics.get("macro_f1", 0)
    if macro_f1 < config.min_macro_f1:
        quality_assessment["passes_quality_gates"] = False
        quality_assessment["quality_issues"].append(
            f"Macro F1 ({macro_f1:.3f}) below threshold ({config.min_macro_f1})"
        )
        quality_assessment["recommendations"].extend([
            "Consider feature engineering improvements",
            "Check class imbalance and apply resampling",
            "Try different model architectures",
            "Increase training data if possible"
        ])
    
    # Check class imbalance
    class_dist = metrics.get("class_distribution", {})
    imbalance_ratio = class_dist.get("imbalance_ratio", 1)
    
    if imbalance_ratio > 10:
        quality_assessment["quality_issues"].append(
            f"High class imbalance (ratio: {imbalance_ratio:.1f})"
        )
        quality_assessment["recommendations"].append(
            "Consider class balancing techniques (SMOTE, class weights)"
        )
    
    # Check feature quality
    feature_stats = metrics.get("feature_stats", {})
    sparsity = feature_stats.get("feature_sparsity", 0)
    
    if sparsity > 0.95:
        quality_assessment["quality_issues"].append(
            f"Very sparse features (sparsity: {sparsity:.2f})"
        )
        quality_assessment["recommendations"].append(
            "Consider feature selection or dimensionality reduction"
        )
    
    return quality_assessment


def main():
    """Main function to run classification pipeline."""
    logger.info("Starting classification pipeline")
    
    # Load processed data
    from .io_loader import load_processed_data
    data = load_processed_data()
    
    if not data:
        logger.error("No processed data found. Run io_loader first.")
        return
    
    # Find opportunity data - prefer original over cleaned for more samples
    opportunity_df = None
    
    # First try to find original (larger) datasets
    for name, df in data.items():
        if ('opportunity' in name.lower() or 'request' in name.lower() or 'rawdata' in name.lower()) and 'clean' not in name.lower():
            opportunity_df = df
            logger.info(f"Using {name} as opportunity data")
            break
    
    # Fallback to cleaned data if original not found
    if opportunity_df is None:
        for name, df in data.items():
            if 'opportunity' in name.lower() or 'request' in name.lower() or 'rawdata' in name.lower():
                opportunity_df = df
                logger.info(f"Using {name} as opportunity data (fallback)")
                break
    
    if opportunity_df is None:
        logger.error("No opportunity data found for classification")
        return
    
    # Train classifier
    try:
        classifier, metrics = train_service_classifier(opportunity_df)
        
        # Evaluate quality
        quality_assessment = evaluate_classification_quality(metrics)
        
        # Save quality assessment
        save_metrics(quality_assessment, "classification_quality.json")
        
        # Log quality results
        if quality_assessment["passes_quality_gates"]:
            logger.info("✅ Classification meets all quality gates")
        else:
            logger.warning("⚠️ Classification quality issues found:")
            for issue in quality_assessment["quality_issues"]:
                logger.warning(f"  - {issue}")
            
            logger.info("💡 Recommendations:")
            for rec in quality_assessment["recommendations"]:
                logger.info(f"  - {rec}")
        
        logger.info("Classification pipeline completed successfully")
        
        return classifier, metrics
        
    except Exception as e:
        logger.error(f"Classification pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
