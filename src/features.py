"""
Feature engineering module for text processing and embeddings.

This module provides text featurization using TF-IDF and sentence embeddings,
categorical encoding, and feature engineering for ML models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from pathlib import Path
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from .utils import config, logger, Timer, save_metrics


class TextVectorizer:
    """Unified text vectorization supporting both TF-IDF and embeddings."""
    
    def __init__(self, mode: str = "auto"):
        """
        Initialize vectorizer.
        
        Args:
            mode: "tfidf", "embeddings", or "auto" (try embeddings, fallback to tfidf)
        """
        self.mode = mode
        self.vectorizer = None
        self.embedding_model = None
        self.is_fitted = False
        
        # Cache for embeddings
        self.embedding_cache = {}
        
    def fit(self, texts: List[str]) -> 'TextVectorizer':
        """Fit the vectorizer on text data."""
        logger.info(f"Fitting text vectorizer in {self.mode} mode")
        
        if self.mode in ["embeddings", "auto"]:
            try:
                self._fit_embeddings(texts)
                self.mode = "embeddings"
                logger.info("Successfully initialized sentence embeddings")
            except Exception as e:
                if self.mode == "auto" and config.fallback_to_tfidf:
                    logger.warning(f"Embeddings failed, falling back to TF-IDF: {e}")
                    self._fit_tfidf(texts)
                    self.mode = "tfidf"
                else:
                    raise e
        else:
            self._fit_tfidf(texts)
        
        self.is_fitted = True
        return self
    
    def _fit_embeddings(self, texts: List[str]):
        """Fit sentence embedding model."""
        from sentence_transformers import SentenceTransformer
        
        self.embedding_model = SentenceTransformer(config.embedding_model)
        
        # Test with a sample
        sample_texts = texts[:5]
        test_embeddings = self.embedding_model.encode(sample_texts)
        
        logger.info(f"Embedding model loaded: {config.embedding_model}")
        logger.info(f"Embedding dimension: {test_embeddings.shape[1]}")
    
    def _fit_tfidf(self, texts: List[str]):
        """Fit TF-IDF vectorizer."""
        # Adjust min_df based on corpus size to avoid empty vocabulary
        min_df = max(1, min(2, len(texts) // 3)) if len(texts) < 10 else 2
        
        self.vectorizer = TfidfVectorizer(
            max_features=config.max_features,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=min_df,
            max_df=0.8,
            lowercase=True,
            strip_accents='unicode'
        )
        
        self.vectorizer.fit(texts)
        
        logger.info(f"TF-IDF vectorizer fitted with {len(self.vectorizer.vocabulary_)} features")
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to feature vectors."""
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        
        if self.mode == "embeddings":
            return self._transform_embeddings(texts)
        else:
            return self._transform_tfidf(texts)
    
    def _transform_embeddings(self, texts: List[str]) -> np.ndarray:
        """Transform texts using sentence embeddings."""
        # Check cache first
        cache_key = hash(tuple(texts))
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        
        # Cache results
        if len(texts) <= 1000:  # Only cache small batches
            self.embedding_cache[cache_key] = embeddings
        
        return embeddings
    
    def _transform_tfidf(self, texts: List[str]) -> np.ndarray:
        """Transform texts using TF-IDF."""
        return self.vectorizer.transform(texts).toarray()
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        if self.mode == "embeddings":
            dim = self.embedding_model.get_sentence_embedding_dimension()
            return [f"emb_{i}" for i in range(dim)]
        else:
            return self.vectorizer.get_feature_names_out().tolist()
    
    def save(self, filepath: str):
        """Save the fitted vectorizer."""
        save_data = {
            "mode": self.mode,
            "is_fitted": self.is_fitted,
            "config_embedding_model": config.embedding_model
        }
        
        if self.mode == "tfidf":
            save_data["vectorizer"] = self.vectorizer
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Vectorizer saved to {filepath}")
    
    def load(self, filepath: str):
        """Load a fitted vectorizer."""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.mode = save_data["mode"]
        self.is_fitted = save_data["is_fitted"]
        
        if self.mode == "tfidf":
            self.vectorizer = save_data["vectorizer"]
        elif self.mode == "embeddings":
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(save_data["config_embedding_model"])
        
        logger.info(f"Vectorizer loaded from {filepath}")


class FeatureEngineer:
    """Feature engineering for ML models."""
    
    def __init__(self):
        self.text_vectorizer = None
        self.label_encoders = {}
        self.scalers = {}
        self.feature_names = []
    
    def create_text_features(self, df: pd.DataFrame, text_columns: List[str], 
                           mode: str = "auto") -> Tuple[np.ndarray, List[str]]:
        """Create text features from specified columns."""
        logger.info(f"Creating text features from {len(text_columns)} columns")
        
        # Combine text columns
        combined_texts = []
        for idx, row in df.iterrows():
            text_parts = []
            for col in text_columns:
                if col in df.columns and pd.notna(row[col]):
                    text_parts.append(str(row[col]))
            combined_texts.append(' '.join(text_parts))
        
        # Initialize and fit vectorizer
        self.text_vectorizer = TextVectorizer(mode=mode)
        self.text_vectorizer.fit(combined_texts)
        
        # Transform texts
        features = self.text_vectorizer.transform(combined_texts)
        feature_names = self.text_vectorizer.get_feature_names()
        
        logger.info(f"Created {features.shape[1]} text features using {self.text_vectorizer.mode}")
        
        return features, feature_names
    
    def create_categorical_features(self, df: pd.DataFrame, 
                                  categorical_columns: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Create features from categorical columns."""
        logger.info(f"Creating categorical features from {len(categorical_columns)} columns")
        
        all_features = []
        all_feature_names = []
        
        for col in categorical_columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found in DataFrame")
                continue
            
            # Handle missing values
            series = df[col].fillna('Unknown')
            
            # Label encoding
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                encoded = self.label_encoders[col].fit_transform(series.astype(str))
            else:
                encoded = self.label_encoders[col].transform(series.astype(str))
            
            # One-hot encoding for low cardinality
            unique_count = series.nunique()
            if unique_count <= 20:  # One-hot encode
                one_hot = pd.get_dummies(series, prefix=col).values
                all_features.append(one_hot)
                feature_names = [f"{col}_{val}" for val in pd.get_dummies(series, prefix=col).columns]
                all_feature_names.extend(feature_names)
            else:  # Use label encoding
                all_features.append(encoded.reshape(-1, 1))
                all_feature_names.append(f"{col}_encoded")
        
        if all_features:
            features = np.hstack(all_features)
        else:
            features = np.empty((len(df), 0))
        
        logger.info(f"Created {features.shape[1]} categorical features")
        
        return features, all_feature_names
    
    def create_numeric_features(self, df: pd.DataFrame, 
                              numeric_columns: List[str],
                              scale: bool = True) -> Tuple[np.ndarray, List[str]]:
        """Create features from numeric columns."""
        logger.info(f"Creating numeric features from {len(numeric_columns)} columns")
        
        # Select and clean numeric columns
        numeric_data = df[numeric_columns].copy()
        
        # Handle missing values
        for col in numeric_columns:
            if col in numeric_data.columns:
                median_val = numeric_data[col].median()
                numeric_data[col] = numeric_data[col].fillna(median_val)
        
        features = numeric_data.values
        feature_names = list(numeric_columns)
        
        # Scale features if requested
        if scale and features.shape[1] > 0:
            scaler_key = "_".join(numeric_columns)
            if scaler_key not in self.scalers:
                self.scalers[scaler_key] = StandardScaler()
                features = self.scalers[scaler_key].fit_transform(features)
            else:
                features = self.scalers[scaler_key].transform(features)
            
            feature_names = [f"{name}_scaled" for name in feature_names]
        
        logger.info(f"Created {features.shape[1]} numeric features")
        
        return features, feature_names
    
    def create_interaction_features(self, features: np.ndarray, 
                                  feature_names: List[str],
                                  max_interactions: int = 10) -> Tuple[np.ndarray, List[str]]:
        """Create interaction features between existing features."""
        logger.info("Creating interaction features")
        
        if features.shape[1] < 2:
            return features, feature_names
        
        # Select top features for interactions (based on variance)
        feature_vars = np.var(features, axis=0)
        top_indices = np.argsort(feature_vars)[-max_interactions:]
        
        interaction_features = []
        interaction_names = []
        
        # Create pairwise interactions
        for i in range(len(top_indices)):
            for j in range(i+1, len(top_indices)):
                idx1, idx2 = top_indices[i], top_indices[j]
                
                # Multiplicative interaction
                interaction = features[:, idx1] * features[:, idx2]
                interaction_features.append(interaction)
                interaction_names.append(f"{feature_names[idx1]}_x_{feature_names[idx2]}")
        
        if interaction_features:
            interaction_matrix = np.column_stack(interaction_features)
            combined_features = np.hstack([features, interaction_matrix])
            combined_names = feature_names + interaction_names
            
            logger.info(f"Created {len(interaction_features)} interaction features")
            
            return combined_features, combined_names
        
        return features, feature_names
    
    def engineer_features(self, df: pd.DataFrame, 
                         text_columns: List[str] = None,
                         categorical_columns: List[str] = None,
                         numeric_columns: List[str] = None,
                         create_interactions: bool = True) -> Tuple[np.ndarray, List[str]]:
        """Engineer comprehensive features from DataFrame."""
        logger.info("Starting comprehensive feature engineering")
        
        # Auto-detect columns if not specified
        if text_columns is None:
            text_columns = [col for col in df.select_dtypes(include=['object']).columns
                           if df[col].astype(str).str.len().mean() > 10]
        
        if categorical_columns is None:
            categorical_columns = [col for col in df.select_dtypes(include=['object']).columns
                                 if col not in text_columns and df[col].nunique() < 50]
        
        if numeric_columns is None:
            numeric_columns = list(df.select_dtypes(include=[np.number]).columns)
        
        all_features = []
        all_feature_names = []
        
        # Text features
        if text_columns:
            text_features, text_names = self.create_text_features(df, text_columns)
            all_features.append(text_features)
            all_feature_names.extend(text_names)
        
        # Categorical features
        if categorical_columns:
            cat_features, cat_names = self.create_categorical_features(df, categorical_columns)
            all_features.append(cat_features)
            all_feature_names.extend(cat_names)
        
        # Numeric features
        if numeric_columns:
            num_features, num_names = self.create_numeric_features(df, numeric_columns)
            all_features.append(num_features)
            all_feature_names.extend(num_names)
        
        # Combine all features
        if all_features:
            combined_features = np.hstack([f for f in all_features if f.shape[1] > 0])
        else:
            combined_features = np.empty((len(df), 0))
        
        # Create interaction features
        if create_interactions and combined_features.shape[1] > 1:
            combined_features, all_feature_names = self.create_interaction_features(
                combined_features, all_feature_names
            )
        
        self.feature_names = all_feature_names
        
        logger.info(f"Feature engineering complete: {combined_features.shape[1]} total features")
        
        return combined_features, all_feature_names
    
    def save_feature_engineering(self, filepath: str):
        """Save fitted feature engineering components."""
        save_data = {
            "text_vectorizer": self.text_vectorizer,
            "label_encoders": self.label_encoders,
            "scalers": self.scalers,
            "feature_names": self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Feature engineering components saved to {filepath}")
    
    def load_feature_engineering(self, filepath: str):
        """Load fitted feature engineering components."""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.text_vectorizer = save_data["text_vectorizer"]
        self.label_encoders = save_data["label_encoders"]
        self.scalers = save_data["scalers"]
        self.feature_names = save_data["feature_names"]
        
        logger.info(f"Feature engineering components loaded from {filepath}")


class FeatureAnalyzer:
    """Analyze features for ML readiness and quality."""
    
    def __init__(self):
        self.analysis_results = {}
    
    def analyze_features(self, X: np.ndarray, y: np.ndarray = None, 
                        feature_names: List[str] = None) -> Dict[str, Any]:
        """Comprehensive feature analysis."""
        logger.info(f"Analyzing features: shape {X.shape}")
        
        analysis = {
            "shape": X.shape,
            "basic_stats": self._get_basic_feature_stats(X),
            "quality_metrics": self._assess_feature_quality(X),
            "correlation_analysis": self._analyze_feature_correlations(X, feature_names),
        }
        
        if y is not None:
            analysis["target_analysis"] = self._analyze_target_relationship(X, y, feature_names)
        
        return analysis
    
    def _get_basic_feature_stats(self, X: np.ndarray) -> Dict[str, Any]:
        """Get basic statistics for feature matrix."""
        return {
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "sparsity": np.count_nonzero(X == 0) / X.size,
            "mean_feature_value": np.mean(X),
            "std_feature_value": np.std(X),
            "min_feature_value": np.min(X),
            "max_feature_value": np.max(X),
            "has_inf": np.isinf(X).any(),
            "has_nan": np.isnan(X).any()
        }
    
    def _assess_feature_quality(self, X: np.ndarray) -> Dict[str, Any]:
        """Assess feature quality for ML."""
        quality = {
            "constant_features": 0,
            "low_variance_features": 0,
            "high_correlation_features": 0,
            "quality_score": 0.0
        }
        
        # Check for constant features
        feature_stds = np.std(X, axis=0)
        quality["constant_features"] = np.sum(feature_stds == 0)
        
        # Check for low variance features
        quality["low_variance_features"] = np.sum(feature_stds < 0.01)
        
        # Check for highly correlated features
        if X.shape[1] > 1:
            try:
                corr_matrix = np.corrcoef(X.T)
                # Count pairs with correlation > 0.95
                high_corr_pairs = 0
                for i in range(X.shape[1]):
                    for j in range(i+1, X.shape[1]):
                        if abs(corr_matrix[i, j]) > 0.95:
                            high_corr_pairs += 1
                quality["high_correlation_features"] = high_corr_pairs
            except:
                quality["high_correlation_features"] = 0
        
        # Calculate overall quality score
        total_features = X.shape[1]
        if total_features > 0:
            quality_issues = (quality["constant_features"] + 
                            quality["low_variance_features"] + 
                            quality["high_correlation_features"])
            quality["quality_score"] = max(0, 1 - (quality_issues / total_features))
        
        return quality
    
    def _analyze_feature_correlations(self, X: np.ndarray, 
                                    feature_names: List[str] = None) -> Dict[str, Any]:
        """Analyze correlations between features."""
        if X.shape[1] < 2:
            return {"error": "Not enough features for correlation analysis"}
        
        try:
            corr_matrix = np.corrcoef(X.T)
            
            # Handle NaN values in correlation matrix
            corr_matrix = np.nan_to_num(corr_matrix)
            
            # Find strongest correlations
            strong_correlations = []
            for i in range(X.shape[1]):
                for j in range(i+1, X.shape[1]):
                    corr_val = corr_matrix[i, j]
                    if abs(corr_val) > 0.7:
                        feat1 = feature_names[i] if feature_names else f"feature_{i}"
                        feat2 = feature_names[j] if feature_names else f"feature_{j}"
                        strong_correlations.append({
                            "feature1": feat1,
                            "feature2": feat2,
                            "correlation": round(corr_val, 3)
                        })
            
            return {
                "matrix_shape": corr_matrix.shape,
                "max_correlation": np.max(np.abs(corr_matrix[corr_matrix != 1])),
                "avg_correlation": np.mean(np.abs(corr_matrix[corr_matrix != 1])),
                "strong_correlations": strong_correlations[:20]  # Top 20
            }
            
        except Exception as e:
            logger.warning(f"Correlation analysis failed: {e}")
            return {"error": str(e)}
    
    def _analyze_target_relationship(self, X: np.ndarray, y: np.ndarray, 
                                   feature_names: List[str] = None) -> Dict[str, Any]:
        """Analyze relationship between features and target."""
        logger.info("Analyzing feature-target relationships")
        
        try:
            # Calculate feature importance using correlation
            feature_target_corr = []
            
            for i in range(X.shape[1]):
                try:
                    corr = np.corrcoef(X[:, i], y)[0, 1]
                    if not np.isnan(corr):
                        feat_name = feature_names[i] if feature_names else f"feature_{i}"
                        feature_target_corr.append({
                            "feature": feat_name,
                            "correlation": round(corr, 3),
                            "abs_correlation": round(abs(corr), 3)
                        })
                except:
                    continue
            
            # Sort by absolute correlation
            feature_target_corr.sort(key=lambda x: x["abs_correlation"], reverse=True)
            
            return {
                "n_features_analyzed": len(feature_target_corr),
                "top_features": feature_target_corr[:20],
                "avg_abs_correlation": np.mean([f["abs_correlation"] for f in feature_target_corr]),
                "max_correlation": max([f["abs_correlation"] for f in feature_target_corr]) if feature_target_corr else 0
            }
            
        except Exception as e:
            logger.warning(f"Target relationship analysis failed: {e}")
            return {"error": str(e)}


def get_vectorizer(mode: str = "auto") -> TextVectorizer:
    """Get configured text vectorizer."""
    return TextVectorizer(mode=mode)


def create_feature_pipeline(df: pd.DataFrame, 
                           target_column: str = None) -> Dict[str, Any]:
    """Create complete feature engineering pipeline for a DataFrame."""
    logger.info("Creating feature engineering pipeline")
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Auto-detect column types
    text_columns = [col for col in df.select_dtypes(include=['object']).columns
                   if df[col].astype(str).str.len().mean() > 10]
    
    categorical_columns = [col for col in df.select_dtypes(include=['object']).columns
                          if col not in text_columns and df[col].nunique() < 50]
    
    numeric_columns = [col for col in df.select_dtypes(include=[np.number]).columns
                      if col != target_column]
    
    logger.info(f"Detected columns - Text: {len(text_columns)}, "
               f"Categorical: {len(categorical_columns)}, Numeric: {len(numeric_columns)}")
    
    # Engineer features
    X, feature_names = engineer.engineer_features(
        df, text_columns, categorical_columns, numeric_columns
    )
    
    # Prepare target if specified
    y = None
    if target_column and target_column in df.columns:
        y = df[target_column].values
        logger.info(f"Target variable: {target_column} with {len(np.unique(y))} unique values")
    
    # Analyze features
    analyzer = FeatureAnalyzer()
    feature_analysis = analyzer.analyze_features(X, y, feature_names)
    
    # Save pipeline
    pipeline_path = Path("models") / "feature_pipeline.pkl"
    pipeline_path.parent.mkdir(exist_ok=True)
    engineer.save_feature_engineering(str(pipeline_path))
    
    results = {
        "features": X,
        "feature_names": feature_names,
        "target": y,
        "analysis": feature_analysis,
        "engineer": engineer,
        "column_mapping": {
            "text_columns": text_columns,
            "categorical_columns": categorical_columns, 
            "numeric_columns": numeric_columns
        }
    }
    
    return results


def main():
    """Main function to run feature engineering pipeline."""
    logger.info("Starting feature engineering pipeline")
    
    # Load cleaned data
    from .cleaning import main as run_cleaning
    data = run_cleaning()
    
    if not data:
        logger.error("No cleaned data available")
        return
    
    all_results = {}
    
    # Process each dataset
    for name, df in data.items():
        logger.info(f"Processing features for {name}")
        
        try:
            results = create_feature_pipeline(df)
            all_results[name] = {
                "shape": results["features"].shape,
                "n_features": len(results["feature_names"]),
                "analysis": results["analysis"],
                "column_mapping": results["column_mapping"]
            }
            
            # Save features
            features_path = Path("data_processed") / f"{name}_features.npz"
            np.savez(
                features_path,
                features=results["features"],
                feature_names=results["feature_names"],
                target=results["target"] if results["target"] is not None else np.array([])
            )
            
            logger.info(f"Features saved for {name}: {results['features'].shape}")
            
        except Exception as e:
            logger.error(f"Feature engineering failed for {name}: {e}")
            continue
    
    # Save feature engineering summary
    save_metrics(all_results, "feature_engineering.json")
    logger.info("Feature engineering pipeline completed")
    
    return all_results


if __name__ == "__main__":
    main()
