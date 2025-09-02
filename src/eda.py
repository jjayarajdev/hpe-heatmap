"""
Exploratory Data Analysis module with profiling and visualization helpers.

This module provides comprehensive EDA capabilities including data profiling,
quality assessment, outlier detection, and feature discovery.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import warnings

from .utils import config, logger, Timer, save_metrics
from .io_loader import load_processed_data

warnings.filterwarnings('ignore')


class DataProfiler:
    """Comprehensive data profiling and quality assessment."""
    
    def __init__(self):
        self.profiles = {}
        self.plots_dir = Path("artifacts/plots")
        self.plots_dir.mkdir(parents=True, exist_ok=True)
    
    def profile_dataset(self, df: pd.DataFrame, name: str) -> Dict[str, Any]:
        """Create comprehensive profile for a dataset."""
        logger.info(f"Profiling dataset: {name}")
        
        profile = {
            "name": name,
            "overview": self._get_overview_stats(df),
            "columns": self._profile_columns(df),
            "quality": self._assess_data_quality(df),
            "relationships": self._analyze_relationships(df),
            "text_analysis": self._analyze_text_content(df),
            "outliers": self._detect_outliers(df)
        }
        
        self.profiles[name] = profile
        return profile
    
    def _get_overview_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic overview statistics."""
        return {
            "shape": df.shape,
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
            "dtypes": df.dtypes.value_counts().to_dict(),
            "null_count": df.isnull().sum().sum(),
            "null_percentage": round(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100, 2),
            "duplicate_rows": df.duplicated().sum(),
            "duplicate_percentage": round(df.duplicated().sum() / len(df) * 100, 2)
        }
    
    def _profile_columns(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Profile each column individually."""
        column_profiles = {}
        
        for col in df.columns:
            try:
                series = df[col]
                profile = {
                    "dtype": str(series.dtype),
                    "null_count": series.isnull().sum(),
                    "null_percentage": round(series.isnull().sum() / len(series) * 100, 2),
                    "unique_count": series.nunique(),
                    "unique_percentage": round(series.nunique() / len(series) * 100, 2)
                }
                
                # Type-specific analysis
                if series.dtype in ['int64', 'float64']:
                    profile.update(self._profile_numeric_column(series))
                elif series.dtype == 'object':
                    profile.update(self._profile_text_column(series))
                elif series.dtype == 'datetime64[ns]':
                    profile.update(self._profile_datetime_column(series))
                elif series.dtype.name == 'category':
                    profile.update(self._profile_categorical_column(series))
                
                column_profiles[col] = profile
                
            except Exception as e:
                logger.warning(f"Error profiling column {col}: {e}")
                column_profiles[col] = {"error": str(e)}
        
        return column_profiles
    
    def _profile_numeric_column(self, series: pd.Series) -> Dict[str, Any]:
        """Profile numeric column."""
        series_clean = series.dropna()
        
        if len(series_clean) == 0:
            return {"error": "No non-null values"}
        
        return {
            "min": series_clean.min(),
            "max": series_clean.max(),
            "mean": series_clean.mean(),
            "median": series_clean.median(),
            "std": series_clean.std(),
            "q25": series_clean.quantile(0.25),
            "q75": series_clean.quantile(0.75),
            "skewness": series_clean.skew(),
            "kurtosis": series_clean.kurtosis(),
            "zeros_count": (series_clean == 0).sum(),
            "negative_count": (series_clean < 0).sum()
        }
    
    def _profile_text_column(self, series: pd.Series) -> Dict[str, Any]:
        """Profile text column."""
        series_clean = series.dropna().astype(str)
        
        if len(series_clean) == 0:
            return {"error": "No non-null values"}
        
        # Length statistics
        lengths = series_clean.str.len()
        
        # Word statistics
        word_counts = series_clean.str.split().str.len()
        
        # Character analysis
        all_text = ' '.join(series_clean)
        char_counter = Counter(all_text.lower())
        
        return {
            "avg_length": lengths.mean(),
            "min_length": lengths.min(),
            "max_length": lengths.max(),
            "avg_words": word_counts.mean(),
            "empty_strings": (series_clean == '').sum(),
            "unique_chars": len(char_counter),
            "most_common_chars": char_counter.most_common(5),
            "contains_digits": series_clean.str.contains(r'\d').sum(),
            "contains_special": series_clean.str.contains(r'[^a-zA-Z0-9\s]').sum()
        }
    
    def _profile_datetime_column(self, series: pd.Series) -> Dict[str, Any]:
        """Profile datetime column."""
        series_clean = series.dropna()
        
        if len(series_clean) == 0:
            return {"error": "No non-null values"}
        
        return {
            "min_date": series_clean.min(),
            "max_date": series_clean.max(),
            "date_range_days": (series_clean.max() - series_clean.min()).days,
            "unique_dates": series_clean.nunique(),
            "most_common_date": series_clean.mode().iloc[0] if not series_clean.mode().empty else None
        }
    
    def _profile_categorical_column(self, series: pd.Series) -> Dict[str, Any]:
        """Profile categorical column."""
        series_clean = series.dropna()
        
        if len(series_clean) == 0:
            return {"error": "No non-null values"}
        
        value_counts = series_clean.value_counts()
        
        return {
            "categories": len(series_clean.cat.categories),
            "most_frequent": value_counts.iloc[0] if len(value_counts) > 0 else None,
            "least_frequent": value_counts.iloc[-1] if len(value_counts) > 0 else None,
            "top_5_values": value_counts.head().to_dict(),
            "bottom_5_values": value_counts.tail().to_dict()
        }
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess overall data quality."""
        quality = {
            "completeness_score": 0.0,
            "consistency_score": 0.0,
            "validity_score": 0.0,
            "quality_issues": []
        }
        
        # Completeness: percentage of non-null values
        total_cells = df.shape[0] * df.shape[1]
        non_null_cells = total_cells - df.isnull().sum().sum()
        quality["completeness_score"] = non_null_cells / total_cells if total_cells > 0 else 0
        
        # Consistency: check for inconsistent formatting
        consistency_issues = 0
        text_columns = df.select_dtypes(include=['object']).columns
        
        for col in text_columns:
            series = df[col].dropna().astype(str)
            if len(series) == 0:
                continue
            
            # Check for inconsistent case
            has_mixed_case = (series.str.islower().any() and 
                            series.str.isupper().any() and 
                            series.str.istitle().any())
            if has_mixed_case:
                consistency_issues += 1
                quality["quality_issues"].append(f"Mixed case in column: {col}")
            
            # Check for leading/trailing whitespace
            has_whitespace = (series != series.str.strip()).any()
            if has_whitespace:
                consistency_issues += 1
                quality["quality_issues"].append(f"Whitespace issues in column: {col}")
        
        quality["consistency_score"] = max(0, 1 - (consistency_issues / max(len(text_columns), 1)))
        
        # Validity: check for reasonable values
        validity_issues = 0
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            
            # Check for impossible values (e.g., negative ratings if rating column)
            if 'rating' in col.lower() or 'score' in col.lower():
                if (series < 0).any() or (series > 10).any():
                    validity_issues += 1
                    quality["quality_issues"].append(f"Invalid ratings in column: {col}")
        
        quality["validity_score"] = max(0, 1 - (validity_issues / max(len(numeric_columns), 1)))
        
        # Overall quality score (weighted average)
        quality["overall_score"] = (0.4 * quality["completeness_score"] + 
                                  0.3 * quality["consistency_score"] + 
                                  0.3 * quality["validity_score"])
        
        return quality
    
    def _analyze_relationships(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze relationships between columns."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        relationships = {
            "correlations": {},
            "strong_correlations": [],
            "potential_duplicates": []
        }
        
        if len(numeric_columns) > 1:
            # Calculate correlation matrix
            corr_matrix = df[numeric_columns].corr()
            relationships["correlations"] = corr_matrix.to_dict()
            
            # Find strong correlations
            for i, col1 in enumerate(numeric_columns):
                for j, col2 in enumerate(numeric_columns):
                    if i < j:  # Avoid duplicates
                        corr_value = corr_matrix.loc[col1, col2]
                        if abs(corr_value) > 0.8:
                            relationships["strong_correlations"].append({
                                "column1": col1,
                                "column2": col2,
                                "correlation": corr_value
                            })
        
        # Check for potential duplicate columns
        for i, col1 in enumerate(df.columns):
            for j, col2 in enumerate(df.columns):
                if i < j:
                    # Check if columns are identical
                    if df[col1].equals(df[col2]):
                        relationships["potential_duplicates"].append({
                            "column1": col1,
                            "column2": col2,
                            "type": "identical"
                        })
        
        return relationships
    
    def _analyze_text_content(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze text content in the dataset."""
        text_columns = df.select_dtypes(include=['object']).columns
        text_analysis = {
            "text_columns": len(text_columns),
            "column_analysis": {},
            "overall_stats": {}
        }
        
        all_text = []
        
        for col in text_columns:
            series = df[col].dropna().astype(str)
            
            if len(series) == 0 or series.str.len().mean() < 3:
                continue  # Skip very short or empty columns
            
            col_analysis = {
                "sample_size": len(series),
                "avg_length": series.str.len().mean(),
                "max_length": series.str.len().max(),
                "language_hints": self._detect_language_hints(series.head(20)),
                "top_ngrams": self._get_top_ngrams(series, n=3, top_k=10)
            }
            
            text_analysis["column_analysis"][col] = col_analysis
            all_text.extend(series.head(100).tolist())  # Sample for overall analysis
        
        # Overall text statistics
        if all_text:
            combined_text = ' '.join(str(text) for text in all_text)
            text_analysis["overall_stats"] = {
                "total_chars": len(combined_text),
                "total_words": len(combined_text.split()),
                "unique_words": len(set(combined_text.lower().split())),
                "avg_word_length": np.mean([len(word) for word in combined_text.split()]),
                "language_distribution": self._detect_language_hints(pd.Series(all_text))
            }
        
        return text_analysis
    
    def _detect_language_hints(self, series: pd.Series) -> Dict[str, float]:
        """Detect language hints in text series."""
        if series.empty:
            return {"unknown": 1.0}
        
        # English indicators
        english_indicators = {
            'the', 'and', 'or', 'is', 'are', 'was', 'were', 'have', 'has', 'had',
            'will', 'would', 'could', 'should', 'can', 'may', 'must', 'shall'
        }
        
        language_scores = {"english": 0, "other": 0}
        
        for text in series.head(20):
            if not isinstance(text, str):
                continue
            
            words = set(re.findall(r'\b\w+\b', text.lower()))
            english_word_count = len(words & english_indicators)
            
            if len(words) > 0:
                english_ratio = english_word_count / len(words)
                if english_ratio > 0.1:
                    language_scores["english"] += 1
                else:
                    language_scores["other"] += 1
        
        total = sum(language_scores.values())
        if total == 0:
            return {"unknown": 1.0}
        
        return {lang: score/total for lang, score in language_scores.items()}
    
    def _get_top_ngrams(self, series: pd.Series, n: int = 3, top_k: int = 10) -> List[Tuple[str, int]]:
        """Get top n-grams from text series."""
        if series.empty:
            return []
        
        try:
            # Combine all text
            text_sample = series.head(100).astype(str)
            combined_text = ' '.join(text_sample)
            
            # Extract n-grams
            words = re.findall(r'\b\w+\b', combined_text.lower())
            
            if len(words) < n:
                return []
            
            ngrams = []
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i+n])
                ngrams.append(ngram)
            
            # Count and return top k
            ngram_counts = Counter(ngrams)
            return ngram_counts.most_common(top_k)
            
        except Exception as e:
            logger.warning(f"Error extracting n-grams: {e}")
            return []
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers in numeric columns."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        outlier_analysis = {}
        
        for col in numeric_columns:
            series = df[col].dropna()
            
            if len(series) < 4:  # Need at least 4 values for quartiles
                continue
            
            # IQR method
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:  # No variation
                continue
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_mask = (series < lower_bound) | (series > upper_bound)
            outlier_count = outliers_mask.sum()
            
            if outlier_count > 0:
                outlier_analysis[col] = {
                    "outlier_count": outlier_count,
                    "outlier_percentage": round(outlier_count / len(series) * 100, 2),
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "outlier_values": series[outliers_mask].tolist()[:10]  # Sample outliers
                }
        
        return outlier_analysis
    
    def _analyze_relationships(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze relationships between columns."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) < 2:
            return {"error": "Not enough numeric columns for relationship analysis"}
        
        # Correlation analysis
        corr_matrix = df[numeric_columns].corr()
        
        # Find strong correlations
        strong_corr = []
        for i, col1 in enumerate(numeric_columns):
            for j, col2 in enumerate(numeric_columns):
                if i < j:
                    corr_val = corr_matrix.loc[col1, col2]
                    if abs(corr_val) > 0.7:
                        strong_corr.append({
                            "column1": col1,
                            "column2": col2,
                            "correlation": round(corr_val, 3),
                            "strength": "strong" if abs(corr_val) > 0.9 else "moderate"
                        })
        
        return {
            "correlation_matrix": corr_matrix.round(3).to_dict(),
            "strong_correlations": strong_corr,
            "max_correlation": corr_matrix.abs().max().max(),
            "avg_correlation": corr_matrix.abs().mean().mean()
        }
    
    def create_eda_visualizations(self, df: pd.DataFrame, name: str) -> Dict[str, str]:
        """Create EDA visualizations and save plots."""
        plot_files = {}
        
        try:
            # 1. Missing values heatmap
            if df.isnull().sum().sum() > 0:
                fig_missing = self._create_missing_values_heatmap(df)
                missing_path = self.plots_dir / f"{name}_missing_values.html"
                fig_missing.write_html(str(missing_path))
                plot_files["missing_values"] = str(missing_path)
            
            # 2. Data type distribution
            fig_dtypes = self._create_dtype_distribution(df)
            dtypes_path = self.plots_dir / f"{name}_dtypes.html"
            fig_dtypes.write_html(str(dtypes_path))
            plot_files["dtypes"] = str(dtypes_path)
            
            # 3. Correlation heatmap (if numeric columns exist)
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 1:
                fig_corr = self._create_correlation_heatmap(df[numeric_columns])
                corr_path = self.plots_dir / f"{name}_correlations.html"
                fig_corr.write_html(str(corr_path))
                plot_files["correlations"] = str(corr_path)
            
            # 4. Text length distribution (if text columns exist)
            text_columns = df.select_dtypes(include=['object']).columns
            if len(text_columns) > 0:
                fig_text = self._create_text_length_distribution(df, text_columns)
                text_path = self.plots_dir / f"{name}_text_analysis.html"
                fig_text.write_html(str(text_path))
                plot_files["text_analysis"] = str(text_path)
            
        except Exception as e:
            logger.warning(f"Error creating visualizations for {name}: {e}")
        
        return plot_files
    
    def _create_missing_values_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create missing values heatmap."""
        missing_matrix = df.isnull()
        
        fig = go.Figure(data=go.Heatmap(
            z=missing_matrix.values,
            x=list(df.columns),
            y=list(range(len(df))),
            colorscale='Reds',
            showscale=True,
            hovertemplate='Column: %{x}<br>Row: %{y}<br>Missing: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Missing Values Heatmap",
            xaxis_title="Columns",
            yaxis_title="Rows",
            height=600
        )
        
        return fig
    
    def _create_dtype_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Create data type distribution chart."""
        dtype_counts = df.dtypes.value_counts()
        
        fig = go.Figure(data=[
            go.Bar(x=dtype_counts.index.astype(str), y=dtype_counts.values)
        ])
        
        fig.update_layout(
            title="Data Type Distribution",
            xaxis_title="Data Types",
            yaxis_title="Number of Columns",
            height=400
        )
        
        return fig
    
    def _create_correlation_heatmap(self, df_numeric: pd.DataFrame) -> go.Figure:
        """Create correlation heatmap for numeric columns."""
        corr_matrix = df_numeric.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=list(corr_matrix.columns),
            y=list(corr_matrix.columns),
            colorscale='RdBu',
            zmid=0,
            showscale=True,
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Correlation Matrix",
            height=600,
            width=600
        )
        
        return fig
    
    def _create_text_length_distribution(self, df: pd.DataFrame, 
                                       text_columns: List[str]) -> go.Figure:
        """Create text length distribution charts."""
        fig = make_subplots(
            rows=min(len(text_columns), 4), 
            cols=1,
            subplot_titles=[f"Length Distribution: {col}" for col in text_columns[:4]]
        )
        
        for i, col in enumerate(text_columns[:4]):
            text_lengths = df[col].astype(str).str.len()
            
            fig.add_trace(
                go.Histogram(x=text_lengths, name=col, showlegend=False),
                row=i+1, col=1
            )
        
        fig.update_layout(
            title="Text Length Distributions",
            height=200 * min(len(text_columns), 4)
        )
        
        return fig


class FeatureDiscovery:
    """Feature discovery and text analysis for ML preparation."""
    
    def __init__(self):
        self.vectorizers = {}
        self.embeddings = {}
    
    def analyze_text_features(self, df: pd.DataFrame, text_column: str) -> Dict[str, Any]:
        """Analyze text features for ML readiness."""
        if text_column not in df.columns:
            return {"error": f"Column {text_column} not found"}
        
        series = df[text_column].dropna().astype(str)
        
        if len(series) == 0:
            return {"error": "No text data available"}
        
        analysis = {
            "tfidf_analysis": self._analyze_tfidf_features(series),
            "embedding_analysis": self._analyze_embedding_features(series),
            "separability": self._analyze_feature_separability(series),
            "topic_discovery": self._discover_topics(series)
        }
        
        return analysis
    
    def _analyze_tfidf_features(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze TF-IDF feature characteristics."""
        try:
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(series)
            feature_names = vectorizer.get_feature_names_out()
            
            # Calculate feature statistics
            feature_means = np.array(tfidf_matrix.mean(axis=0)).flatten()
            feature_stds = np.array(tfidf_matrix.std(axis=0)).flatten()
            
            # Get top features
            top_features_idx = np.argsort(feature_means)[-20:]
            top_features = [(feature_names[i], feature_means[i]) for i in top_features_idx]
            
            self.vectorizers["tfidf"] = vectorizer
            
            return {
                "vocabulary_size": len(feature_names),
                "matrix_shape": tfidf_matrix.shape,
                "sparsity": 1 - (tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])),
                "top_features": top_features,
                "feature_variance": feature_stds.mean(),
                "max_feature_value": feature_means.max()
            }
            
        except Exception as e:
            logger.warning(f"TF-IDF analysis failed: {e}")
            return {"error": str(e)}
    
    def _analyze_embedding_features(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze sentence embedding characteristics."""
        try:
            # Try to use sentence transformers if available
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(config.embedding_model)
                
                # Sample for performance
                sample_size = min(100, len(series))
                sample_texts = series.sample(n=sample_size, random_state=42).tolist()
                
                embeddings = model.encode(sample_texts)
                
                # Calculate embedding statistics
                embedding_means = np.mean(embeddings, axis=0)
                embedding_stds = np.std(embeddings, axis=0)
                
                # Calculate pairwise similarities for sample
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity(embeddings)
                
                # Remove diagonal (self-similarities)
                np.fill_diagonal(similarities, np.nan)
                
                return {
                    "embedding_dim": embeddings.shape[1],
                    "sample_size": sample_size,
                    "avg_embedding_norm": np.linalg.norm(embeddings, axis=1).mean(),
                    "embedding_variance": embedding_stds.mean(),
                    "avg_pairwise_similarity": np.nanmean(similarities),
                    "similarity_std": np.nanstd(similarities),
                    "model_used": config.embedding_model
                }
                
            except ImportError:
                logger.warning("Sentence transformers not available, skipping embedding analysis")
                return {"error": "sentence_transformers not available"}
                
        except Exception as e:
            logger.warning(f"Embedding analysis failed: {e}")
            return {"error": str(e)}
    
    def _analyze_feature_separability(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze how separable the features are (useful for classification)."""
        try:
            if len(series) < 10:
                return {"error": "Not enough samples for separability analysis"}
            
            # Use TF-IDF for separability analysis
            vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(series)
            
            # Apply dimensionality reduction
            svd = TruncatedSVD(n_components=min(50, tfidf_matrix.shape[1]-1), random_state=42)
            reduced_features = svd.fit_transform(tfidf_matrix)
            
            # Calculate variance explained
            variance_explained = svd.explained_variance_ratio_
            cumulative_variance = np.cumsum(variance_explained)
            
            # Calculate feature density (how spread out the features are)
            feature_distances = []
            for i in range(min(100, len(reduced_features))):
                for j in range(i+1, min(100, len(reduced_features))):
                    distance = np.linalg.norm(reduced_features[i] - reduced_features[j])
                    feature_distances.append(distance)
            
            return {
                "dimensions_for_90pct_variance": np.argmax(cumulative_variance >= 0.9) + 1,
                "top_10_variance_explained": variance_explained[:10].tolist(),
                "avg_pairwise_distance": np.mean(feature_distances) if feature_distances else 0,
                "feature_spread": np.std(feature_distances) if feature_distances else 0,
                "separability_score": np.mean(feature_distances) / (np.std(feature_distances) + 1e-8) if feature_distances else 0
            }
            
        except Exception as e:
            logger.warning(f"Separability analysis failed: {e}")
            return {"error": str(e)}
    
    def _discover_topics(self, series: pd.Series) -> Dict[str, Any]:
        """Discover topics using simple clustering."""
        try:
            if len(series) < 10:
                return {"error": "Not enough samples for topic discovery"}
            
            # Use TF-IDF + K-means for simple topic discovery
            vectorizer = TfidfVectorizer(
                max_features=200,
                stop_words='english',
                min_df=2,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(series)
            feature_names = vectorizer.get_feature_names_out()
            
            # Simple clustering
            from sklearn.cluster import KMeans
            n_clusters = min(5, len(series) // 10)
            
            if n_clusters < 2:
                return {"error": "Not enough samples for clustering"}
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # Get representative terms for each cluster
            cluster_topics = {}
            for cluster_id in range(n_clusters):
                cluster_center = kmeans.cluster_centers_[cluster_id]
                top_feature_indices = np.argsort(cluster_center)[-10:][::-1]
                top_features = [feature_names[i] for i in top_feature_indices]
                
                cluster_size = (cluster_labels == cluster_id).sum()
                cluster_topics[f"topic_{cluster_id}"] = {
                    "size": cluster_size,
                    "percentage": round(cluster_size / len(series) * 100, 1),
                    "top_terms": top_features
                }
            
            return {
                "n_topics": n_clusters,
                "topics": cluster_topics,
                "silhouette_score": self._calculate_silhouette_score(tfidf_matrix, cluster_labels)
            }
            
        except Exception as e:
            logger.warning(f"Topic discovery failed: {e}")
            return {"error": str(e)}
    
    def _calculate_silhouette_score(self, X, labels) -> float:
        """Calculate silhouette score for clustering quality."""
        try:
            from sklearn.metrics import silhouette_score
            return silhouette_score(X, labels)
        except:
            return 0.0


def main():
    """Main function to run EDA pipeline."""
    logger.info("Starting EDA pipeline")
    
    # Load processed data
    data = load_processed_data()
    
    if not data:
        logger.error("No processed data found. Run io_loader and cleaning first.")
        return
    
    # Initialize profiler
    profiler = DataProfiler()
    feature_discovery = FeatureDiscovery()
    
    all_profiles = {}
    all_visualizations = {}
    
    # Profile each dataset
    for name, df in data.items():
        logger.info(f"Running EDA for {name}")
        
        with Timer(logger, f"EDA for {name}"):
            # Create profile
            profile = profiler.profile_dataset(df, name)
            all_profiles[name] = profile
            
            # Create visualizations
            plot_files = profiler.create_eda_visualizations(df, name)
            all_visualizations[name] = plot_files
            
            # Feature discovery for text columns
            text_columns = df.select_dtypes(include=['object']).columns
            for text_col in text_columns:
                if df[text_col].astype(str).str.len().mean() > 10:  # Only analyze substantial text
                    feature_analysis = feature_discovery.analyze_text_features(df, text_col)
                    profile[f"feature_analysis_{text_col}"] = feature_analysis
    
    # Save comprehensive EDA results
    eda_results = {
        "profiles": all_profiles,
        "visualizations": all_visualizations,
        "summary": {
            "datasets_analyzed": len(data),
            "total_rows": sum(df.shape[0] for df in data.values()),
            "total_columns": sum(df.shape[1] for df in data.values()),
            "quality_scores": {name: profile["quality"]["overall_score"] 
                             for name, profile in all_profiles.items()}
        }
    }
    
    save_metrics(eda_results, "eda_results.json")
    logger.info("EDA results saved to artifacts/metrics/eda_results.json")
    
    logger.info("EDA pipeline completed successfully")
    
    return eda_results


if __name__ == "__main__":
    main()
