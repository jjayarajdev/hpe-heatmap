"""
Data cleaning and normalization module.

This module handles data cleaning, deduplication, normalization,
and text preprocessing for the HPE pipeline.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path
from collections import Counter

from .utils import config, logger, Timer, validate_dataframe, save_metrics
from .io_loader import load_processed_data


class DataCleaner:
    """Comprehensive data cleaning and normalization."""
    
    def __init__(self):
        self.cleaning_stats = {}
        self.text_stats = {}
        
    def clean_all_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Clean all datasets and return cleaned versions."""
        cleaned_data = {}
        
        with Timer(logger, "cleaning all datasets"):
            for name, df in data.items():
                logger.info(f"Cleaning {name} dataset...")
                cleaned_df = self.clean_dataframe(df, name)
                cleaned_data[name] = cleaned_df
                
                # Log cleaning stats
                original_shape = df.shape
                cleaned_shape = cleaned_df.shape
                logger.info(f"  {name}: {original_shape} -> {cleaned_shape}")
        
        return cleaned_data
    
    def clean_dataframe(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Clean a single DataFrame."""
        df_clean = df.copy()
        
        stats = {
            "dataset": dataset_name,
            "original_shape": df.shape,
            "operations": []
        }
        
        # 1. Handle missing values
        df_clean, missing_stats = self._handle_missing_values(df_clean)
        stats["operations"].append(("handle_missing", missing_stats))
        
        # 2. Remove duplicates
        df_clean, dup_stats = self._remove_duplicates(df_clean)
        stats["operations"].append(("remove_duplicates", dup_stats))
        
        # 3. Normalize text columns
        df_clean, text_stats = self._normalize_text_columns(df_clean)
        stats["operations"].append(("normalize_text", text_stats))
        
        # 4. Fix data types
        df_clean, dtype_stats = self._fix_data_types(df_clean)
        stats["operations"].append(("fix_dtypes", dtype_stats))
        
        # 5. Handle outliers (for numeric columns)
        df_clean, outlier_stats = self._handle_outliers(df_clean)
        stats["operations"].append(("handle_outliers", outlier_stats))
        
        # 6. Standardize categorical values
        df_clean, cat_stats = self._standardize_categorical(df_clean)
        stats["operations"].append(("standardize_categorical", cat_stats))
        
        stats["final_shape"] = df_clean.shape
        stats["rows_removed"] = df.shape[0] - df_clean.shape[0]
        stats["columns_removed"] = df.shape[1] - df_clean.shape[1]
        
        self.cleaning_stats[dataset_name] = stats
        
        return df_clean
    
    def _handle_missing_values(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Handle missing values with appropriate strategies."""
        df_clean = df.copy()
        stats = {"columns_processed": 0, "values_imputed": 0, "columns_dropped": []}
        
        # Calculate missing percentages
        missing_pct = df.isnull().sum() / len(df)
        
        for col in df.columns:
            if missing_pct[col] > 0:
                stats["columns_processed"] += 1
                
                # Drop columns with >90% missing values
                if missing_pct[col] > 0.9:
                    df_clean = df_clean.drop(columns=[col])
                    stats["columns_dropped"].append(col)
                    continue
                
                # Handle missing values based on column type and content
                if df[col].dtype in ['object']:
                    # Text columns - fill with empty string or 'Unknown'
                    if any(keyword in col.lower() for keyword in ['id', 'name', 'title']):
                        # Don't impute ID or name fields
                        continue
                    else:
                        fill_value = 'Unknown' if 'category' in col.lower() else ''
                        df_clean[col] = df_clean[col].fillna(fill_value)
                        stats["values_imputed"] += df[col].isnull().sum()
                
                elif df[col].dtype in ['int64', 'float64']:
                    # Numeric columns - use median for skewed data, mean for normal
                    if col.lower() in ['rating', 'score', 'proficiency']:
                        # For rating-like columns, use median
                        fill_value = df[col].median()
                    else:
                        # For other numeric columns, use mean
                        fill_value = df[col].mean()
                    
                    df_clean[col] = df_clean[col].fillna(fill_value)
                    stats["values_imputed"] += df[col].isnull().sum()
                
                elif df[col].dtype == 'datetime64[ns]':
                    # DateTime columns - forward fill or use current date
                    df_clean[col] = df_clean[col].fillna(method='ffill')
                    stats["values_imputed"] += df[col].isnull().sum()
        
        return df_clean, stats
    
    def _remove_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Remove duplicate rows with intelligent handling."""
        original_length = len(df)
        
        # First, remove exact duplicates
        df_clean = df.drop_duplicates()
        exact_duplicates_removed = original_length - len(df_clean)
        
        stats = {
            "exact_duplicates_removed": exact_duplicates_removed,
            "fuzzy_duplicates_removed": 0,
            "final_length": len(df_clean)
        }
        
        # For datasets with text content, check for fuzzy duplicates
        text_columns = df_clean.select_dtypes(include=['object']).columns
        if len(text_columns) > 0:
            df_clean, fuzzy_removed = self._remove_fuzzy_duplicates(df_clean, text_columns)
            stats["fuzzy_duplicates_removed"] = fuzzy_removed
            stats["final_length"] = len(df_clean)
        
        return df_clean, stats
    
    def _remove_fuzzy_duplicates(self, df: pd.DataFrame, 
                                text_columns: List[str]) -> Tuple[pd.DataFrame, int]:
        """Remove fuzzy duplicates based on text similarity."""
        if len(df) < 2:
            return df, 0
        
        # For large datasets, sample for fuzzy duplicate detection
        sample_size = min(1000, len(df))
        if len(df) > sample_size:
            df_sample = df.sample(n=sample_size, random_state=42)
        else:
            df_sample = df
        
        # Simple fuzzy duplicate detection based on text similarity
        duplicates_to_remove = set()
        
        for col in text_columns[:2]:  # Limit to first 2 text columns for performance
            if col not in df_sample.columns:
                continue
                
            text_series = df_sample[col].astype(str).str.lower().str.strip()
            
            for i, text1 in enumerate(text_series):
                if i in duplicates_to_remove or pd.isna(text1) or text1 == '':
                    continue
                
                for j, text2 in enumerate(text_series[i+1:], i+1):
                    if j in duplicates_to_remove or pd.isna(text2) or text2 == '':
                        continue
                    
                    # Simple similarity check
                    if len(text1) > 10 and len(text2) > 10:
                        similarity = self._simple_text_similarity(text1, text2)
                        if similarity > 0.9:  # High similarity threshold
                            duplicates_to_remove.add(j)
        
        # Remove identified duplicates from original DataFrame
        if duplicates_to_remove:
            indices_to_remove = df_sample.iloc[list(duplicates_to_remove)].index
            df_clean = df.drop(indices_to_remove)
            return df_clean, len(duplicates_to_remove)
        
        return df, 0
    
    def _simple_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity based on character overlap."""
        if not text1 or not text2:
            return 0.0
        
        # Convert to sets of characters
        set1 = set(text1.lower())
        set2 = set(text2.lower())
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _normalize_text_columns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Normalize text columns with comprehensive cleaning."""
        df_clean = df.copy()
        text_columns = df.select_dtypes(include=['object']).columns
        
        stats = {
            "columns_processed": 0,
            "text_cleaning_operations": {},
            "encoding_fixes": 0,
            "language_detection": {}
        }
        
        for col in text_columns:
            if col not in df_clean.columns:
                continue
                
            original_values = df_clean[col].astype(str)
            
            # Skip if all values are very short (likely categorical)
            avg_length = original_values.str.len().mean()
            if avg_length < 5:
                continue
            
            stats["columns_processed"] += 1
            col_stats = {}
            
            # Text cleaning pipeline
            cleaned_values = original_values.copy()
            
            # 1. Fix encoding issues
            cleaned_values, encoding_fixes = self._fix_text_encoding(cleaned_values)
            col_stats["encoding_fixes"] = encoding_fixes
            stats["encoding_fixes"] += encoding_fixes
            
            # 2. Normalize whitespace
            cleaned_values = cleaned_values.str.replace(r'\s+', ' ', regex=True)
            cleaned_values = cleaned_values.str.strip()
            
            # 3. Remove special characters (but preserve basic punctuation)
            cleaned_values = cleaned_values.str.replace(r'[^\w\s\-\.,!?()]', ' ', regex=True)
            
            # 4. Fix common text issues
            cleaned_values = cleaned_values.str.replace(r'\b(\w)\1{3,}\b', r'\1', regex=True)  # Remove repeated chars
            
            # 5. Detect language (simple heuristic)
            language_dist = self._detect_languages(cleaned_values.dropna().head(100))
            col_stats["language_distribution"] = language_dist
            stats["language_detection"][col] = language_dist
            
            # 6. Calculate text quality metrics
            col_stats.update(self._calculate_text_metrics(original_values, cleaned_values))
            
            df_clean[col] = cleaned_values
            stats["text_cleaning_operations"][col] = col_stats
        
        return df_clean, stats
    
    def _fix_text_encoding(self, series: pd.Series) -> Tuple[pd.Series, int]:
        """Fix common text encoding issues."""
        fixed_series = series.copy()
        fixes = 0
        
        # Common encoding fixes
        encoding_fixes = {
            'â€™': "'",
            'â€œ': '"',
            'â€\x9d': '"',
            'â€"': '–',
            'â€"': '—',
            'Ã¡': 'á',
            'Ã©': 'é',
            'Ã­': 'í',
            'Ã³': 'ó',
            'Ãº': 'ú',
        }
        
        for bad, good in encoding_fixes.items():
            mask = fixed_series.str.contains(bad, na=False)
            if mask.any():
                fixed_series = fixed_series.str.replace(bad, good, regex=False)
                fixes += mask.sum()
        
        return fixed_series, fixes
    
    def _detect_languages(self, series: pd.Series) -> Dict[str, float]:
        """Simple language detection based on common words."""
        if series.empty:
            return {"unknown": 1.0}
        
        # English stopwords
        english_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
        
        language_counts = {"english": 0, "other": 0, "unknown": 0}
        
        for text in series.dropna().head(50):  # Sample for performance
            if not isinstance(text, str) or len(text.strip()) == 0:
                language_counts["unknown"] += 1
                continue
            
            words = re.findall(r'\b\w+\b', text.lower())
            if not words:
                language_counts["unknown"] += 1
                continue
            
            english_word_count = sum(1 for word in words if word in english_words)
            english_ratio = english_word_count / len(words)
            
            if english_ratio > 0.1:
                language_counts["english"] += 1
            else:
                language_counts["other"] += 1
        
        total = sum(language_counts.values())
        if total == 0:
            return {"unknown": 1.0}
        
        return {lang: count/total for lang, count in language_counts.items()}
    
    def _calculate_text_metrics(self, original: pd.Series, cleaned: pd.Series) -> Dict[str, Any]:
        """Calculate text quality metrics."""
        metrics = {}
        
        # Length statistics
        original_lengths = original.str.len().dropna()
        cleaned_lengths = cleaned.str.len().dropna()
        
        if not original_lengths.empty:
            metrics["original_avg_length"] = original_lengths.mean()
            metrics["original_max_length"] = original_lengths.max()
            metrics["cleaned_avg_length"] = cleaned_lengths.mean()
            metrics["length_reduction"] = (original_lengths.mean() - cleaned_lengths.mean()) / original_lengths.mean()
        
        # Word statistics
        original_words = original.str.split().str.len().dropna()
        if not original_words.empty:
            metrics["avg_words"] = original_words.mean()
            metrics["max_words"] = original_words.max()
        
        # Character diversity
        all_text = ' '.join(cleaned.dropna().astype(str))
        if all_text:
            char_counts = Counter(all_text.lower())
            metrics["unique_chars"] = len(char_counts)
            metrics["most_common_chars"] = char_counts.most_common(10)
        
        return metrics
    
    def _fix_data_types(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Fix and optimize data types."""
        df_clean = df.copy()
        stats = {"conversions": {}, "errors": {}}
        
        for col in df.columns:
            original_dtype = str(df[col].dtype)
            
            try:
                # Try to convert object columns to more specific types
                if df[col].dtype == 'object':
                    # Try datetime conversion for date-like columns
                    if any(date_word in col.lower() for date_word in ['date', 'time', 'created', 'modified']):
                        try:
                            df_clean[col] = pd.to_datetime(df[col], errors='coerce')
                            stats["conversions"][col] = f"object -> datetime64[ns]"
                            continue
                        except:
                            pass
                    
                    # Try numeric conversion
                    numeric_series = pd.to_numeric(df[col], errors='coerce')
                    if numeric_series.notna().sum() > len(df) * 0.8:  # >80% convertible
                        df_clean[col] = numeric_series
                        stats["conversions"][col] = f"object -> {numeric_series.dtype}"
                        continue
                    
                    # Try categorical conversion for low-cardinality columns
                    unique_ratio = df[col].nunique() / len(df)
                    if unique_ratio < 0.1 and df[col].nunique() < 50:
                        df_clean[col] = df[col].astype('category')
                        stats["conversions"][col] = "object -> category"
                        continue
                
                # Optimize numeric types
                elif df[col].dtype in ['int64', 'float64']:
                    # Try to downcast to smaller types
                    if df[col].dtype == 'int64':
                        downcast_series = pd.to_numeric(df[col], downcast='integer')
                        if str(downcast_series.dtype) != original_dtype:
                            df_clean[col] = downcast_series
                            stats["conversions"][col] = f"{original_dtype} -> {downcast_series.dtype}"
                    
                    elif df[col].dtype == 'float64':
                        downcast_series = pd.to_numeric(df[col], downcast='float')
                        if str(downcast_series.dtype) != original_dtype:
                            df_clean[col] = downcast_series
                            stats["conversions"][col] = f"{original_dtype} -> {downcast_series.dtype}"
            
            except Exception as e:
                stats["errors"][col] = str(e)
        
        return df_clean, stats
    
    def _handle_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Handle outliers in numeric columns."""
        df_clean = df.copy()
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        stats = {"columns_processed": 0, "outliers_capped": {}, "outliers_removed": 0}
        
        for col in numeric_columns:
            if col not in df_clean.columns:
                continue
            
            series = df_clean[col].dropna()
            if len(series) < 10:  # Skip if too few values
                continue
            
            stats["columns_processed"] += 1
            
            # Use IQR method for outlier detection
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:  # Skip if no variation
                continue
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers_mask = (series < lower_bound) | (series > upper_bound)
            outlier_count = outliers_mask.sum()
            
            if outlier_count > 0:
                outlier_ratio = outlier_count / len(series)
                
                if outlier_ratio < 0.05:  # Less than 5% outliers - cap them
                    df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
                    stats["outliers_capped"][col] = outlier_count
                elif outlier_ratio > 0.2:  # More than 20% outliers - likely not outliers
                    continue
                # For 5-20% outliers, leave as is (might be legitimate extreme values)
        
        return df_clean, stats
    
    def _standardize_categorical(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Standardize categorical values."""
        df_clean = df.copy()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        
        stats = {"columns_processed": 0, "standardizations": {}}
        
        for col in categorical_columns:
            if col not in df_clean.columns:
                continue
            
            # Skip if too many unique values (likely not categorical)
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.5:
                continue
            
            stats["columns_processed"] += 1
            col_stats = {"original_unique": df[col].nunique()}
            
            # Standardize case
            df_clean[col] = df_clean[col].astype(str).str.strip().str.title()
            
            # Group similar values
            value_counts = df_clean[col].value_counts()
            standardization_map = {}
            
            for value in value_counts.index:
                if pd.isna(value) or value == 'nan':
                    continue
                
                # Find similar values to group together
                similar_values = []
                for other_value in value_counts.index:
                    if value != other_value and self._are_similar_categorical_values(value, other_value):
                        similar_values.append(other_value)
                
                if similar_values:
                    # Use the most frequent value as the standard
                    all_values = [value] + similar_values
                    most_frequent = max(all_values, key=lambda x: value_counts.get(x, 0))
                    
                    for val in all_values:
                        if val != most_frequent:
                            standardization_map[val] = most_frequent
            
            # Apply standardizations
            if standardization_map:
                df_clean[col] = df_clean[col].replace(standardization_map)
                col_stats["values_standardized"] = len(standardization_map)
                col_stats["final_unique"] = df_clean[col].nunique()
            
            stats["standardizations"][col] = col_stats
        
        return df_clean, stats
    
    def _are_similar_categorical_values(self, val1: str, val2: str) -> bool:
        """Check if two categorical values are similar enough to be standardized."""
        if not isinstance(val1, str) or not isinstance(val2, str):
            return False
        
        val1_clean = val1.lower().strip()
        val2_clean = val2.lower().strip()
        
        # Check for exact match after cleaning
        if val1_clean == val2_clean:
            return True
        
        # Check for substring relationships
        if val1_clean in val2_clean or val2_clean in val1_clean:
            return True
        
        # Check for common abbreviations
        abbreviation_map = {
            'mgmt': 'management',
            'dev': 'development',
            'admin': 'administration',
            'tech': 'technology',
            'eng': 'engineering',
            'ops': 'operations',
        }
        
        for abbrev, full in abbreviation_map.items():
            if (abbrev in val1_clean and full in val2_clean) or (abbrev in val2_clean and full in val1_clean):
                return True
        
        return False
    
    def get_cleaning_summary(self) -> Dict[str, Any]:
        """Get summary of all cleaning operations."""
        return {
            "datasets_cleaned": len(self.cleaning_stats),
            "cleaning_stats": self.cleaning_stats,
            "text_stats": self.text_stats
        }


def main():
    """Main function to run data cleaning pipeline."""
    logger.info("Starting data cleaning pipeline")
    
    # Load processed data
    data = load_processed_data()
    
    if not data:
        logger.error("No processed data found. Run io_loader first.")
        return
    
    # Initialize cleaner
    cleaner = DataCleaner()
    
    # Clean all datasets
    cleaned_data = cleaner.clean_all_data(data)
    
    # Save cleaned data
    output_path = Path(config.data_processed_path)
    
    for name, df in cleaned_data.items():
        # Save with _clean suffix
        clean_filename = f"{name}_clean.parquet"
        parquet_path = output_path / clean_filename
        df.to_parquet(parquet_path, index=False)
        logger.info(f"Saved cleaned data: {parquet_path}")
    
    # Save cleaning summary
    cleaning_summary = cleaner.get_cleaning_summary()
    save_metrics(cleaning_summary, "data_cleaning.json")
    
    logger.info("Data cleaning pipeline completed successfully")
    
    return cleaned_data


if __name__ == "__main__":
    main()
