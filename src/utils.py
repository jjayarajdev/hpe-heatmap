"""
Utility functions and configuration management for the HPE pipeline.
"""

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Configuration management using Pydantic."""
    
    # Data paths
    data_raw_path: str = Field(default="data_raw", env="DATA_RAW_PATH")
    data_processed_path: str = Field(default="data_processed", env="DATA_PROCESSED_PATH")
    
    # Reproducibility
    random_seed: int = Field(default=42, env="RANDOM_SEED")
    
    # Model configuration
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    use_embeddings: bool = Field(default=True, env="USE_EMBEDDINGS")
    fallback_to_tfidf: bool = Field(default=True, env="FALLBACK_TO_TFIDF")
    
    # Classification
    classifier_type: str = Field(default="xgboost", env="CLASSIFIER_TYPE")
    train_test_split: float = Field(default=0.8, env="TRAIN_TEST_SPLIT")
    validation_split: float = Field(default=0.2, env="VALIDATION_SPLIT")
    cv_folds: int = Field(default=5, env="CV_FOLDS")
    
    # Taxonomy
    taxonomy_alpha: float = Field(default=0.4, env="TAXONOMY_ALPHA")
    taxonomy_beta: float = Field(default=0.3, env="TAXONOMY_BETA")
    taxonomy_gamma: float = Field(default=0.3, env="TAXONOMY_GAMMA")
    edge_threshold: float = Field(default=0.5, env="EDGE_THRESHOLD")
    suggestions_per_node: int = Field(default=10, env="SUGGESTIONS_PER_NODE")
    
    # Matching weights
    cosine_weight: float = Field(default=0.4, env="COSINE_WEIGHT")
    tfidf_weight: float = Field(default=0.3, env="TFIDF_WEIGHT")
    frequency_weight: float = Field(default=0.2, env="FREQUENCY_WEIGHT")
    recency_weight: float = Field(default=0.1, env="RECENCY_WEIGHT")
    
    # Recommendation weights
    skill_match_weight: float = Field(default=0.4, env="SKILL_MATCH_WEIGHT")
    experience_weight: float = Field(default=0.3, env="EXPERIENCE_WEIGHT")
    utilization_weight: float = Field(default=0.1, env="UTILIZATION_WEIGHT")
    
    # Performance
    n_jobs: int = Field(default=-1, env="N_JOBS")
    batch_size: int = Field(default=1000, env="BATCH_SIZE")
    max_features: int = Field(default=10000, env="MAX_FEATURES")
    
    # Quality gates
    min_join_success_rate: float = Field(default=0.95, env="MIN_JOIN_SUCCESS_RATE")
    min_macro_f1: float = Field(default=0.7, env="MIN_MACRO_F1")
    min_taxonomy_coverage: float = Field(default=0.8, env="MIN_TAXONOMY_COVERAGE")
    min_test_coverage: float = Field(default=0.8, env="MIN_TEST_COVERAGE")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/pipeline.log", env="LOG_FILE")

    class Config:
        env_file = ".env"


def setup_logging(config: Config) -> logging.Logger:
    """Setup logging configuration."""
    # Create logs directory if it doesn't exist
    log_dir = Path(config.log_file).parent
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(config.log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set seeds for ML libraries
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_config() -> Config:
    """Load configuration from environment."""
    load_dotenv()
    return Config()


def ensure_directories(config: Config) -> None:
    """Ensure all required directories exist."""
    directories = [
        config.data_processed_path,
        "models",
        "artifacts",
        "artifacts/plots",
        "artifacts/metrics",
        Path(config.log_file).parent,
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def save_metrics(metrics: Dict[str, Any], filename: str = "metrics.json") -> None:
    """Save metrics to JSON file."""
    import json
    
    metrics_path = Path("artifacts/metrics") / filename
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif hasattr(obj, 'dtype') and 'int' in str(obj.dtype):
            return int(obj)
        elif hasattr(obj, 'dtype') and 'float' in str(obj.dtype):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        elif isinstance(obj, tuple):
            return [convert_numpy(item) for item in obj]
        return obj
    
    metrics_clean = convert_numpy(metrics)
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_clean, f, indent=2)


def load_metrics(filename: str = "metrics.json") -> Dict[str, Any]:
    """Load metrics from JSON file."""
    import json
    
    metrics_path = Path("artifacts/metrics") / filename
    
    if not metrics_path.exists():
        return {}
    
    with open(metrics_path, 'r') as f:
        return json.load(f)


def get_file_info(file_path: Path) -> Dict[str, Any]:
    """Get file information including size and modification time."""
    if not file_path.exists():
        return {"exists": False}
    
    stat = file_path.stat()
    return {
        "exists": True,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "modified": pd.Timestamp.fromtimestamp(stat.st_mtime).isoformat(),
    }


def memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def log_memory_usage(logger: logging.Logger, context: str = "") -> None:
    """Log current memory usage."""
    memory_mb = memory_usage_mb()
    logger.info(f"Memory usage {context}: {memory_mb:.1f} MB")


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        self.start_time = pd.Timestamp.now()
        self.logger.info(f"Starting {self.operation}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = pd.Timestamp.now() - self.start_time
        self.logger.info(f"Completed {self.operation} in {duration.total_seconds():.2f} seconds")


def validate_dataframe(df: pd.DataFrame, name: str, 
                      required_columns: Optional[list] = None) -> Dict[str, Any]:
    """Validate DataFrame and return quality metrics."""
    metrics = {
        "name": name,
        "shape": df.shape,
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        "null_counts": df.isnull().sum().to_dict(),
        "null_percentage": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
    
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        metrics["missing_required_columns"] = list(missing_cols)
        metrics["has_all_required_columns"] = len(missing_cols) == 0
    
    return metrics


def detect_language(text: str) -> str:
    """Simple language detection (English bias)."""
    if not isinstance(text, str) or not text.strip():
        return "unknown"
    
    # Simple heuristic - count English stopwords
    english_stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being'
    }
    
    words = text.lower().split()
    if not words:
        return "unknown"
    
    english_word_count = sum(1 for word in words if word in english_stopwords)
    english_ratio = english_word_count / len(words)
    
    return "english" if english_ratio > 0.1 else "other"


def clean_text_basic(text: str) -> str:
    """Basic text cleaning."""
    if not isinstance(text, str):
        return ""
    
    import re
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\-\.,!?]', ' ', text)
    
    return text.strip()


# Global configuration instance
config = load_config()
logger = setup_logging(config)

# Set random seeds on import
set_random_seeds(config.random_seed)

# Ensure directories exist
ensure_directories(config)
