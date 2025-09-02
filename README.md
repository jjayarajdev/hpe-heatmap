# HPE Resource Assignment System - Production Pipeline

A comprehensive, production-grade ML pipeline for intelligent resource assignment, service classification, and bidirectional taxonomy management.

## 🎯 Features

- **Comprehensive EDA**: Automated data profiling, quality assessment, and feature discovery
- **Service Classification**: Multi-class/multi-label classification with baseline and advanced models
- **Bidirectional Taxonomy**: Intelligent skills ↔ skillsets ↔ services mapping with suggestions
- **Smart Skill Matching**: Semantic matching with ranking and rationale
- **Real-time Recommendations**: Resource recommendations with transparent scoring
- **Interactive Dashboard**: 4-tab Streamlit app for exploration and testing

## 🚀 Quick Start

### 1. Setup Environment

```bash
make setup
source venv/bin/activate
```

### 2. Run Complete Pipeline

```bash
make pipeline
```

This will:
- Load and preprocess all Excel data
- Train classification models
- Build bidirectional taxonomy
- Generate artifacts and metrics

### 3. Launch Interactive App

```bash
make app
```

Access the dashboard at: `http://localhost:8501`

## 📊 Data Sources

The pipeline processes these Excel files:

| File | Purpose | Key Entities |
|------|---------|--------------|
| `RAWDATA.xlsx` | Opportunities/requests | Opportunity ID, title, description, status |
| `DETAILS (28).xlsx` | Resource details | Resource profiles, skills, ratings |
| `Services_to_skillsets Mapping.xlsx` | Service mappings | Service ↔ skillset relationships |
| `Skillsets_to_Skills_mapping.xlsx` | Skill taxonomy | Skillset ↔ skill hierarchies |
| `data - 2025-08-22T000557.141.xlsx` | Business metrics | Geographic and business area data |

## 🏗️ Architecture

```
project/
├── data_raw/                    # Input Excel files (symlinked)
├── data_processed/             # Cleaned parquet files
├── src/                        # Core pipeline modules
│   ├── io_loader.py           # Data loading and schema harmonization
│   ├── cleaning.py            # Data cleaning and normalization
│   ├── eda.py                 # Exploratory data analysis
│   ├── features.py            # Feature engineering (TF-IDF, embeddings)
│   ├── taxonomy.py            # Bidirectional taxonomy builder
│   ├── classify.py            # Service classification models
│   ├── match.py               # Smart skill matching
│   ├── recommend.py           # Resource recommendation engine
│   └── utils.py               # Utilities and configuration
├── models/                     # Trained models
├── artifacts/                  # Generated artifacts and metrics
├── app/                        # Streamlit dashboard
├── tests/                      # Unit tests
└── notebooks/                  # EDA and analysis notebooks
```

## 🔧 Core Components

### Data Processing (`src/io_loader.py`, `src/cleaning.py`)
- Automated schema detection and harmonization
- Null handling, deduplication, normalization
- Foreign key relationship validation
- Standardized parquet output

### Classification (`src/classify.py`)
- Baseline: TF-IDF + LogisticRegression
- Advanced: Sentence embeddings + XGBoost
- Multi-class and multi-label support
- Comprehensive evaluation metrics

### Taxonomy (`src/taxonomy.py`)
- Bipartite graph construction (services ↔ skills)
- Weighted edges with multiple signals
- Intelligent edge suggestions
- Query utilities for bidirectional lookup

### Matching (`src/match.py`)
- Multi-stage candidate generation
- Weighted scoring with multiple features
- Learning-to-rank optimization
- Transparent rationale generation

### Recommendations (`src/recommend.py`)
- Resource scoring with multiple factors
- Real-time inference
- Configurable weighting parameters
- Detailed explanations

## 📱 Interactive Dashboard

### Tab 1: EDA
- Dataset overview and quality metrics
- Missingness heatmaps
- Class distribution analysis
- Interactive UMAP visualizations

### Tab 2: Taxonomy Explorer
- Searchable bipartite graph
- Service ↔ skill relationships
- Suggested edge management
- Confidence scoring

### Tab 3: Classifier Tester
- Real-time service prediction
- Probability distributions
- Feature importance highlights
- Model comparison

### Tab 4: Recommendations
- Freeform query interface
- Ranked results with rationale
- Parameter tuning controls
- Resource recommendations

## 🧪 Testing and Quality

```bash
# Run all tests
make test

# Code quality checks
make lint

# Format code
make format

# Development cycle
make dev
```

### Quality Gates
- ≥95% data join success rate
- Baseline macro-F1 meets threshold
- Taxonomy suggestions validation
- Unit test coverage >80%

## 📈 Performance Metrics

The pipeline generates comprehensive metrics in `artifacts/metrics.json`:

- Data quality scores
- Model performance (accuracy, F1, AUC)
- Taxonomy coverage and confidence
- Recommendation relevance scores

## 🔧 Configuration

Create `.env` file for custom settings:

```bash
cp .env.example .env
```

Key parameters:
- Model hyperparameters
- Similarity thresholds
- Ranking weights
- Feature engineering options

## 📚 API Usage

### Service Classification
```python
from src.classify import predict_services

predictions = predict_services("Cloud infrastructure deployment", top_k=5)
# Returns: [("Cloud Services", 0.89), ("Infrastructure", 0.76), ...]
```

### Skill Matching
```python
from src.match import match_skills_and_services

results = match_skills_and_services("Python development project")
# Returns: {'skills': [...], 'skillsets': [...], 'services': [...]}
```

### Resource Recommendations
```python
from src.recommend import recommend_resources

resources = recommend_resources("Need Azure expert for migration", n=5)
# Returns: [{'name': 'John Doe', 'score': 0.92, 'rationale': '...'}, ...]
```

## 🔄 Development Workflow

1. **Data Changes**: Update Excel files → `make preprocess`
2. **Model Updates**: Modify algorithms → `make train`
3. **Taxonomy Changes**: Update mappings → `make taxonomy`
4. **Full Rebuild**: `make pipeline`
5. **Testing**: `make dev`

## 📊 Monitoring

- Model drift detection in EDA notebooks
- Data quality monitoring
- Performance metric tracking
- Taxonomy coverage analysis

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Run `make dev` to ensure quality
4. Submit pull request

## 📄 License

Copyright (c) 2025 HPE. All rights reserved.

---

**Built with ❤️ for intelligent resource management**
