.PHONY: setup preprocess train taxonomy app test clean lint format

# Python and virtual environment
PYTHON := python3.11
VENV := venv
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python

# Setup virtual environment and install dependencies
setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✅ Environment setup complete. Activate with: source $(VENV)/bin/activate"

# Data preprocessing pipeline
preprocess:
	@echo "🔄 Running data preprocessing..."
	$(PYTHON_VENV) -m src.io_loader
	$(PYTHON_VENV) -m src.cleaning
	@echo "✅ Data preprocessing complete"

# Train classification models
train:
	@echo "🤖 Training classification models..."
	$(PYTHON_VENV) -m src.classify
	@echo "✅ Model training complete"

# Build taxonomy artifacts
taxonomy:
	@echo "🌐 Building taxonomy..."
	$(PYTHON_VENV) -m src.taxonomy
	@echo "✅ Taxonomy artifacts created"

# Run complete pipeline
pipeline: preprocess train taxonomy
	@echo "🚀 Complete pipeline executed successfully"

# Launch Streamlit app
app:
	@echo "🌟 Launching Streamlit app..."
	$(PYTHON_VENV) -m streamlit run app/streamlit_app.py --server.port 8501

# Run tests
test:
	@echo "🧪 Running tests..."
	$(PYTHON_VENV) -m pytest tests/ -v --cov=src --cov-report=html
	@echo "✅ Tests complete"

# Code quality
lint:
	@echo "🔍 Running linting..."
	$(PYTHON_VENV) -m flake8 src/ app/ tests/
	$(PYTHON_VENV) -m mypy src/ app/

format:
	@echo "✨ Formatting code..."
	$(PYTHON_VENV) -m black src/ app/ tests/
	$(PYTHON_VENV) -m isort src/ app/ tests/

# Clean up generated files
clean:
	rm -rf $(VENV)
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf data_processed/*.parquet
	rm -rf models/*.pkl
	rm -rf artifacts/*.json artifacts/*.parquet
	@echo "🧹 Cleanup complete"

# Development workflow
dev-setup: setup
	$(PIP) install jupyter notebook ipykernel
	$(PYTHON_VENV) -m ipykernel install --user --name=hpe-pipeline
	@echo "📚 Jupyter kernel installed as 'hpe-pipeline'"

# Quick development cycle
dev: format lint test
	@echo "🔄 Development cycle complete"

# Help
help:
	@echo "Available commands:"
	@echo "  setup      - Create virtual environment and install dependencies"
	@echo "  preprocess - Load and clean data, create parquet files"
	@echo "  train      - Train classification models"
	@echo "  taxonomy   - Build bidirectional taxonomy"
	@echo "  pipeline   - Run complete data pipeline"
	@echo "  app        - Launch Streamlit application"
	@echo "  test       - Run unit tests with coverage"
	@echo "  lint       - Run code quality checks"
	@echo "  format     - Format code with black and isort"
	@echo "  clean      - Remove generated files and virtual environment"
	@echo "  dev-setup  - Setup development environment with Jupyter"
	@echo "  dev        - Run development cycle (format, lint, test)"
