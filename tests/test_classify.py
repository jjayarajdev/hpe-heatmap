"""
Unit tests for classification module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.classify import ServiceClassifier, train_service_classifier, predict_services


class TestServiceClassifier:
    """Test service classifier functionality."""
    
    def setup_method(self):
        """Setup test data."""
        # Create sample opportunity data
        self.sample_data = pd.DataFrame({
            'title': [
                'Cloud infrastructure deployment',
                'VMware virtualization project',
                'Cybersecurity assessment',
                'Python application development',
                'Network security implementation'
            ],
            'description': [
                'Deploy cloud infrastructure using AWS services',
                'Implement VMware vSphere virtualization solution',
                'Conduct comprehensive cybersecurity assessment',
                'Develop Python web application with Django',
                'Implement network security with firewalls'
            ],
            'service_type': [
                'Cloud Services',
                'Infrastructure Services', 
                'Security Services',
                'Development Services',
                'Security Services'
            ]
        })
    
    def test_classifier_initialization(self):
        """Test classifier initialization."""
        classifier = ServiceClassifier("logistic")
        
        assert classifier.classifier_type == "logistic"
        assert classifier.model is None
        assert not classifier.is_multilabel
    
    def test_multilabel_detection(self):
        """Test multi-label problem detection."""
        # Single label data
        single_labels = pd.Series(['Cloud', 'Security', 'Infrastructure'])
        classifier = ServiceClassifier()
        
        is_multilabel_single = classifier._detect_multilabel(single_labels)
        assert not is_multilabel_single
        
        # Multi-label data
        multi_labels = pd.Series(['Cloud, Security', 'Infrastructure; Network', 'Development & Testing'])
        is_multilabel_multi = classifier._detect_multilabel(multi_labels)
        assert is_multilabel_multi
    
    def test_training_data_preparation(self):
        """Test training data preparation."""
        classifier = ServiceClassifier("logistic")
        
        X, y, metrics = classifier._prepare_training_data(
            self.sample_data, 
            ['title', 'description'], 
            'service_type'
        )
        
        assert X is not None
        assert y is not None
        assert X.shape[0] == len(self.sample_data)
        assert len(y) == len(self.sample_data)
        assert "class_distribution" in metrics
    
    def test_model_training(self):
        """Test model training process."""
        classifier = ServiceClassifier("logistic")
        
        trained_classifier, metrics = classifier.train(
            self.sample_data,
            ['title', 'description'],
            'service_type',
            test_size=0.2
        )
        
        assert trained_classifier.model is not None
        assert "accuracy" in metrics
        assert "macro_f1" in metrics
        assert metrics["training_samples"] > 0
    
    def test_prediction_shape(self):
        """Test prediction output shape and format."""
        classifier = ServiceClassifier("logistic")
        
        # Train classifier
        classifier.train(
            self.sample_data,
            ['title', 'description'],
            'service_type'
        )
        
        # Test prediction
        predictions = classifier.predict_services("Cloud deployment project", top_k=3)
        
        assert isinstance(predictions, list)
        assert len(predictions) <= 3
        
        if predictions:
            # Check format
            for service_name, confidence in predictions:
                assert isinstance(service_name, str)
                assert isinstance(confidence, float)
                assert 0 <= confidence <= 1
    
    def test_save_load_model(self):
        """Test model saving and loading."""
        classifier = ServiceClassifier("logistic")
        
        # Train classifier
        classifier.train(
            self.sample_data,
            ['title', 'description'],
            'service_type'
        )
        
        # Save model
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_classifier.pkl"
            classifier.save_model(str(model_path))
            
            assert model_path.exists()
            
            # Load model
            new_classifier = ServiceClassifier()
            new_classifier.load_model(str(model_path))
            
            # Test that loaded model works
            predictions = new_classifier.predict_services("Test text", top_k=2)
            assert isinstance(predictions, list)


def test_train_service_classifier_function():
    """Test the main training function."""
    # Create sample data
    sample_data = pd.DataFrame({
        'opportunity_title': [
            'AWS cloud migration project',
            'VMware infrastructure upgrade', 
            'Security vulnerability assessment',
            'Python web development',
            'Network firewall configuration'
        ],
        'opportunity_description': [
            'Migrate legacy systems to AWS cloud platform',
            'Upgrade VMware vSphere infrastructure',
            'Assess security vulnerabilities in network',
            'Develop web application using Python Django',
            'Configure enterprise firewall systems'
        ],
        'service_category': [
            'Cloud Migration',
            'Infrastructure',
            'Security',
            'Development', 
            'Network Security'
        ]
    })
    
    # Test training
    classifier, metrics = train_service_classifier(
        sample_data,
        text_columns=['opportunity_title', 'opportunity_description'],
        label_column='service_category'
    )
    
    # Verify results
    assert classifier is not None
    assert classifier.model is not None
    assert "best_model" in metrics
    assert "best_f1" in metrics
    assert metrics["best_f1"] >= 0


def test_predict_services_function():
    """Test the main prediction function."""
    # First train a model
    sample_data = pd.DataFrame({
        'title': ['Cloud project', 'Security audit', 'Infrastructure setup'],
        'description': ['AWS deployment', 'Security assessment', 'Server configuration'],
        'service_type': ['Cloud', 'Security', 'Infrastructure']
    })
    
    # Train model
    classifier, _ = train_service_classifier(
        sample_data,
        text_columns=['title', 'description'],
        label_column='service_type'
    )
    
    # Save model temporarily
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "test_model.pkl"
        classifier.save_model(str(model_path))
        
        # Test prediction function
        predictions = predict_services(
            "Need help with AWS cloud deployment",
            top_k=2,
            model_path=str(model_path)
        )
        
        assert isinstance(predictions, list)
        assert len(predictions) <= 2
        
        if predictions:
            for service_name, confidence in predictions:
                assert isinstance(service_name, str)
                assert isinstance(confidence, float)


def test_label_mapping_consistency():
    """Test that label mappings are consistent."""
    classifier = ServiceClassifier("logistic")
    
    # Test data with consistent labels
    test_data = pd.DataFrame({
        'text': ['cloud project', 'security audit', 'cloud deployment'],
        'labels': ['Cloud', 'Security', 'Cloud']
    })
    
    X, y, metrics = classifier._prepare_training_data(
        test_data, ['text'], 'labels'
    )
    
    # Check that same labels get same encoded values
    cloud_indices = [i for i, label in enumerate(test_data['labels']) if label == 'Cloud']
    cloud_encoded_values = [y[i] for i in cloud_indices]
    
    # All 'Cloud' labels should have same encoded value
    assert len(set(cloud_encoded_values)) == 1


def test_edge_cases():
    """Test edge cases and error handling."""
    classifier = ServiceClassifier("logistic")
    
    # Empty DataFrame
    empty_df = pd.DataFrame()
    
    with pytest.raises(Exception):
        classifier._prepare_training_data(empty_df, ['text'], 'label')
    
    # Missing label column
    df_no_labels = pd.DataFrame({
        'text': ['sample text'],
        'other_col': ['other value']
    })
    
    X, y, metrics = classifier._prepare_training_data(df_no_labels, ['text'], 'missing_label')
    assert X is None
    assert y is None
    
    # Missing text columns
    df_no_text = pd.DataFrame({
        'label': ['sample label'],
        'other_col': ['other value']
    })
    
    X, y, metrics = classifier._prepare_training_data(df_no_text, ['missing_text'], 'label')
    assert X is None
    assert y is None


if __name__ == "__main__":
    pytest.main([__file__])
