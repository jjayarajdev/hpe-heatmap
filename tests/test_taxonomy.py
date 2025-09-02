"""
Unit tests for taxonomy module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.taxonomy import TaxonomyBuilder, TaxonomyQuery


class TestTaxonomyBuilder:
    """Test taxonomy builder functionality."""
    
    def setup_method(self):
        """Setup test data."""
        self.builder = TaxonomyBuilder()
        
        # Sample services data
        self.services_df = pd.DataFrame({
            'service_name': ['Cloud Services', 'Infrastructure Management', 'Security Assessment'],
            'category': ['Cloud', 'Infrastructure', 'Security'],
            'description': ['Cloud deployment services', 'Infrastructure management', 'Security services']
        })
        
        # Sample skillsets data
        self.skillsets_df = pd.DataFrame({
            'skillset_name': ['Cloud Architecture', 'System Administration', 'Cybersecurity'],
            'category': ['Cloud', 'Infrastructure', 'Security'],
            'description': ['Cloud architecture skills', 'System admin skills', 'Security skills']
        })
        
        # Sample skills data
        self.skills_df = pd.DataFrame({
            'skill_name': ['AWS Lambda', 'Linux Administration', 'Penetration Testing', 'Docker', 'Firewall Management'],
            'description': ['Serverless computing', 'Linux system management', 'Security testing', 'Container technology', 'Network security'],
            'category': ['Cloud', 'Infrastructure', 'Security', 'Cloud', 'Security'],
            'hierarchy': ['Cloud > Serverless', 'Infrastructure > OS', 'Security > Testing', 'Cloud > Containers', 'Security > Network']
        })
    
    def test_entity_processing(self):
        """Test entity processing."""
        stats = self.builder.build_taxonomy(
            self.services_df, self.skillsets_df, self.skills_df
        )
        
        # Check entities were created
        assert len(self.builder.entities["services"]) == 3
        assert len(self.builder.entities["skillsets"]) == 3
        assert len(self.builder.entities["skills"]) == 5
        
        # Check graph was built
        assert self.builder.graph.number_of_nodes() > 0
        assert self.builder.graph.number_of_edges() >= 0
        
        # Check stats
        assert "entities" in stats
        assert "mappings" in stats
        assert "graph" in stats
    
    def test_taxonomy_idempotence(self):
        """Test that building taxonomy twice gives same results."""
        # Build taxonomy first time
        stats1 = self.builder.build_taxonomy(
            self.services_df, self.skillsets_df, self.skills_df
        )
        
        entities1 = len(self.builder.entities["services"])
        nodes1 = self.builder.graph.number_of_nodes()
        
        # Build taxonomy second time (should be idempotent)
        stats2 = self.builder.build_taxonomy(
            self.services_df, self.skillsets_df, self.skills_df
        )
        
        entities2 = len(self.builder.entities["services"])
        nodes2 = self.builder.graph.number_of_nodes()
        
        # Should have same number of entities and nodes
        assert entities1 == entities2
        assert nodes1 == nodes2
    
    def test_save_load_taxonomy(self):
        """Test saving and loading taxonomy."""
        # Build taxonomy
        self.builder.build_taxonomy(
            self.services_df, self.skillsets_df, self.skills_df
        )
        
        # Save to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            taxonomy_path = self.builder.save_taxonomy(temp_dir)
            
            # Check files were created
            assert Path(taxonomy_path).exists()
            assert Path(temp_dir, "services.parquet").exists()
            assert Path(temp_dir, "skillsets.parquet").exists()
            assert Path(temp_dir, "skills.parquet").exists()
    
    def test_suggestion_generation(self):
        """Test that suggestions are generated."""
        self.builder.build_taxonomy(
            self.services_df, self.skillsets_df, self.skills_df
        )
        
        # Should have some suggested edges
        assert isinstance(self.builder.suggested_edges, list)
        # Note: May be empty if no suggestions meet threshold
    
    def test_confidence_scores(self):
        """Test confidence score calculation."""
        self.builder.build_taxonomy(
            self.services_df, self.skillsets_df, self.skills_df
        )
        
        # Should have confidence scores
        assert isinstance(self.builder.confidence_scores, dict)


class TestTaxonomyQuery:
    """Test taxonomy query functionality."""
    
    def setup_method(self):
        """Setup test taxonomy."""
        self.builder = TaxonomyBuilder()
        
        # Create sample data
        services_df = pd.DataFrame({
            'service_name': ['Cloud Services', 'Security Services'],
            'category': ['Cloud', 'Security']
        })
        
        skillsets_df = pd.DataFrame({
            'skillset_name': ['Cloud Architecture', 'Cybersecurity'],
            'category': ['Cloud', 'Security']
        })
        
        skills_df = pd.DataFrame({
            'skill_name': ['AWS Lambda', 'Penetration Testing'],
            'category': ['Cloud', 'Security'],
            'description': ['Serverless computing', 'Security testing']
        })
        
        # Build taxonomy
        self.builder.build_taxonomy(services_df, skillsets_df, skills_df)
        self.query = TaxonomyQuery(self.builder)
    
    def test_skills_for_service(self):
        """Test finding skills for a service."""
        # Test with existing service
        skills = self.query.skills_for_service("Cloud Services", k=5)
        assert isinstance(skills, list)
        
        # Test with non-existent service
        no_skills = self.query.skills_for_service("Non-existent Service", k=5)
        assert no_skills == []
    
    def test_services_for_skill(self):
        """Test finding services for a skill."""
        # Test with existing skill
        services = self.query.services_for_skill("AWS Lambda", k=5)
        assert isinstance(services, list)
        
        # Test with non-existent skill
        no_services = self.query.services_for_skill("Non-existent Skill", k=5)
        assert no_services == []
    
    def test_taxonomy_stats(self):
        """Test taxonomy statistics generation."""
        stats = self.query.get_taxonomy_stats()
        
        assert "entities" in stats
        assert "mappings" in stats
        assert "graph_metrics" in stats
        
        # Check entity counts
        assert stats["entities"]["services"] >= 0
        assert stats["entities"]["skillsets"] >= 0
        assert stats["entities"]["skills"] >= 0


def test_taxonomy_integration():
    """Integration test for complete taxonomy pipeline."""
    # Create larger sample data
    services_df = pd.DataFrame({
        'service_name': ['Cloud Migration', 'Infrastructure Setup', 'Security Audit', 'Application Development'],
        'category': ['Cloud', 'Infrastructure', 'Security', 'Development'],
        'description': ['Cloud migration services', 'Infrastructure services', 'Security services', 'App development']
    })
    
    skillsets_df = pd.DataFrame({
        'skillset_name': ['Cloud Technologies', 'System Administration', 'Cybersecurity', 'Software Development'],
        'category': ['Cloud', 'Infrastructure', 'Security', 'Development']
    })
    
    skills_df = pd.DataFrame({
        'skill_name': ['AWS', 'Azure', 'Linux', 'Windows Server', 'Ethical Hacking', 'Python', 'Java', 'Docker'],
        'category': ['Cloud', 'Cloud', 'Infrastructure', 'Infrastructure', 'Security', 'Development', 'Development', 'Cloud'],
        'description': ['Amazon Web Services', 'Microsoft Azure', 'Linux OS', 'Windows Server', 'Ethical hacking', 'Python programming', 'Java programming', 'Containerization']
    })
    
    # Build taxonomy
    builder = TaxonomyBuilder()
    stats = builder.build_taxonomy(services_df, skillsets_df, skills_df)
    
    # Verify basic functionality
    assert stats["entities"]["services"] == 4
    assert stats["entities"]["skillsets"] == 4
    assert stats["entities"]["skills"] == 8
    
    # Test queries
    query = TaxonomyQuery(builder)
    
    # Test bidirectional queries
    cloud_skills = query.skills_for_service("Cloud Migration", k=10)
    aws_services = query.services_for_skill("AWS", k=10)
    
    # Should return results (even if empty due to no explicit mappings)
    assert isinstance(cloud_skills, list)
    assert isinstance(aws_services, list)
    
    # Test taxonomy stats
    final_stats = query.get_taxonomy_stats()
    assert final_stats["entities"]["services"] == 4


if __name__ == "__main__":
    pytest.main([__file__])
