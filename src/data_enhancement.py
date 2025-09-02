"""
Data Enhancement Module for HPE Talent Intelligence Platform.

Transforms and enriches existing columns to create more meaningful business insights
and better relationship mapping.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Tuple
from pathlib import Path

class DataEnhancer:
    """Enhance and transform existing data columns for better business insights."""
    
    def __init__(self):
        self.skill_hierarchy_mapping = {}
        self.geographic_mapping = {}
        self.domain_categories = {}
    
    def enhance_skillset_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Split and enhance Skill_Set_Name column for better analysis."""
        enhanced_df = df.copy()
        
        # Extract components from skill set names like "A Ps_Cld_Client Virtualization Platform Architecture_L2"
        enhanced_df['Skill_Practice_Area'] = enhanced_df['Skill_Set_Name'].str.extract(r'^([A-Z]\s*[A-Z][a-z]*)')
        enhanced_df['Skill_Domain_Code'] = enhanced_df['Skill_Set_Name'].str.extract(r'_([A-Z][a-z]{2,3})_')
        enhanced_df['Skill_Level'] = enhanced_df['Skill_Set_Name'].str.extract(r'_(L\d)$')
        enhanced_df['Skill_Category'] = enhanced_df['Skill_Set_Name'].str.extract(r'_([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)_L\d')
        
        # Map domain codes to full names
        domain_code_mapping = {
            'Cld': 'Cloud',
            'Adv': 'Advisory', 
            'Ins': 'Infrastructure',
            'Cyb': 'Cybersecurity',
            'Ai': 'Artificial Intelligence',
            'Data': 'Data Analytics',
            'App': 'Application',
            'Pur': 'Purchasing',
            'Arch': 'Architecture',
            'Pm': 'Project Management',
            'Sdcm': 'Software Development'
        }
        
        enhanced_df['Skill_Domain_Full'] = enhanced_df['Skill_Domain_Code'].map(domain_code_mapping)
        
        # Extract skill level number
        enhanced_df['Skill_Level_Number'] = enhanced_df['Skill_Level'].str.extract(r'L(\d)').astype(float)
        
        return enhanced_df
    
    def standardize_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize rating columns for better analysis."""
        enhanced_df = df.copy()
        
        # Extract numeric rating from text
        enhanced_df['Rating_Numeric'] = enhanced_df['Rating'].str.extract(r'^(\d+)').astype(float)
        
        # Extract rating label
        enhanced_df['Rating_Label'] = enhanced_df['Rating'].str.extract(r'\d+\s*-\s*(.+)$')
        
        # Create standardized experience categories
        def categorize_experience(rating_num):
            if rating_num >= 5:
                return 'Expert (5+ years)'
            elif rating_num >= 4:
                return 'Senior (3-5 years)'
            elif rating_num >= 3:
                return 'Mid-level (1-3 years)'
            elif rating_num >= 2:
                return 'Junior (6mo-1yr)'
            else:
                return 'Entry-level (<6mo)'
        
        enhanced_df['Experience_Category'] = enhanced_df['Rating_Numeric'].apply(categorize_experience)
        
        # Create seniority score for ranking
        enhanced_df['Seniority_Score'] = enhanced_df['Rating_Numeric'] * 20  # 0-100 scale
        
        return enhanced_df
    
    def enhance_geographic_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhance geographic and organizational data."""
        enhanced_df = df.copy()
        
        # Split MRU into components
        enhanced_df['Business_Unit'] = enhanced_df['RMR_MRU'].str.extract(r'^([A-Za-z\s]+)')
        enhanced_df['Geographic_Region'] = enhanced_df['RMR_MRU'].str.extract(r'(Gcc\s+\w+)$')
        
        # Create region categories
        def categorize_region(mru):
            if pd.isna(mru):
                return 'Unknown'
            mru_str = str(mru).lower()
            if 'bangalore' in mru_str or 'india' in mru_str:
                return 'India'
            elif 'sofia' in mru_str or 'bulgaria' in mru_str:
                return 'Eastern Europe'
            elif 'tunis' in mru_str or 'tunisia' in mru_str:
                return 'North Africa'
            elif 'dalian' in mru_str or 'china' in mru_str:
                return 'Asia Pacific'
            else:
                return 'Other'
        
        enhanced_df['Region_Category'] = enhanced_df['RMR_MRU'].apply(categorize_region)
        
        # Create timezone categories for global operations
        def get_timezone_category(region):
            timezone_map = {
                'India': 'APAC (UTC+5:30)',
                'Eastern Europe': 'EMEA (UTC+2)', 
                'North Africa': 'EMEA (UTC+1)',
                'Asia Pacific': 'APAC (UTC+8)',
                'Other': 'Unknown'
            }
            return timezone_map.get(region, 'Unknown')
        
        enhanced_df['Timezone_Category'] = enhanced_df['Region_Category'].apply(get_timezone_category)
        
        return enhanced_df
    
    def create_domain_hierarchy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create domain hierarchy for better business categorization."""
        enhanced_df = df.copy()
        
        # Create high-level domain categories
        def categorize_domain(domain):
            if pd.isna(domain):
                return 'Unknown'
            
            domain_str = str(domain).lower()
            
            if any(term in domain_str for term in ['cloud', 'platform', 'infrastructure']):
                return 'Cloud & Infrastructure'
            elif any(term in domain_str for term in ['data', 'ai', 'analytics']):
                return 'Data & AI'
            elif any(term in domain_str for term in ['cyber', 'security']):
                return 'Cybersecurity'
            elif any(term in domain_str for term in ['project', 'management']):
                return 'Project Management'
            elif any(term in domain_str for term in ['business', 'operations']):
                return 'Business Operations'
            else:
                return 'Specialized Services'
        
        enhanced_df['Domain_Category'] = enhanced_df['domain'].apply(categorize_domain)
        
        # Create technology focus areas
        def get_tech_focus(domain):
            domain_str = str(domain).lower()
            
            if 'cloud' in domain_str:
                return 'Cloud Technologies'
            elif 'data' in domain_str or 'ai' in domain_str:
                return 'Data & AI Technologies'
            elif 'cyber' in domain_str:
                return 'Security Technologies'
            elif 'infrastructure' in domain_str:
                return 'Infrastructure Technologies'
            elif 'application' in domain_str:
                return 'Application Technologies'
            else:
                return 'Business Technologies'
        
        enhanced_df['Technology_Focus'] = enhanced_df['domain'].apply(get_tech_focus)
        
        return enhanced_df
    
    def create_skill_taxonomy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced skill taxonomy with categories."""
        enhanced_df = df.copy()
        
        # Categorize skills by type
        def categorize_skill(skill):
            if pd.isna(skill):
                return 'Unknown'
            
            skill_str = str(skill).lower()
            
            # Programming languages
            if any(lang in skill_str for lang in ['python', 'java', 'javascript', 'c++', 'c#', 'go', 'rust']):
                return 'Programming Languages'
            
            # Cloud platforms
            elif any(cloud in skill_str for cloud in ['aws', 'azure', 'gcp', 'cloud']):
                return 'Cloud Platforms'
            
            # DevOps & Tools
            elif any(tool in skill_str for tool in ['docker', 'kubernetes', 'jenkins', 'terraform', 'ansible']):
                return 'DevOps & Automation'
            
            # Databases
            elif any(db in skill_str for db in ['sql', 'mongodb', 'postgresql', 'oracle', 'mysql']):
                return 'Database Technologies'
            
            # Security
            elif any(sec in skill_str for sec in ['security', 'firewall', 'encryption', 'penetration']):
                return 'Security & Compliance'
            
            # Business Skills
            elif any(biz in skill_str for biz in ['management', 'leadership', 'communication', 'analytical']):
                return 'Business & Soft Skills'
            
            # Architecture
            elif any(arch in skill_str for arch in ['architecture', 'design', 'solution']):
                return 'Architecture & Design'
            
            else:
                return 'Technical Specialization'
        
        enhanced_df['Skill_Category'] = enhanced_df['Skill_Certification_Name'].apply(categorize_skill)
        
        # Create skill complexity score
        def get_skill_complexity(skill):
            skill_str = str(skill).lower()
            complexity = 1  # Base complexity
            
            # Add complexity for advanced technologies
            if any(tech in skill_str for tech in ['kubernetes', 'terraform', 'machine learning', 'ai']):
                complexity += 3
            elif any(tech in skill_str for tech in ['docker', 'jenkins', 'python', 'java']):
                complexity += 2
            elif any(tech in skill_str for tech in ['sql', 'linux', 'networking']):
                complexity += 1
            
            return min(5, complexity)  # Cap at 5
        
        enhanced_df['Skill_Complexity'] = enhanced_df['Skill_Certification_Name'].apply(get_skill_complexity)
        
        return enhanced_df
    
    def create_business_value_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create business value and demand metrics."""
        enhanced_df = df.copy()
        
        # Create demand score based on skill frequency
        skill_counts = enhanced_df['Skill_Certification_Name'].value_counts()
        enhanced_df['Skill_Demand_Score'] = enhanced_df['Skill_Certification_Name'].map(skill_counts)
        
        # Normalize to 0-100 scale
        max_demand = enhanced_df['Skill_Demand_Score'].max()
        enhanced_df['Skill_Demand_Normalized'] = (enhanced_df['Skill_Demand_Score'] / max_demand * 100).round(1)
        
        # Create resource value score (combination of rating, demand, and complexity)
        enhanced_df['Resource_Value_Score'] = (
            enhanced_df['Rating_Numeric'] * 0.4 +  # 40% skill level
            enhanced_df['Skill_Demand_Normalized'] * 0.01 * 0.3 +  # 30% market demand
            enhanced_df['Skill_Complexity'] * 0.3  # 30% skill complexity
        ).round(2)
        
        # Create scarcity indicator
        def get_scarcity_level(demand_score):
            if demand_score < 10:
                return 'High Scarcity'
            elif demand_score < 50:
                return 'Medium Scarcity'
            elif demand_score < 200:
                return 'Common'
            else:
                return 'Abundant'
        
        enhanced_df['Skill_Scarcity'] = enhanced_df['Skill_Demand_Score'].apply(get_scarcity_level)
        
        return enhanced_df
    
    def enhance_all_data(self, resource_df: pd.DataFrame) -> pd.DataFrame:
        """Apply all enhancements to the resource data."""
        print("🚀 Enhancing resource data with new columns...")
        
        # Apply all enhancement functions
        enhanced_df = self.enhance_skillset_names(resource_df)
        enhanced_df = self.standardize_ratings(enhanced_df)
        enhanced_df = self.enhance_geographic_data(enhanced_df)
        enhanced_df = self.create_domain_hierarchy(enhanced_df)
        enhanced_df = self.create_skill_taxonomy(enhanced_df)
        enhanced_df = self.create_business_value_metrics(enhanced_df)
        
        print(f"✅ Enhanced data: {len(enhanced_df)} resources with {len(enhanced_df.columns)} columns")
        
        # Show new columns created
        original_cols = set(resource_df.columns)
        new_cols = set(enhanced_df.columns) - original_cols
        print(f"🆕 New columns created ({len(new_cols)}):")
        for col in sorted(new_cols):
            print(f"   • {col}")
        
        return enhanced_df

def enhance_dataset(dataset_name: str = 'resource_DETAILS_28_Export_clean_clean') -> pd.DataFrame:
    """Main function to enhance a dataset."""
    
    # Load the dataset
    file_path = f'data_processed/{dataset_name}.parquet'
    
    try:
        df = pd.read_parquet(file_path)
        print(f"📊 Loaded {dataset_name}: {df.shape}")
        
        # Enhance the data
        enhancer = DataEnhancer()
        enhanced_df = enhancer.enhance_all_data(df)
        
        # Save enhanced version
        output_path = f'data_processed/{dataset_name}_enhanced.parquet'
        enhanced_df.to_parquet(output_path, index=False)
        print(f"💾 Saved enhanced data: {output_path}")
        
        return enhanced_df
        
    except Exception as e:
        print(f"❌ Enhancement failed: {e}")
        return None

def main():
    """Test data enhancement."""
    enhanced_df = enhance_dataset()
    
    if enhanced_df is not None:
        print("\n🎯 ENHANCEMENT SUMMARY:")
        print(f"   Original columns: 10")
        print(f"   Enhanced columns: {len(enhanced_df.columns)}")
        print(f"   New insights: {len(enhanced_df.columns) - 10}")
        
        # Show sample enhanced data
        print("\n📋 SAMPLE ENHANCED DATA:")
        sample_cols = ['resource_name', 'Skill_Category', 'Experience_Category', 
                      'Region_Category', 'Domain_Category', 'Resource_Value_Score']
        
        available_cols = [col for col in sample_cols if col in enhanced_df.columns]
        print(enhanced_df[available_cols].head(5).to_string(index=False))

if __name__ == "__main__":
    main()
