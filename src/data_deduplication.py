"""
Data Deduplication Module for HPE Talent Intelligence Platform.

Fixes the critical issue where same resources have multiple IDs,
creating a proper one-person-one-record structure.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from pathlib import Path

class ResourceDeduplicator:
    """Deduplicate resources and create proper person-centric records."""
    
    def __init__(self):
        self.dedup_stats = {}
    
    def deduplicate_resources(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deduplicate resources by person name and aggregate their skills.
        
        Problem: Same person (e.g., Danail Denev) has multiple resource_ids
        Solution: One record per person with all their skills aggregated
        """
        print(f"🔍 Starting deduplication of {len(df)} records...")
        
        # Group by person (resource_name + manager for uniqueness)
        grouped = df.groupby(['resource_name', 'manager'])
        
        deduplicated_records = []
        
        for (name, manager), group in grouped:
            # Aggregate all skills for this person
            skills_list = group['Skill_Certification_Name'].dropna().unique().tolist()
            skill_sets_list = group['Skill_Set_Name'].dropna().unique().tolist()
            
            # Calculate aggregated ratings
            numeric_ratings = group['Proficieny_Rating'].dropna()
            avg_rating = numeric_ratings.mean() if len(numeric_ratings) > 0 else 0
            max_rating = numeric_ratings.max() if len(numeric_ratings) > 0 else 0
            
            # Get the most common/highest values for other fields
            domain = group['domain'].mode().iloc[0] if len(group['domain'].mode()) > 0 else group['domain'].iloc[0]
            city = group['RMR_City'].mode().iloc[0] if len(group['RMR_City'].mode()) > 0 else group['RMR_City'].iloc[0]
            mru = group['RMR_MRU'].mode().iloc[0] if len(group['RMR_MRU'].mode()) > 0 else group['RMR_MRU'].iloc[0]
            
            # Create consolidated record
            consolidated_record = {
                'resource_id': f"PERSON_{len(deduplicated_records):06d}",  # New unique ID
                'resource_name': name,
                'manager': manager,
                'primary_domain': domain,
                'city': city,
                'mru': mru,
                'skill_count': len(skills_list),
                'skillset_count': len(skill_sets_list),
                'avg_rating': round(avg_rating, 2),
                'max_rating': max_rating,
                'all_skills': '; '.join(skills_list),
                'all_skillsets': '; '.join(skill_sets_list),
                'original_record_count': len(group)
            }
            
            # Add skill categories if enhanced data is available
            if 'Skill_Category' in group.columns:
                skill_categories = group['Skill_Category'].dropna().unique().tolist()
                consolidated_record['skill_categories'] = '; '.join(skill_categories)
                consolidated_record['primary_skill_category'] = group['Skill_Category'].mode().iloc[0] if len(skill_categories) > 0 else 'Unknown'
            
            # Add experience level
            if 'Experience_Category' in group.columns:
                consolidated_record['experience_level'] = group['Experience_Category'].mode().iloc[0]
            
            # Add geographic data
            if 'Region_Category' in group.columns:
                consolidated_record['region'] = group['Region_Category'].mode().iloc[0]
            
            # Add business value score
            if 'Resource_Value_Score' in group.columns:
                consolidated_record['value_score'] = group['Resource_Value_Score'].mean()
            
            deduplicated_records.append(consolidated_record)
        
        # Create deduplicated dataframe
        dedup_df = pd.DataFrame(deduplicated_records)
        
        # Store statistics
        self.dedup_stats = {
            'original_records': len(df),
            'unique_persons': len(dedup_df),
            'duplicate_ratio': len(df) / len(dedup_df),
            'avg_skills_per_person': dedup_df['skill_count'].mean(),
            'max_skills_per_person': dedup_df['skill_count'].max()
        }
        
        print(f"✅ Deduplication completed:")
        print(f"   Original records: {self.dedup_stats['original_records']:,}")
        print(f"   Unique persons: {self.dedup_stats['unique_persons']:,}")
        print(f"   Duplicate ratio: {self.dedup_stats['duplicate_ratio']:.1f}x")
        print(f"   Avg skills per person: {self.dedup_stats['avg_skills_per_person']:.1f}")
        
        return dedup_df
    
    def create_skill_matrix(self, dedup_df: pd.DataFrame) -> pd.DataFrame:
        """Create a person-skill matrix for better analysis."""
        
        print("🔧 Creating person-skill matrix...")
        
        # Extract all unique skills
        all_skills = set()
        for skills_str in dedup_df['all_skills'].dropna():
            skills = skills_str.split('; ')
            all_skills.update(skills)
        
        all_skills = sorted(list(all_skills))
        print(f"   Found {len(all_skills)} unique skills across all resources")
        
        # Create binary skill matrix
        skill_matrix = []
        
        for _, person in dedup_df.iterrows():
            person_skills = person['all_skills'].split('; ') if pd.notna(person['all_skills']) else []
            
            skill_row = {
                'resource_id': person['resource_id'],
                'resource_name': person['resource_name'],
                'manager': person['manager'],
                'domain': person['primary_domain'],
                'city': person['city'],
                'skill_count': person['skill_count'],
                'avg_rating': person['avg_rating']
            }
            
            # Add binary indicators for top 50 most common skills
            top_skills = dedup_df['all_skills'].str.split('; ').explode().value_counts().head(50).index
            
            for skill in top_skills:
                skill_key = f"has_{skill.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')[:30]}"
                skill_row[skill_key] = 1 if skill in person_skills else 0
            
            skill_matrix.append(skill_row)
        
        matrix_df = pd.DataFrame(skill_matrix)
        print(f"✅ Skill matrix created: {len(matrix_df)} persons × {len(matrix_df.columns)} features")
        
        return matrix_df

def deduplicate_and_enhance() -> Dict[str, pd.DataFrame]:
    """Main function to deduplicate and enhance resource data."""
    
    # Load enhanced data
    try:
        enhanced_df = pd.read_parquet('data_processed/resource_DETAILS_28_Export_clean_clean_enhanced.parquet')
        print(f"📊 Loaded enhanced data: {enhanced_df.shape}")
    except:
        print("❌ Enhanced data not found, using original data")
        from src.io_loader import load_processed_data
        data = load_processed_data()
        enhanced_df = data['resource_DETAILS_28_Export_clean_clean']
    
    # Initialize deduplicator
    deduplicator = ResourceDeduplicator()
    
    # Deduplicate resources
    dedup_df = deduplicator.deduplicate_resources(enhanced_df)
    
    # Create skill matrix
    skill_matrix = deduplicator.create_skill_matrix(dedup_df)
    
    # Save deduplicated data
    dedup_path = 'data_processed/resources_deduplicated.parquet'
    matrix_path = 'data_processed/skill_matrix.parquet'
    
    dedup_df.to_parquet(dedup_path, index=False)
    skill_matrix.to_parquet(matrix_path, index=False)
    
    print(f"💾 Saved deduplicated data:")
    print(f"   📁 {dedup_path}")
    print(f"   📁 {matrix_path}")
    
    return {
        'deduplicated_resources': dedup_df,
        'skill_matrix': skill_matrix,
        'stats': deduplicator.dedup_stats
    }

def main():
    """Test deduplication functionality."""
    
    try:
        results = deduplicate_and_enhance()
        
        print("\n🎯 DEDUPLICATION RESULTS:")
        print("=" * 50)
        
        stats = results['stats']
        print(f"✅ Original records: {stats['original_records']:,}")
        print(f"✅ Unique persons: {stats['unique_persons']:,}")
        print(f"✅ Reduction ratio: {stats['duplicate_ratio']:.1f}:1")
        print(f"✅ Avg skills per person: {stats['avg_skills_per_person']:.1f}")
        
        # Show sample of deduplicated data
        dedup_df = results['deduplicated_resources']
        print(f"\n📋 SAMPLE DEDUPLICATED DATA:")
        sample_cols = ['resource_name', 'manager', 'skill_count', 'avg_rating', 'primary_domain', 'city']
        available_cols = [col for col in sample_cols if col in dedup_df.columns]
        print(dedup_df[available_cols].head(5).to_string(index=False))
        
        # Show example of person with multiple skills
        multi_skill_person = dedup_df[dedup_df['skill_count'] > 1].iloc[0]
        print(f"\n🎯 EXAMPLE - {multi_skill_person['resource_name']}:")
        print(f"   Skills: {multi_skill_person['all_skills']}")
        print(f"   Skill Count: {multi_skill_person['skill_count']}")
        print(f"   Average Rating: {multi_skill_person['avg_rating']}")
        
    except Exception as e:
        print(f"❌ Deduplication failed: {e}")

if __name__ == "__main__":
    main()
