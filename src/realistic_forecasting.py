"""
Realistic Resource Forecasting based on actual service requirements and skill mappings.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any

class RealisticForecaster:
    """Realistic resource forecasting based on actual service requirements."""
    
    def __init__(self):
        self.service_requirements = {}
        self.available_capacity = {}
        self.skill_gaps = {}
    
    def analyze_service_requirements(self, service_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze actual service requirements from the mapping data."""
        
        print("🔍 Analyzing service requirements...")
        
        # Group by service to understand requirements
        service_analysis = service_df.groupby('New Service Name').agg({
            'Skill Set': 'count',
            'Technical  Domain': lambda x: ', '.join(x.unique()),
            'Mandatory/ Optional': lambda x: (x == 'Mandatory').sum()
        })
        
        service_analysis.columns = ['Total_Skills_Required', 'Domains', 'Mandatory_Skills']
        service_analysis = service_analysis.reset_index()
        service_analysis = service_analysis.sort_values('Total_Skills_Required', ascending=False)
        
        # Calculate complexity score for each service
        service_analysis['Complexity_Score'] = (
            service_analysis['Total_Skills_Required'] * 0.7 + 
            service_analysis['Mandatory_Skills'] * 0.3
        ).round(1)
        
        # Estimate resource requirements per service
        service_analysis['Estimated_Resources_Needed'] = np.ceil(
            service_analysis['Total_Skills_Required'] / 8  # Assume 8 skills per resource on average
        ).astype(int)
        
        print(f"✅ Analyzed {len(service_analysis)} services")
        return service_analysis
    
    def analyze_available_capacity(self, corrected_resources: pd.DataFrame) -> Dict[str, Any]:
        """Analyze available talent capacity."""
        
        print("📊 Analyzing available talent capacity...")
        
        # Create skill availability matrix
        skill_availability = {}
        domain_capacity = {}
        
        for _, person in corrected_resources.iterrows():
            domain = person['primary_domain']
            skills = person['all_skills'].split('; ') if pd.notna(person['all_skills']) else []
            skill_count = person['skill_count']
            avg_rating = person['avg_rating']
            
            # Track domain capacity
            if domain not in domain_capacity:
                domain_capacity[domain] = {
                    'people_count': 0,
                    'total_skills': 0,
                    'avg_rating': 0,
                    'high_performers': 0
                }
            
            domain_capacity[domain]['people_count'] += 1
            domain_capacity[domain]['total_skills'] += skill_count
            domain_capacity[domain]['avg_rating'] += avg_rating
            
            if avg_rating >= 4.0:
                domain_capacity[domain]['high_performers'] += 1
            
            # Track individual skill availability
            for skill in skills:
                if skill not in skill_availability:
                    skill_availability[skill] = {
                        'available_people': 0,
                        'total_rating': 0,
                        'avg_rating': 0
                    }
                
                skill_availability[skill]['available_people'] += 1
                skill_availability[skill]['total_rating'] += avg_rating
        
        # Calculate averages
        for domain, data in domain_capacity.items():
            if data['people_count'] > 0:
                data['avg_rating'] = data['avg_rating'] / data['people_count']
                data['avg_skills_per_person'] = data['total_skills'] / data['people_count']
        
        for skill, data in skill_availability.items():
            if data['available_people'] > 0:
                data['avg_rating'] = data['total_rating'] / data['available_people']
        
        print(f"✅ Analyzed capacity across {len(domain_capacity)} domains and {len(skill_availability)} skills")
        
        return {
            'domain_capacity': domain_capacity,
            'skill_availability': skill_availability
        }
    
    def create_realistic_demand_forecast(self, service_requirements: pd.DataFrame) -> pd.DataFrame:
        """Create realistic demand forecast based on service complexity."""
        
        print("📈 Creating realistic demand forecast...")
        
        forecast_data = []
        
        # Generate 12-month forecast
        for month_offset in range(12):
            forecast_date = datetime.now() + timedelta(days=30 * month_offset)
            month_name = forecast_date.strftime('%Y-%m')
            
            # Simulate realistic project pipeline
            # High complexity services need more resources and time
            monthly_projects = []
            
            # Select services for this month based on complexity and business priorities
            high_priority_services = service_requirements.head(10)  # Top 10 most complex
            medium_priority_services = service_requirements.iloc[10:25]
            
            # High complexity projects (1-2 per month)
            num_high_projects = np.random.randint(1, 3)
            for _ in range(num_high_projects):
                service = high_priority_services.sample(1).iloc[0]
                monthly_projects.append({
                    'service_name': service['New Service Name'],
                    'resources_needed': service['Estimated_Resources_Needed'],
                    'mandatory_skills': service['Mandatory_Skills'],
                    'complexity': 'High',
                    'duration_months': np.random.randint(3, 8)
                })
            
            # Medium complexity projects (2-4 per month)
            num_medium_projects = np.random.randint(2, 5)
            for _ in range(num_medium_projects):
                service = medium_priority_services.sample(1).iloc[0]
                monthly_projects.append({
                    'service_name': service['New Service Name'],
                    'resources_needed': service['Estimated_Resources_Needed'],
                    'mandatory_skills': service['Mandatory_Skills'],
                    'complexity': 'Medium',
                    'duration_months': np.random.randint(2, 6)
                })
            
            # Calculate total demand for this month
            total_resources_needed = sum(p['resources_needed'] for p in monthly_projects)
            total_mandatory_skills = sum(p['mandatory_skills'] for p in monthly_projects)
            
            forecast_data.append({
                'month': month_name,
                'forecast_date': forecast_date,
                'projects_count': len(monthly_projects),
                'total_resources_needed': total_resources_needed,
                'total_mandatory_skills': total_mandatory_skills,
                'high_complexity_projects': sum(1 for p in monthly_projects if p['complexity'] == 'High'),
                'medium_complexity_projects': sum(1 for p in monthly_projects if p['complexity'] == 'Medium'),
                'avg_project_duration': np.mean([p['duration_months'] for p in monthly_projects]),
                'peak_concurrent_demand': total_resources_needed * 1.2  # Account for overlapping projects
            })
        
        forecast_df = pd.DataFrame(forecast_data)
        print(f"✅ Created 12-month realistic forecast")
        
        return forecast_df
    
    def calculate_skill_gaps(self, service_requirements: pd.DataFrame, 
                           capacity_analysis: Dict) -> pd.DataFrame:
        """Calculate specific skill gaps based on service requirements."""
        
        print("🔍 Calculating skill gaps...")
        
        skill_gaps = []
        skill_availability = capacity_analysis['skill_availability']
        
        # Get required skills from service mappings
        required_skillsets = service_requirements['New Service Name'].value_counts()
        
        for service, demand_count in required_skillsets.head(20).items():
            # Get skills required for this service
            service_skills = service_requirements[
                service_requirements['New Service Name'] == service
            ]['Skill Set'].tolist()
            
            # Calculate availability for these skills
            total_available = 0
            skill_matches = []
            
            for required_skill in service_skills:
                # Find similar skills in our talent pool
                matching_skills = [skill for skill in skill_availability.keys() 
                                 if any(word in skill.lower() for word in required_skill.lower().split())]
                
                skill_capacity = sum(skill_availability[skill]['available_people'] 
                                   for skill in matching_skills)
                total_available += skill_capacity
                
                if skill_capacity > 0:
                    skill_matches.append({
                        'required_skill': required_skill,
                        'available_capacity': skill_capacity,
                        'matching_skills': matching_skills[:3]  # Top 3 matches
                    })
            
            # Calculate gap
            estimated_demand = demand_count * 2  # Assume 2 people per service instance
            gap = max(0, estimated_demand - total_available)
            
            skill_gaps.append({
                'service_name': service,
                'estimated_demand': estimated_demand,
                'available_capacity': total_available,
                'skill_gap': gap,
                'fulfillment_rate': min(1.0, total_available / max(1, estimated_demand)),
                'required_skills_count': len(service_skills),
                'matchable_skills': len(skill_matches),
                'gap_severity': 'High' if gap > 5 else 'Medium' if gap > 2 else 'Low'
            })
        
        gaps_df = pd.DataFrame(skill_gaps)
        gaps_df = gaps_df.sort_values('skill_gap', ascending=False)
        
        print(f"✅ Calculated skill gaps for {len(gaps_df)} services")
        
        return gaps_df

def create_realistic_forecast_report() -> Dict[str, Any]:
    """Create comprehensive realistic forecast report."""
    
    print("🚀 Creating realistic forecast based on actual requirements...")
    
    # Load data
    try:
        service_df = pd.read_parquet('data_processed/service_skillset_Services_to_skillsets_Mapping_Master_v5_clean_clean.parquet')
        corrected_resources = pd.read_parquet('data_processed/resources_deduplicated.parquet')
        
        forecaster = RealisticForecaster()
        
        # Analyze requirements
        service_requirements = forecaster.analyze_service_requirements(service_df)
        capacity_analysis = forecaster.analyze_available_capacity(corrected_resources)
        demand_forecast = forecaster.create_realistic_demand_forecast(service_requirements)
        skill_gaps = forecaster.calculate_skill_gaps(service_requirements, capacity_analysis)
        
        # Create summary
        summary = {
            'total_services_analyzed': len(service_requirements),
            'available_talent_pool': len(corrected_resources),
            'avg_skills_per_person': corrected_resources['skill_count'].mean(),
            'peak_monthly_demand': demand_forecast['total_resources_needed'].max(),
            'highest_gap_service': skill_gaps.iloc[0]['service_name'] if len(skill_gaps) > 0 else 'N/A',
            'services_with_gaps': len(skill_gaps[skill_gaps['skill_gap'] > 0]),
            'fulfillable_services': len(skill_gaps[skill_gaps['fulfillment_rate'] >= 0.8])
        }
        
        return {
            'service_requirements': service_requirements,
            'capacity_analysis': capacity_analysis,
            'demand_forecast': demand_forecast,
            'skill_gaps': skill_gaps,
            'summary': summary
        }
        
    except Exception as e:
        print(f"❌ Forecast creation failed: {e}")
        return None

def main():
    """Test realistic forecasting."""
    results = create_realistic_forecast_report()
    
    if results:
        summary = results['summary']
        print(f"\n🎯 REALISTIC FORECAST SUMMARY:")
        print(f"   📊 Services analyzed: {summary['total_services_analyzed']}")
        print(f"   👥 Available talent: {summary['available_talent_pool']} people")
        print(f"   🛠️ Avg skills per person: {summary['avg_skills_per_person']:.1f}")
        print(f"   📈 Peak monthly demand: {summary['peak_monthly_demand']} resources")
        print(f"   🚨 Services with gaps: {summary['services_with_gaps']}")
        print(f"   ✅ Fulfillable services: {summary['fulfillable_services']}")

if __name__ == "__main__":
    main()
