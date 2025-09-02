"""
Resource Forecasting Module for HPE Talent Intelligence Platform.

Provides resource availability forecasting, project timeline analysis,
and capacity planning capabilities.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ResourceForecaster:
    """Resource availability and capacity forecasting system."""
    
    def __init__(self, resource_df: pd.DataFrame, opportunity_df: pd.DataFrame = None):
        self.resource_df = resource_df
        self.opportunity_df = opportunity_df
        self.current_date = datetime.now()
        
    def simulate_project_timelines(self) -> pd.DataFrame:
        """Simulate realistic project timelines and resource allocation."""
        projects = []
        
        if self.opportunity_df is not None:
            for _, opp in self.opportunity_df.iterrows():
                # Generate realistic project timeline
                start_date = self.current_date + timedelta(days=np.random.randint(0, 90))
                duration_days = np.random.randint(30, 365)  # 1 month to 1 year
                end_date = start_date + timedelta(days=duration_days)
                
                # Estimate resource needs based on project
                resource_count = np.random.randint(1, 8)  # 1-8 resources per project
                
                projects.append({
                    'project_id': opp.get('opportunity_id', f"PROJ_{len(projects)}"),
                    'project_name': opp.get('RR Project Name', f"Project {len(projects)}"),
                    'start_date': start_date,
                    'end_date': end_date,
                    'duration_days': duration_days,
                    'required_resources': resource_count,
                    'status': opp.get('RR Status', 'Planning'),
                    'delivery_location': opp.get('RR Delivery Location', 'Unknown'),
                    'practice': opp.get('RR Practice Name', 'Unknown'),
                    'assigned_resource': opp.get('RR Resource Assigned Name', None),
                    'required_skill': opp.get('Skill_Certification_Name', 'General')
                })
        
        # Add some simulated future projects for forecasting
        for i in range(20):  # Add 20 simulated future projects
            start_date = self.current_date + timedelta(days=np.random.randint(30, 180))
            duration_days = np.random.randint(60, 300)
            end_date = start_date + timedelta(days=duration_days)
            
            # Simulate common project types
            project_types = [
                'Cloud Migration', 'Data Analytics', 'Security Assessment',
                'Application Development', 'Infrastructure Upgrade',
                'Digital Transformation', 'AI Implementation'
            ]
            
            project_type = np.random.choice(project_types)
            
            projects.append({
                'project_id': f"FUTURE_PROJ_{i:03d}",
                'project_name': f"{project_type} Project {i+1}",
                'start_date': start_date,
                'end_date': end_date,
                'duration_days': duration_days,
                'required_resources': np.random.randint(2, 10),
                'status': 'Forecasted',
                'delivery_location': np.random.choice(['UK', 'Germany', 'Belgium', 'India']),
                'practice': np.random.choice(['Cloud Services', 'Data Analytics', 'Security', 'Advisory']),
                'assigned_resource': None,
                'required_skill': np.random.choice(['Python', 'Java', 'Cloud Architecture', 'Data Science', 'Security'])
            })
        
        return pd.DataFrame(projects)
    
    def calculate_resource_availability(self, forecast_months: int = 12) -> pd.DataFrame:
        """Calculate resource availability over time."""
        
        # Simulate current utilization for each resource
        availability_data = []
        
        for _, resource in self.resource_df.iterrows():
            resource_name = resource.get('resource_name', 'Unknown')
            domain = resource.get('domain', 'Unknown')
            city = resource.get('RMR_City', 'Unknown')
            skill = resource.get('Skill_Certification_Name', 'Unknown')
            rating = resource.get('Rating', 'Unknown')
            
            # Simulate availability patterns
            base_utilization = np.random.uniform(0.3, 0.9)  # 30-90% utilization
            
            # Generate monthly availability forecast
            for month in range(forecast_months):
                forecast_date = self.current_date + timedelta(days=30 * month)
                
                # Add seasonal variation
                seasonal_factor = 1.0 + 0.2 * np.sin(month * np.pi / 6)  # Seasonal variation
                monthly_utilization = min(0.95, base_utilization * seasonal_factor)
                availability = 1.0 - monthly_utilization
                
                availability_data.append({
                    'resource_name': resource_name,
                    'forecast_date': forecast_date,
                    'forecast_month': forecast_date.strftime('%Y-%m'),
                    'domain': domain,
                    'city': city,
                    'skill': skill,
                    'rating': rating,
                    'utilization': monthly_utilization,
                    'availability': availability,
                    'capacity_hours': availability * 160  # Assuming 160 hours/month
                })
        
        return pd.DataFrame(availability_data)
    
    def forecast_resource_demand(self, projects_df: pd.DataFrame) -> pd.DataFrame:
        """Forecast resource demand based on project timelines."""
        
        demand_data = []
        
        # Create monthly demand forecast
        for month_offset in range(12):
            forecast_date = self.current_date + timedelta(days=30 * month_offset)
            month_start = forecast_date.replace(day=1)
            month_end = (month_start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
            
            # Find projects active in this month
            active_projects = projects_df[
                (projects_df['start_date'] <= month_end) & 
                (projects_df['end_date'] >= month_start)
            ]
            
            # Calculate demand by skill/domain
            skill_demand = {}
            domain_demand = {}
            location_demand = {}
            
            for _, project in active_projects.iterrows():
                skill = project.get('required_skill', 'General')
                location = project.get('delivery_location', 'Unknown')
                practice = project.get('practice', 'Unknown')
                required_count = project.get('required_resources', 1)
                
                skill_demand[skill] = skill_demand.get(skill, 0) + required_count
                domain_demand[practice] = domain_demand.get(practice, 0) + required_count
                location_demand[location] = location_demand.get(location, 0) + required_count
            
            demand_data.append({
                'forecast_month': forecast_date.strftime('%Y-%m'),
                'forecast_date': forecast_date,
                'active_projects': len(active_projects),
                'total_resource_demand': sum(skill_demand.values()),
                'skill_demand': skill_demand,
                'domain_demand': domain_demand,
                'location_demand': location_demand
            })
        
        return pd.DataFrame(demand_data)
    
    def identify_capacity_gaps(self, availability_df: pd.DataFrame, demand_df: pd.DataFrame) -> Dict[str, Any]:
        """Identify capacity gaps and surpluses."""
        
        gaps = []
        
        for _, demand_row in demand_df.iterrows():
            forecast_month = demand_row['forecast_month']
            total_demand = demand_row['total_resource_demand']
            
            # Calculate available capacity for this month
            month_availability = availability_df[availability_df['forecast_month'] == forecast_month]
            total_available_hours = month_availability['capacity_hours'].sum()
            total_available_resources = len(month_availability[month_availability['availability'] > 0.2])
            
            # Calculate gap
            resource_gap = total_demand - total_available_resources
            
            gaps.append({
                'month': forecast_month,
                'demand': total_demand,
                'available_resources': total_available_resources,
                'available_hours': total_available_hours,
                'resource_gap': resource_gap,
                'capacity_utilization': min(1.0, total_demand / max(1, total_available_resources)),
                'risk_level': 'High' if resource_gap > 5 else 'Medium' if resource_gap > 0 else 'Low'
            })
        
        return {
            'capacity_analysis': pd.DataFrame(gaps),
            'peak_demand_month': max(gaps, key=lambda x: x['demand'])['month'],
            'highest_gap': max(gaps, key=lambda x: x['resource_gap'])['resource_gap'],
            'average_utilization': np.mean([g['capacity_utilization'] for g in gaps])
        }
    
    def create_forecasting_charts(self, availability_df: pd.DataFrame, 
                                demand_df: pd.DataFrame, capacity_analysis: Dict) -> Dict[str, Any]:
        """Create forecasting visualizations."""
        
        charts = {}
        
        # 1. Resource Availability Timeline
        monthly_availability = availability_df.groupby('forecast_month').agg({
            'availability': 'mean',
            'capacity_hours': 'sum'
        }).reset_index()
        
        fig_availability = px.line(
            monthly_availability,
            x='forecast_month',
            y='availability',
            title='Average Resource Availability Over Time',
            labels={'availability': 'Availability %', 'forecast_month': 'Month'}
        )
        charts['availability_timeline'] = fig_availability
        
        # 2. Demand vs Capacity
        capacity_df = capacity_analysis['capacity_analysis']
        
        fig_capacity = go.Figure()
        fig_capacity.add_trace(go.Scatter(
            x=capacity_df['month'],
            y=capacity_df['demand'],
            mode='lines+markers',
            name='Resource Demand',
            line=dict(color='red')
        ))
        fig_capacity.add_trace(go.Scatter(
            x=capacity_df['month'],
            y=capacity_df['available_resources'],
            mode='lines+markers',
            name='Available Resources',
            line=dict(color='green')
        ))
        fig_capacity.update_layout(
            title='Resource Demand vs Availability Forecast',
            xaxis_title='Month',
            yaxis_title='Number of Resources'
        )
        charts['demand_vs_capacity'] = fig_capacity
        
        # 3. Capacity Utilization Heatmap
        utilization_data = []
        for _, row in capacity_df.iterrows():
            utilization_data.append({
                'Month': row['month'],
                'Utilization': row['capacity_utilization'],
                'Risk': row['risk_level']
            })
        
        util_df = pd.DataFrame(utilization_data)
        
        fig_heatmap = px.bar(
            util_df,
            x='Month',
            y='Utilization',
            color='Risk',
            title='Resource Utilization Risk by Month',
            color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'}
        )
        charts['utilization_risk'] = fig_heatmap
        
        return charts

def generate_resource_forecast(resource_df: pd.DataFrame, 
                             opportunity_df: pd.DataFrame = None,
                             forecast_months: int = 12) -> Dict[str, Any]:
    """Main function to generate comprehensive resource forecast."""
    
    forecaster = ResourceForecaster(resource_df, opportunity_df)
    
    # Generate forecasts
    projects_timeline = forecaster.simulate_project_timelines()
    availability_forecast = forecaster.calculate_resource_availability(forecast_months)
    demand_forecast = forecaster.forecast_resource_demand(projects_timeline)
    capacity_analysis = forecaster.identify_capacity_gaps(availability_forecast, demand_forecast)
    charts = forecaster.create_forecasting_charts(availability_forecast, demand_forecast, capacity_analysis)
    
    return {
        'projects_timeline': projects_timeline,
        'availability_forecast': availability_forecast,
        'demand_forecast': demand_forecast,
        'capacity_analysis': capacity_analysis,
        'charts': charts,
        'summary': {
            'total_resources': len(resource_df),
            'forecast_months': forecast_months,
            'peak_demand_month': capacity_analysis['peak_demand_month'],
            'highest_gap': capacity_analysis['highest_gap'],
            'average_utilization': capacity_analysis['average_utilization']
        }
    }

def main():
    """Test forecasting functionality."""
    from io_loader import load_processed_data
    
    data = load_processed_data()
    
    if 'resource_DETAILS_28_Export_clean_clean' in data:
        resource_df = data['resource_DETAILS_28_Export_clean_clean']
        opportunity_df = data.get('opportunity_RAWDATA_Export_clean_clean')
        
        forecast_results = generate_resource_forecast(resource_df, opportunity_df)
        
        print("📊 RESOURCE FORECASTING RESULTS:")
        print(f"   Total Resources: {forecast_results['summary']['total_resources']:,}")
        print(f"   Peak Demand Month: {forecast_results['summary']['peak_demand_month']}")
        print(f"   Highest Resource Gap: {forecast_results['summary']['highest_gap']}")
        print(f"   Average Utilization: {forecast_results['summary']['average_utilization']:.1%}")
        
        return forecast_results
    else:
        print("❌ Resource data not available")
        return None

if __name__ == "__main__":
    main()
