"""
Enhanced Resource Forecasting Module
Provides comprehensive capacity planning with predictive analytics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class EnhancedForecaster:
    """Advanced forecasting with multiple analysis dimensions."""
    
    def __init__(self, resources_df: pd.DataFrame, services_df: pd.DataFrame = None):
        """Initialize with resource and service data."""
        self.resources_df = resources_df
        self.services_df = services_df
        self.analysis_cache = {}
        
    def calculate_capacity_metrics(self) -> Dict:
        """Calculate comprehensive capacity metrics."""
        metrics = {}
        
        # Basic counts
        metrics['total_resources'] = len(self.resources_df)
        metrics['unique_skills'] = self._count_unique_skills()
        
        # Skill distribution
        metrics['skill_distribution'] = self._analyze_skill_distribution()
        
        # Domain analysis
        metrics['domain_analysis'] = self._analyze_domains()
        
        # Geographic analysis
        metrics['geographic_analysis'] = self._analyze_geography()
        
        # Utilization metrics
        metrics['utilization'] = self._calculate_utilization()
        
        # Skill gaps
        metrics['skill_gaps'] = self._identify_skill_gaps()
        
        # Growth areas
        metrics['growth_areas'] = self._identify_growth_areas()
        
        return metrics
    
    def _count_unique_skills(self) -> int:
        """Count unique skills across all resources."""
        if 'all_skills' in self.resources_df.columns:
            all_skills = []
            for skills in self.resources_df['all_skills'].dropna():
                all_skills.extend(skills.split('; '))
            return len(set(all_skills))
        return 0
    
    def _analyze_skill_distribution(self) -> Dict:
        """Analyze skill distribution patterns."""
        distribution = {
            'beginner': 0,      # 1-5 skills
            'intermediate': 0,  # 6-15 skills
            'advanced': 0,      # 16-25 skills
            'expert': 0         # 26+ skills
        }
        
        if 'skill_count' in self.resources_df.columns:
            for count in self.resources_df['skill_count']:
                if count <= 5:
                    distribution['beginner'] += 1
                elif count <= 15:
                    distribution['intermediate'] += 1
                elif count <= 25:
                    distribution['advanced'] += 1
                else:
                    distribution['expert'] += 1
        
        # Add percentages
        total = sum(distribution.values())
        if total > 0:
            distribution['percentages'] = {
                k: (v/total)*100 for k, v in distribution.items() if k != 'percentages'
            }
        
        return distribution
    
    def _analyze_domains(self) -> Dict:
        """Detailed domain analysis."""
        domain_analysis = {}
        
        if 'primary_domain' in self.resources_df.columns:
            # Count by domain
            domain_counts = self.resources_df['primary_domain'].value_counts()
            
            for domain, count in domain_counts.items():
                domain_df = self.resources_df[self.resources_df['primary_domain'] == domain]
                
                domain_analysis[domain] = {
                    'count': count,
                    'percentage': (count / len(self.resources_df)) * 100,
                    'avg_skills': domain_df['skill_count'].mean() if 'skill_count' in domain_df.columns else 0,
                    'avg_rating': domain_df['avg_rating'].mean() if 'avg_rating' in domain_df.columns else 0,
                    'cities': domain_df['city'].nunique() if 'city' in domain_df.columns else 0,
                    'top_skills': self._get_top_skills_for_domain(domain_df)
                }
        
        return domain_analysis
    
    def _get_top_skills_for_domain(self, domain_df: pd.DataFrame, top_n: int = 5) -> List[str]:
        """Get top skills for a domain."""
        if 'all_skills' not in domain_df.columns:
            return []
        
        skill_counts = {}
        for skills in domain_df['all_skills'].dropna():
            for skill in skills.split('; '):
                skill = skill.strip()
                if skill:
                    skill_counts[skill] = skill_counts.get(skill, 0) + 1
        
        sorted_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)
        return [skill for skill, _ in sorted_skills[:top_n]]
    
    def _analyze_geography(self) -> Dict:
        """Geographic distribution analysis."""
        geo_analysis = {}
        
        if 'city' in self.resources_df.columns:
            city_counts = self.resources_df['city'].value_counts()
            
            # Concentration risk
            top_city_percentage = (city_counts.iloc[0] / len(self.resources_df)) * 100 if len(city_counts) > 0 else 0
            
            geo_analysis = {
                'total_locations': len(city_counts),
                'top_locations': dict(city_counts.head(10)),
                'concentration_risk': 'High' if top_city_percentage > 50 else 'Medium' if top_city_percentage > 30 else 'Low',
                'top_city_percentage': top_city_percentage,
                'geographic_diversity_score': self._calculate_diversity_score(city_counts)
            }
        
        return geo_analysis
    
    def _calculate_diversity_score(self, counts: pd.Series) -> float:
        """Calculate diversity score (0-100)."""
        if len(counts) == 0:
            return 0
        
        # Shannon entropy normalized to 0-100
        proportions = counts / counts.sum()
        entropy = -sum(p * np.log(p) for p in proportions if p > 0)
        max_entropy = np.log(len(counts))
        
        return (entropy / max_entropy * 100) if max_entropy > 0 else 0
    
    def _calculate_utilization(self) -> Dict:
        """Calculate resource utilization metrics."""
        # This would connect to actual assignment data in production
        return {
            'estimated_utilization': 75,  # Placeholder
            'bench_strength': 25,
            'available_immediately': int(len(self.resources_df) * 0.25),
            'available_within_month': int(len(self.resources_df) * 0.40)
        }
    
    def _identify_skill_gaps(self) -> List[Dict]:
        """Identify critical skill gaps."""
        gaps = []
        
        # Define critical skills and minimum requirements
        critical_skills = {
            'AI/Machine Learning': 50,
            'Cloud Architecture': 40,
            'Cybersecurity': 30,
            'Data Engineering': 35,
            'DevOps': 45,
            'Kubernetes': 25,
            'Python': 60,
            'Java': 50,
            'SAP': 20,
            'Network Security': 15
        }
        
        # Count actual resources with these skills
        for skill, required in critical_skills.items():
            if 'all_skills' in self.resources_df.columns:
                actual = sum(1 for skills in self.resources_df['all_skills'].dropna() 
                           if skill.lower() in skills.lower())
                
                if actual < required:
                    gaps.append({
                        'skill': skill,
                        'required': required,
                        'actual': actual,
                        'gap': required - actual,
                        'severity': 'Critical' if actual < required * 0.5 else 'High' if actual < required * 0.75 else 'Medium'
                    })
        
        return sorted(gaps, key=lambda x: x['gap'], reverse=True)
    
    def _identify_growth_areas(self) -> List[Dict]:
        """Identify areas with growth potential."""
        growth_areas = []
        
        # Emerging technology areas
        emerging_tech = {
            'Generative AI': {'current': 5, 'projected': 50, 'timeline': '6 months'},
            'Edge Computing': {'current': 10, 'projected': 30, 'timeline': '12 months'},
            'Quantum Computing': {'current': 1, 'projected': 10, 'timeline': '24 months'},
            '5G Networks': {'current': 8, 'projected': 25, 'timeline': '12 months'},
            'Blockchain': {'current': 3, 'projected': 15, 'timeline': '18 months'},
            'IoT Security': {'current': 12, 'projected': 35, 'timeline': '12 months'}
        }
        
        for tech, metrics in emerging_tech.items():
            growth_areas.append({
                'technology': tech,
                'current_resources': metrics['current'],
                'projected_need': metrics['projected'],
                'growth_required': metrics['projected'] - metrics['current'],
                'timeline': metrics['timeline'],
                'priority': 'High' if metrics['timeline'] == '6 months' else 'Medium'
            })
        
        return sorted(growth_areas, key=lambda x: x['growth_required'], reverse=True)
    
    def forecast_demand(self, months: int = 12, growth_rate: float = 0.15) -> Dict:
        """Forecast resource demand."""
        current_total = len(self.resources_df)
        
        # Monthly projections
        projections = []
        for month in range(1, months + 1):
            growth_factor = (1 + growth_rate/12) ** month
            projected_need = int(current_total * growth_factor)
            
            projections.append({
                'month': month,
                'projected_need': projected_need,
                'additional_required': projected_need - current_total
            })
        
        # Domain-specific forecasts
        domain_forecasts = {}
        if 'primary_domain' in self.resources_df.columns:
            for domain in self.resources_df['primary_domain'].unique():
                if pd.notna(domain):
                    domain_count = len(self.resources_df[self.resources_df['primary_domain'] == domain])
                    domain_forecasts[domain] = {
                        'current': domain_count,
                        'projected': int(domain_count * (1 + growth_rate)),
                        'gap': int(domain_count * growth_rate)
                    }
        
        return {
            'timeline': f'{months} months',
            'growth_rate': f'{growth_rate*100:.1f}%',
            'current_resources': current_total,
            'projected_need': projections[-1]['projected_need'],
            'total_hiring_required': projections[-1]['additional_required'],
            'monthly_projections': projections,
            'domain_forecasts': domain_forecasts
        }
    
    def generate_recommendations(self, metrics: Dict) -> List[Dict]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Check skill gaps
        if 'skill_gaps' in metrics:
            critical_gaps = [g for g in metrics['skill_gaps'] if g['severity'] == 'Critical']
            if critical_gaps:
                recommendations.append({
                    'priority': 'Critical',
                    'category': 'Skill Gap',
                    'title': 'Address Critical Skill Shortages',
                    'description': f"Immediate hiring needed for {len(critical_gaps)} critical skills",
                    'action_items': [
                        f"Hire {g['gap']} {g['skill']} specialists" for g in critical_gaps[:3]
                    ],
                    'impact': 'High',
                    'timeline': 'Immediate'
                })
        
        # Check geographic concentration
        if 'geographic_analysis' in metrics:
            geo = metrics['geographic_analysis']
            if geo.get('concentration_risk') == 'High':
                recommendations.append({
                    'priority': 'High',
                    'category': 'Risk Mitigation',
                    'title': 'Reduce Geographic Concentration Risk',
                    'description': f"{geo['top_city_percentage']:.1f}% of resources in single location",
                    'action_items': [
                        'Establish satellite offices in 2-3 new locations',
                        'Implement remote work program',
                        'Create disaster recovery plan'
                    ],
                    'impact': 'High',
                    'timeline': '3-6 months'
                })
        
        # Check domain balance
        if 'domain_analysis' in metrics:
            domains = metrics['domain_analysis']
            
            # Find underrepresented domains
            avg_percentage = 100 / len(domains) if domains else 0
            underrepresented = [d for d, v in domains.items() 
                              if v['percentage'] < avg_percentage * 0.5]
            
            if underrepresented:
                recommendations.append({
                    'priority': 'Medium',
                    'category': 'Capability Building',
                    'title': 'Strengthen Underrepresented Domains',
                    'description': f"{len(underrepresented)} domains need reinforcement",
                    'action_items': [
                        f"Increase {d} team by 50%" for d in underrepresented[:3]
                    ],
                    'impact': 'Medium',
                    'timeline': '6-12 months'
                })
        
        # Check skill distribution
        if 'skill_distribution' in metrics:
            dist = metrics['skill_distribution']
            if 'percentages' in dist:
                expert_percentage = dist['percentages'].get('expert', 0)
                if expert_percentage < 10:
                    recommendations.append({
                        'priority': 'Medium',
                        'category': 'Talent Development',
                        'title': 'Develop More Expert-Level Resources',
                        'description': f"Only {expert_percentage:.1f}% of resources at expert level",
                        'action_items': [
                            'Create advanced certification program',
                            'Implement mentorship program',
                            'Provide specialized training paths'
                        ],
                        'impact': 'Medium',
                        'timeline': '6-12 months'
                    })
        
        # Growth areas
        if 'growth_areas' in metrics:
            high_growth = [g for g in metrics['growth_areas'] if g['priority'] == 'High']
            if high_growth:
                recommendations.append({
                    'priority': 'High',
                    'category': 'Strategic Growth',
                    'title': 'Prepare for Emerging Technologies',
                    'description': f"{len(high_growth)} high-priority growth areas identified",
                    'action_items': [
                        f"Build {g['technology']} capability ({g['growth_required']} resources needed)"
                        for g in high_growth[:3]
                    ],
                    'impact': 'High',
                    'timeline': '3-6 months'
                })
        
        return sorted(recommendations, key=lambda x: 
                     {'Critical': 0, 'High': 1, 'Medium': 2}.get(x['priority'], 3))
    
    def create_scenario_analysis(self, scenarios: List[Dict]) -> Dict:
        """Run what-if scenario analysis."""
        results = {}
        
        for scenario in scenarios:
            name = scenario.get('name', 'Unnamed')
            growth = scenario.get('growth_rate', 0.15)
            attrition = scenario.get('attrition_rate', 0.10)
            timeline = scenario.get('months', 12)
            
            current = len(self.resources_df)
            
            # Calculate net resources after growth and attrition
            monthly_growth = growth / 12
            monthly_attrition = attrition / 12
            
            projections = []
            resources = current
            
            for month in range(1, timeline + 1):
                # Apply attrition
                resources = resources * (1 - monthly_attrition)
                # Apply growth (hiring)
                resources = resources * (1 + monthly_growth)
                
                projections.append({
                    'month': month,
                    'resources': int(resources),
                    'net_change': int(resources - current)
                })
            
            results[name] = {
                'initial': current,
                'final': projections[-1]['resources'],
                'net_change': projections[-1]['net_change'],
                'monthly_projections': projections,
                'parameters': scenario
            }
        
        return results


def create_capacity_dashboard(resources_df: pd.DataFrame, 
                            services_df: pd.DataFrame = None) -> Dict:
    """Create comprehensive capacity dashboard data."""
    forecaster = EnhancedForecaster(resources_df, services_df)
    
    # Calculate all metrics
    metrics = forecaster.calculate_capacity_metrics()
    
    # Generate forecast
    forecast = forecaster.forecast_demand(months=12, growth_rate=0.15)
    
    # Generate recommendations
    recommendations = forecaster.generate_recommendations(metrics)
    
    # Run scenarios
    scenarios = [
        {'name': 'Conservative', 'growth_rate': 0.10, 'attrition_rate': 0.08, 'months': 12},
        {'name': 'Baseline', 'growth_rate': 0.15, 'attrition_rate': 0.10, 'months': 12},
        {'name': 'Aggressive', 'growth_rate': 0.25, 'attrition_rate': 0.12, 'months': 12}
    ]
    scenario_results = forecaster.create_scenario_analysis(scenarios)
    
    return {
        'metrics': metrics,
        'forecast': forecast,
        'recommendations': recommendations,
        'scenarios': scenario_results,
        'timestamp': datetime.now().isoformat()
    }