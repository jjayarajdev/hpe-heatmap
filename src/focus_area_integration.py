"""
Focus Area Integration Module
Integrates Focus Areas from opportunities data with services, skills, and resources
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class FocusAreaIntegrator:
    """Manages Focus Area relationships across all datasets."""
    
    def __init__(self):
        """Initialize with all 31 Focus Areas from opportunities data."""
        self.focus_areas = self._initialize_focus_areas()
        self.focus_area_map = self._create_focus_area_mapping()
        self.revenue_map = self._create_revenue_map()
        
    def _initialize_focus_areas(self) -> List[str]:
        """Initialize the complete list of 31 Focus Areas."""
        return [
            "AI Solutions",
            "AI Platforms", 
            "AI/PCAI Enablement/Adoption",
            "G500 (PM) Other",
            "Data Solutions",
            "Data Platforms",
            "SP00 (XM) Other",
            "Storage Design & Implementation",
            "Storage Migrations",
            "6C00 (IK) Other",
            "Data Center facilities and Sustainability-related consulting & solutions",
            "Co-Lo Services",
            "Modular Data Centres & HPE PODs",
            "Platform Modernization (by adoption of modern landing zones)",
            "MultiCloud Management (Morpheus, OpsRamp, FinOps)",
            "Cloud-Native Platforms",
            "6000 (IJ) Other",
            "SAP",
            "Legacy Application Modernization",
            "Data Center, Cloud and Telco Networking",
            "Edge Networking",
            "Client Virtualisation (VDI)",
            "Workplace Solutions",
            "Security for AI & AI Powered Security",
            "Data Protection & Data Platform Security",
            "Cybersecurity Advisory & Transformation",
            "Cloud Platforms & App Modernization Security",
            "Network Security",
            "Hybrid Workplace Security",
            "HPE GreenLake Solutions Enablement/Adoption",
            "4J00 (SX) Other"
        ]
    
    def _create_focus_area_mapping(self) -> Dict[str, Dict]:
        """Create comprehensive Focus Area to domain/service mapping."""
        return {
            "AI Solutions": {
                "domains": ["AI", "ADV", "DATA"],
                "keywords": ["artificial intelligence", "machine learning", "deep learning", "ai model"],
                "services": ["AI Proof of Value", "AI Transformation", "AI Use case"],
                "priority": 1,
                "revenue_potential": 51.3
            },
            "AI Platforms": {
                "domains": ["AI", "DATA", "CLD"],
                "keywords": ["mlops", "ai infrastructure", "model deployment", "ai platform"],
                "services": ["MLOps Ecosystem", "AI Model Migration", "MLOps Transformation"],
                "priority": 1,
                "revenue_potential": 25.1
            },
            "Data Solutions": {
                "domains": ["DATA", "APP"],
                "keywords": ["database", "data governance", "data management", "analytics"],
                "services": ["Data Governance", "Database Assessment", "Data Services"],
                "priority": 1,
                "revenue_potential": 22.0
            },
            "Platform Modernization (by adoption of modern landing zones)": {
                "domains": ["CLD", "INS", "ARCH"],
                "keywords": ["cloud migration", "modernization", "landing zone", "cloud adoption"],
                "services": ["Cloud Migration", "Platform Modernization", "Landing Zone Design"],
                "priority": 1,
                "revenue_potential": 33.3
            },
            "Cloud-Native Platforms": {
                "domains": ["CLD", "APP"],
                "keywords": ["kubernetes", "containers", "microservices", "cloud native"],
                "services": ["Container Adoption", "Workload Migration", "Integration for Cloud Native"],
                "priority": 1,
                "revenue_potential": 18.5
            },
            "MultiCloud Management (Morpheus, OpsRamp, FinOps)": {
                "domains": ["CLD", "ADV"],
                "keywords": ["multicloud", "finops", "cloud management", "morpheus", "opsramp"],
                "services": ["Cloud Cost Optimization", "MultiCloud Management", "FinOps Implementation"],
                "priority": 2,
                "revenue_potential": 15.2
            },
            "SAP": {
                "domains": ["APP", "DATA"],
                "keywords": ["sap", "s4hana", "sap migration", "sap implementation"],
                "services": ["SAP S/4HANA", "SAP Automation", "SAP Blueprint"],
                "priority": 2,
                "revenue_potential": 12.8
            },
            "Security for AI & AI Powered Security": {
                "domains": ["CYB", "AI"],
                "keywords": ["ai security", "security ai", "threat detection", "security automation"],
                "services": ["AI Security Assessment", "Security Automation", "Threat Intelligence"],
                "priority": 1,
                "revenue_potential": 28.7
            },
            "Cybersecurity Advisory & Transformation": {
                "domains": ["CYB", "ADV"],
                "keywords": ["cybersecurity", "security advisory", "security transformation", "risk assessment"],
                "services": ["Security Assessment", "Cybersecurity Advisory", "Security Transformation"],
                "priority": 1,
                "revenue_potential": 24.5
            },
            "Data Storage": {
                "domains": ["INS", "DATA"],
                "keywords": ["storage", "san", "nas", "backup", "data protection"],
                "services": ["Storage Design", "Storage Migration", "Data Protection"],
                "priority": 2,
                "revenue_potential": 16.9
            },
            "Network Security": {
                "domains": ["CYB", "NET"],
                "keywords": ["network security", "firewall", "intrusion detection", "network protection"],
                "services": ["Network Security Assessment", "Firewall Configuration", "Network Protection"],
                "priority": 2,
                "revenue_potential": 11.3
            },
            "Hybrid Workplace Security": {
                "domains": ["CYB", "INS"],
                "keywords": ["workplace security", "endpoint security", "zero trust", "remote security"],
                "services": ["Endpoint Security", "Zero Trust Implementation", "Workplace Security"],
                "priority": 2,
                "revenue_potential": 9.8
            },
            "Legacy Application Modernization": {
                "domains": ["APP", "CLD"],
                "keywords": ["legacy modernization", "application migration", "refactoring", "replatforming"],
                "services": ["Application Modernization", "Legacy Migration", "Application Refactoring"],
                "priority": 2,
                "revenue_potential": 14.2
            },
            "HPE GreenLake Solutions Enablement/Adoption": {
                "domains": ["CLD", "INS"],
                "keywords": ["greenlake", "hpe greenlake", "consumption model", "as a service"],
                "services": ["GreenLake Implementation", "GreenLake Advisory", "Consumption Model Design"],
                "priority": 1,
                "revenue_potential": 21.6
            }
        }
    
    def _create_revenue_map(self) -> Dict[str, Dict[str, float]]:
        """Create Focus Area to geographic revenue mapping."""
        return {
            "AI Solutions & Platforms": {
                "APAC": 7.7, "Central": 5.5, "India": 10.1, 
                "Japan": 5.9, "LASER": 4.2, "North America": 7.6,
                "NWE": 3.1, "UKIMEA": 7.3, "Total": 51.3
            },
            "Data Solutions & Platforms": {
                "APAC": 4.1, "Central": 2.7, "India": 2.8,
                "Japan": 2.7, "LASER": 2.6, "North America": 1.9,
                "NWE": 1.9, "UKIMEA": 4.1, "Total": 22.0
            }
        }
    
    def extract_focus_areas_from_opportunities(self, opp_df: pd.DataFrame) -> pd.DataFrame:
        """Extract and structure Focus Areas from opportunities data."""
        try:
            # Extract Focus Areas from the special structure
            focus_areas_data = []
            
            # The data has business areas in first column and focus areas in second
            for idx, row in opp_df.iterrows():
                if pd.notna(row.get('Unnamed: 1')) and row['Unnamed: 1'] not in ['Focus Area', 'Total', 'M$', '#']:
                    focus_area = row['Unnamed: 1']
                    business_area = row.get('Geo', '')
                    
                    # Extract geographic data
                    geo_data = {
                        'focus_area': focus_area,
                        'business_area': business_area,
                        'apac_count': row.get('APAC', 0),
                        'apac_revenue': row.get('APAC.1', 0),
                        'central_count': row.get('Central', 0),
                        'central_revenue': row.get('Central.1', 0),
                        'india_count': row.get('India', 0),
                        'india_revenue': row.get('India.1', 0),
                        'total_count': row.get('Total', 0),
                        'total_revenue': row.get('Total.1', 0)
                    }
                    focus_areas_data.append(geo_data)
            
            return pd.DataFrame(focus_areas_data)
        except Exception as e:
            logger.error(f"Error extracting Focus Areas: {e}")
            return pd.DataFrame()
    
    def map_services_to_focus_areas(self, services_df: pd.DataFrame) -> pd.DataFrame:
        """Map services to proper Focus Areas (fixing single value issue)."""
        # Preserve existing Focus Area column
        if 'FY25 Focus Area' in services_df.columns:
            services_df['Original_Focus_Area'] = services_df['FY25 Focus Area']
        
        # Enhanced mapping based on service names and domains
        def assign_focus_area(row):
            service_name = str(row.get('New Service Name', '')).lower()
            skill_set = str(row.get('Skill Set', '')).lower()
            domain = str(row.get('Technical \nDomain', row.get('Technical  Domain', ''))).upper()
            
            # Check each Focus Area's keywords and domains
            best_match = None
            best_score = 0
            
            for fa, config in self.focus_area_map.items():
                score = 0
                
                # Domain match
                if domain in config['domains']:
                    score += 3
                
                # Keyword matches in service name
                for keyword in config['keywords']:
                    if keyword in service_name:
                        score += 2
                    if keyword in skill_set:
                        score += 1
                
                # Service name matches
                for service in config['services']:
                    if service.lower() in service_name:
                        score += 5
                
                if score > best_score:
                    best_score = score
                    best_match = fa
            
            # Default based on domain if no match
            if not best_match:
                domain_defaults = {
                    'AI': 'AI Solutions',
                    'DATA': 'Data Solutions',
                    'CLD': 'Cloud-Native Platforms',
                    'CYB': 'Cybersecurity Advisory & Transformation',
                    'APP': 'Legacy Application Modernization',
                    'INS': 'Platform Modernization (by adoption of modern landing zones)'
                }
                best_match = domain_defaults.get(domain, 'Platform Modernization (by adoption of modern landing zones)')
            
            return best_match
        
        # Apply Focus Area assignment
        services_df['Enhanced_Focus_Area'] = services_df.apply(assign_focus_area, axis=1)
        
        # Use enhanced if original is missing or has only one value
        if 'FY25 Focus Area' not in services_df.columns or services_df['FY25 Focus Area'].nunique() <= 1:
            services_df['FY25 Focus Area'] = services_df['Enhanced_Focus_Area']
        
        # Add Focus Area metadata
        services_df['Focus_Area_Priority'] = services_df['FY25 Focus Area'].map(
            lambda x: self.focus_area_map.get(x, {}).get('priority', 3)
        )
        services_df['Focus_Area_Revenue'] = services_df['FY25 Focus Area'].map(
            lambda x: self.focus_area_map.get(x, {}).get('revenue_potential', 0)
        )
        
        return services_df
    
    def link_resources_to_focus_areas(self, resources_df: pd.DataFrame, 
                                     services_df: pd.DataFrame,
                                     skills_df: pd.DataFrame) -> pd.DataFrame:
        """Link resources to Focus Areas through their skills."""
        # Create skillset to Focus Area mapping
        skillset_to_fa = {}
        if 'Skill Set' in services_df.columns and 'FY25 Focus Area' in services_df.columns:
            for _, row in services_df.iterrows():
                skillset = row.get('Skill Set')
                fa = row.get('FY25 Focus Area')
                if pd.notna(skillset) and pd.notna(fa):
                    if skillset not in skillset_to_fa:
                        skillset_to_fa[skillset] = []
                    if fa not in skillset_to_fa[skillset]:
                        skillset_to_fa[skillset].append(fa)
        
        # Map resources to Focus Areas
        def get_resource_focus_areas(skillset_name):
            if pd.isna(skillset_name):
                return None
            
            # Direct match
            if skillset_name in skillset_to_fa:
                return '; '.join(skillset_to_fa[skillset_name])
            
            # Partial match
            focus_areas = []
            for ss, fas in skillset_to_fa.items():
                if ss in str(skillset_name) or str(skillset_name) in ss:
                    focus_areas.extend(fas)
            
            return '; '.join(list(set(focus_areas))) if focus_areas else None
        
        # Apply to resources
        if 'Skill_Set_Name' in resources_df.columns:
            resources_df['Focus_Areas'] = resources_df['Skill_Set_Name'].apply(get_resource_focus_areas)
            
            # Count Focus Areas per resource
            resources_df['Focus_Area_Count'] = resources_df['Focus_Areas'].apply(
                lambda x: len(x.split('; ')) if pd.notna(x) else 0
            )
        
        return resources_df
    
    def calculate_focus_area_coverage(self, resources_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate resource coverage for each Focus Area."""
        coverage_data = []
        
        for fa in self.focus_areas:
            # Count resources with this Focus Area
            if 'Focus_Areas' in resources_df.columns:
                matching = resources_df[
                    resources_df['Focus_Areas'].str.contains(fa, na=False, case=False)
                ]
                resource_count = matching['Resource_Name'].nunique() if 'Resource_Name' in matching.columns else len(matching)
            else:
                resource_count = 0
            
            # Get metadata
            fa_config = self.focus_area_map.get(fa, {})
            
            coverage_data.append({
                'Focus_Area': fa,
                'Resource_Count': resource_count,
                'Priority': fa_config.get('priority', 3),
                'Revenue_Potential': fa_config.get('revenue_potential', 0),
                'Primary_Domains': ', '.join(fa_config.get('domains', [])),
                'Coverage_Status': 'Good' if resource_count > 20 else 'Limited' if resource_count > 5 else 'Critical'
            })
        
        return pd.DataFrame(coverage_data).sort_values('Revenue_Potential', ascending=False)
    
    def classify_text_to_focus_area(self, text: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Classify free text to Focus Areas with confidence scores."""
        text_lower = text.lower()
        scores = {}
        
        for fa, config in self.focus_area_map.items():
            score = 0
            
            # Check keywords
            for keyword in config['keywords']:
                if keyword in text_lower:
                    score += 10
            
            # Check service names
            for service in config['services']:
                if service.lower() in text_lower:
                    score += 15
            
            # Domain keywords
            domain_keywords = {
                'AI': ['artificial', 'intelligence', 'machine learning', 'deep learning'],
                'DATA': ['data', 'database', 'analytics', 'governance'],
                'CLD': ['cloud', 'container', 'kubernetes', 'microservice'],
                'CYB': ['security', 'cyber', 'threat', 'protection'],
                'APP': ['application', 'software', 'development', 'code']
            }
            
            for domain in config['domains']:
                for keyword in domain_keywords.get(domain, []):
                    if keyword in text_lower:
                        score += 5
            
            if score > 0:
                scores[fa] = score
        
        # Normalize scores to probabilities
        total_score = sum(scores.values())
        if total_score > 0:
            probabilities = [(fa, score/total_score) for fa, score in scores.items()]
            probabilities.sort(key=lambda x: x[1], reverse=True)
            return probabilities[:top_k]
        
        return [("Platform Modernization (by adoption of modern landing zones)", 1.0)]
    
    def get_focus_area_requirements(self, focus_area: str) -> Dict:
        """Get detailed requirements for a Focus Area."""
        config = self.focus_area_map.get(focus_area, {})
        
        return {
            'focus_area': focus_area,
            'required_domains': config.get('domains', []),
            'key_services': config.get('services', []),
            'search_keywords': config.get('keywords', []),
            'priority_level': config.get('priority', 3),
            'revenue_potential_millions': config.get('revenue_potential', 0),
            'recommended_skills': self._get_recommended_skills(focus_area)
        }
    
    def _get_recommended_skills(self, focus_area: str) -> List[str]:
        """Get recommended skills for a Focus Area."""
        skill_recommendations = {
            "AI Solutions": ["Python", "TensorFlow", "PyTorch", "Machine Learning", "Deep Learning"],
            "Cloud-Native Platforms": ["Kubernetes", "Docker", "Microservices", "CI/CD", "DevOps"],
            "Data Solutions": ["SQL", "NoSQL", "Data Governance", "ETL", "Data Warehousing"],
            "Cybersecurity Advisory & Transformation": ["CISSP", "Security Architecture", "Risk Assessment", "Compliance"],
            "SAP": ["SAP S/4HANA", "ABAP", "SAP Basis", "SAP Migration", "SAP Security"]
        }
        
        return skill_recommendations.get(focus_area, ["Domain Expertise", "Project Management", "Communication"])


def integrate_focus_areas(data_path: str = "data/") -> Dict[str, pd.DataFrame]:
    """Main function to integrate Focus Areas across all datasets."""
    integrator = FocusAreaIntegrator()
    results = {}
    
    try:
        # Load and process opportunities data
        opp_path = Path(data_path) / "data - 2025-08-22T000557.141.xlsx"
        if opp_path.exists():
            opp_df = pd.read_excel(opp_path)
            focus_areas_df = integrator.extract_focus_areas_from_opportunities(opp_df)
            results['focus_areas'] = focus_areas_df
            logger.info(f"Extracted {len(focus_areas_df)} Focus Areas from opportunities")
        
        # Process services mapping
        services_path = Path(data_path) / "Services_to_skillsets Mapping.xlsx"
        if services_path.exists():
            services_df = pd.read_excel(services_path)
            services_df = integrator.map_services_to_focus_areas(services_df)
            results['services_enhanced'] = services_df
            logger.info(f"Enhanced {len(services_df)} services with Focus Areas")
            logger.info(f"Unique Focus Areas in services: {services_df['FY25 Focus Area'].nunique()}")
        
        # Process resources
        details_path = Path(data_path) / "DETAILS (28).xlsx"
        if details_path.exists() and 'services_enhanced' in results:
            resources_df = pd.read_excel(details_path)
            skills_df = pd.read_excel(Path(data_path) / "Skillsets_to_Skills_mapping.xlsx")
            resources_df = integrator.link_resources_to_focus_areas(
                resources_df, results['services_enhanced'], skills_df
            )
            results['resources_with_focus'] = resources_df
            logger.info(f"Linked {len(resources_df)} resources to Focus Areas")
        
        # Calculate coverage
        if 'resources_with_focus' in results:
            coverage_df = integrator.calculate_focus_area_coverage(results['resources_with_focus'])
            results['focus_area_coverage'] = coverage_df
            logger.info("Calculated Focus Area coverage metrics")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in Focus Area integration: {e}")
        return results


if __name__ == "__main__":
    # Test the integration
    logging.basicConfig(level=logging.INFO)
    results = integrate_focus_areas()
    
    if results:
        print("\n=== Focus Area Integration Results ===")
        for key, df in results.items():
            print(f"\n{key}: {len(df)} records")
            if 'focus' in key.lower():
                print(df.head())