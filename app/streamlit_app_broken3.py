"""
HPE Talent Intelligence Platform - Clean & Simple Interface
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

# Setup paths for new folder structure
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
project_root = os.path.dirname(os.path.dirname(__file__))
os.chdir(project_root)

# Page config
st.set_page_config(
    page_title="HPE Talent Intelligence",
    page_icon="🎯",
    layout="wide"
)

# Simple CSS
st.markdown("""
<style>
    .main .block-container {
        max-width: 1000px;
        margin: 0 auto;
        padding: 2rem;
    }
    .header {
        background: linear-gradient(90deg, #0073e6, #004c99);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load enhanced data with error handling."""
    try:
        from src.io_loader import load_processed_data
        data = load_processed_data()
        
        # Load DEDUPLICATED data as primary source
        try:
            # First priority: Deduplicated data (correct person count)
            dedup_path = 'data_processed/resources_deduplicated.parquet'
            dedup_df = pd.read_parquet(dedup_path)
            data['resources_corrected'] = dedup_df
            
            # Secondary: Enhanced data for reference
            enhanced_path = 'data_processed/resource_DETAILS_28_Export_clean_clean_enhanced.parquet'
            enhanced_df = pd.read_parquet(enhanced_path)
            data['resource_enhanced'] = enhanced_df
        except:
            pass
        
        return data
    except Exception as e:
        st.error(f"Data loading error: {e}")
        return {}

def show_header():
    """Header."""
    st.markdown("""
    <div class="header">
        <h1>🎯 HPE Talent Intelligence Platform</h1>
        <p>Smart Workforce Planning & Resource Management</p>
    </div>
    """, unsafe_allow_html=True)

def show_overview(data):
    """Overview."""
    st.header("📊 Executive Dashboard")
    
    if not data:
        st.error("⚠️ No data available.")
        return
    
    # Calculate metrics
    total_resources = 0
    total_opportunities = 0
    
    try:
        # Use CORRECTED deduplicated data for accurate counts
        if 'resources_corrected' in data:
            total_resources = len(data['resources_corrected'])
        elif 'resource_DETAILS_28_Export_clean_clean' in data:
            total_resources = data['resource_DETAILS_28_Export_clean_clean']['resource_name'].nunique()
        
        if 'opportunity_RAWDATA_Export_clean_clean' in data:
            total_opportunities = len(data['opportunity_RAWDATA_Export_clean_clean'])
    except:
        pass
    
    # Display enhanced metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👥 Total Resources", f"{total_resources:,}")
    
    with col2:
        st.metric("🎯 Active Opportunities", f"{total_opportunities:,}")
    
    with col3:
        # Show enhanced skill categories if available
        if 'resource_enhanced' in data:
            skill_categories = data['resource_enhanced']['Skill_Category'].nunique()
            st.metric("🛠️ Skill Categories", f"{skill_categories}")
        else:
            st.metric("🛠️ Skills", "139")
    
    with col4:
        st.metric("⚙️ Services", "160")
    
    # Show CORRECTED insights using deduplicated data
    if 'resources_corrected' in data:
        st.markdown("---")
        st.markdown("### ✅ CORRECTED Business Intelligence")
        
        corrected_df = data['resources_corrected']
        
        # Show data quality correction notice
        st.info(f"📊 **Data Quality Fixed**: Showing {len(corrected_df)} unique persons (was {data.get('resource_DETAILS_28_Export_clean_clean', pd.DataFrame()).shape[0] if 'resource_DETAILS_28_Export_clean_clean' in data else 0} inflated records)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🏆 True Skill Distribution:**")
            # Create skill category analysis from deduplicated data
            if 'all_skills' in corrected_df.columns:
                # Count unique persons per skill category
                skill_counts = {}
                for _, person in corrected_df.iterrows():
                    skills = person['all_skills'].split('; ') if pd.notna(person['all_skills']) else []
                    for skill in skills:
                        if 'python' in skill.lower() or 'java' in skill.lower():
                            skill_counts['Programming'] = skill_counts.get('Programming', 0) + 1
                        elif 'cloud' in skill.lower() or 'aws' in skill.lower():
                            skill_counts['Cloud Technologies'] = skill_counts.get('Cloud Technologies', 0) + 1
                        elif 'security' in skill.lower() or 'cyber' in skill.lower():
                            skill_counts['Security'] = skill_counts.get('Security', 0) + 1
                        elif 'management' in skill.lower() or 'leadership' in skill.lower():
                            skill_counts['Management'] = skill_counts.get('Management', 0) + 1
                
                for category, count in sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                    st.markdown(f"• **{category}**: {count} people")
        
        with col2:
            st.markdown("**🌍 Geographic Distribution:**")
            if 'city' in corrected_df.columns:
                cities = corrected_df['city'].value_counts().head(5)
                for city, count in cities.items():
                    st.markdown(f"• **{city}**: {count} people")
        
        with col3:
            st.markdown("**⭐ Skill Depth Analysis:**")
            if 'skill_count' in corrected_df.columns:
                avg_skills = corrected_df['skill_count'].mean()
                max_skills = corrected_df['skill_count'].max()
                multi_skilled = len(corrected_df[corrected_df['skill_count'] >= 10])
                
                st.markdown(f"• **Avg Skills/Person**: {avg_skills:.1f}")
                st.markdown(f"• **Max Skills**: {max_skills}")
                st.markdown(f"• **Multi-skilled (10+)**: {multi_skilled} people")
                st.markdown(f"• **Highly Skilled (20+)**: {len(corrected_df[corrected_df['skill_count'] >= 20])} people")

def show_smart_recommendations(data):
    """AI recommendations - FIXED NO NESTED COLUMNS."""
    st.header("🤖 Smart AI Recommendations")
    
    st.markdown("### 🎯 AI-Powered Resource Matching")
    st.markdown("Describe your project requirements and get intelligent resource recommendations.")
    
    # Input
    project_description = st.text_area(
        "📝 Describe your project needs:",
        placeholder="Example: Need Python developers for AWS cloud infrastructure migration.",
        height=100
    )
    
    num_recommendations = st.slider("Number of recommendations:", 1, 20, 10)
    
    if st.button("🚀 Get Recommendations", type="primary", disabled=not project_description):
        with st.spinner("🤖 AI analyzing requirements..."):
            # Use SMART KEYWORD MATCHING for better results
            if 'resource_DETAILS_28_Export_clean_clean' in data:
                resource_df = data['resource_DETAILS_28_Export_clean_clean']
                
                # IMPROVED Smart keyword matching
                keywords = project_description.lower().split()
                matched_resources = []
                
                # Extract specific technologies with PRIORITY MAPPING
                tech_keywords = []
                primary_tech = None
                
                # Define technology hierarchy (exact matches get priority)
                tech_mapping = {
                    'java': {
                        'variations': ['java'],
                        'exact_skills': ['coding - java', 'java architecture', 'java programming'],
                        'related_skills': ['javascript'],  # Lower priority
                        'priority': 1
                    },
                    'python': {
                        'variations': ['python', 'py'],
                        'exact_skills': ['coding - python', 'python programming', 'data ai - python'],
                        'related_skills': [],
                        'priority': 1
                    },
                    'javascript': {
                        'variations': ['javascript', 'js', 'node'],
                        'exact_skills': ['javascript', 'node.js', 'nodejs'],
                        'related_skills': [],
                        'priority': 1
                    },
                    'aws': {
                        'variations': ['aws', 'amazon web services'],
                        'exact_skills': ['aws', 'amazon web services', 'aws landing zone'],
                        'related_skills': ['cloud'],
                        'priority': 2
                    },
                    'cloud': {
                        'variations': ['cloud', 'azure', 'gcp'],
                        'exact_skills': ['cloud', 'azure', 'private cloud', 'public cloud'],
                        'related_skills': [],
                        'priority': 2
                    }
                }
                
                # Identify primary technology mentioned
                for tech, config in tech_mapping.items():
                    for variation in config['variations']:
                        if variation in project_description.lower():
                            tech_keywords.append(tech)
                            if primary_tech is None:  # First match becomes primary
                                primary_tech = tech
                            break
                
                for _, resource in resource_df.iterrows():
                    skill = str(resource.get('Skill_Certification_Name', '')).lower()
                    domain = str(resource.get('domain', '')).lower()
                    
                    # Calculate relevance score with PRIORITY LOGIC
                    score = 0
                    skill_type = None
                    
                    # PRIORITY 1: Exact technology matches (highest scores)
                    if primary_tech:
                        tech_config = tech_mapping[primary_tech]
                        
                        # Check for exact skill matches
                        for exact_skill in tech_config['exact_skills']:
                            if exact_skill in skill:
                                score += 20  # Highest priority for exact matches
                                skill_type = 'exact'
                                break
                        
                        # Check for related skills (lower priority)
                        if skill_type != 'exact':
                            for related_skill in tech_config['related_skills']:
                                if related_skill in skill:
                                    score += 8  # Lower priority for related skills
                                    skill_type = 'related'
                                    break
                    
                    # PRIORITY 2: Other technology matches
                    for tech in tech_keywords:
                        if tech != primary_tech and tech in skill:
                            score += 10
                    
                    # PRIORITY 3: Domain matching
                    for tech in tech_keywords:
                        if tech in domain:
                            score += 3
                    
                    # PRIORITY 4: General keyword matching (lowest)
                    for keyword in keywords:
                        if keyword not in ['developer', 'developers', 'engineer', 'engineers', 'for', 'and', 'with']:
                            if keyword in skill:
                                score += 1
                    
                    if score > 0:
                        resource_dict = resource.to_dict()
                        resource_dict['relevance_score'] = score
                        matched_resources.append(resource_dict)
                
                # Sort by relevance score
                matched_resources.sort(key=lambda x: x['relevance_score'], reverse=True)
                
                st.success("✅ Smart Analysis Complete!")
                
                # Show what we found
                if matched_resources:
                    # Show relevant skills
                    relevant_skills = set()
                    for resource in matched_resources[:20]:
                        skill = resource.get('Skill_Certification_Name', '')
                        for keyword in keywords:
                            if keyword.lower() in skill.lower():
                                relevant_skills.add(skill)
                    
                    if relevant_skills:
                        st.markdown("**🛠️ Relevant Skills Found:**")
                        for skill in list(relevant_skills)[:8]:
                            st.markdown(f"- **{skill}**")
                    
                    st.markdown("---")
                    
                    # Show top recommendations
                    st.markdown("### 👥 Recommended Resources")
                    
                    for i, resource in enumerate(matched_resources[:num_recommendations], 1):
                        name = resource.get('resource_name', 'Unknown')
                        skill = resource.get('Skill_Certification_Name', 'N/A')
                        rating = resource.get('Rating', 'N/A')
                        domain = resource.get('domain', 'N/A')
                        manager = resource.get('manager', 'N/A')
                        city = resource.get('RMR_City', 'N/A')
                        mru = resource.get('RMR_MRU', 'N/A')
                        relevance = resource.get('relevance_score', 0)
                        
                        # Extract experience years from skill or rating if available
                        experience_years = "N/A"
                        if 'proficiency_rating' in resource:
                            prof_rating = resource.get('Proficieny_Rating', 0)  # Note: typo in original data
                            if prof_rating >= 5:
                                experience_years = "5+ years"
                            elif prof_rating >= 3:
                                experience_years = "3-5 years"
                            elif prof_rating >= 1:
                                experience_years = "1-3 years"
                        
                        # Determine skill match type for better display
                        skill_match_type = "Related"
                        if primary_tech:
                            tech_config = tech_mapping[primary_tech]
                            for exact_skill in tech_config['exact_skills']:
                                if exact_skill in skill.lower():
                                    skill_match_type = "Exact Match"
                                    break
                        
                        st.markdown(f"#### #{i} - {name}")
                        st.markdown(f"**Relevance Score:** {relevance} ({skill_match_type})")
                        st.markdown(f"**🛠️ Primary Skill:** {skill}")
                        st.markdown(f"**⭐ Rating:** {rating}")
                        st.markdown(f"**📅 Experience:** {experience_years}")
                        st.markdown(f"**🏢 Domain:** {domain}")
                        st.markdown(f"**📍 Location:** {city}")
                        st.markdown(f"**👤 Manager:** {manager}")
                        st.markdown(f"**🌍 MRU:** {mru}")
                        
                        # Generate ENHANCED rationale
                        rationale_parts = []
                        
                        # Technology expertise with priority indication
                        if primary_tech and primary_tech in skill.lower():
                            if skill_match_type == "Exact Match":
                                rationale_parts.append(f"EXACT {primary_tech.upper()} specialist")
                            else:
                                rationale_parts.append(f"Expert in {primary_tech.upper()}")
                        
                        # Skill level with experience context
                        if 'master' in rating.lower() or '5' in rating:
                            rationale_parts.append("Master-level expertise")
                        elif 'expert' in rating.lower() or '4' in rating:
                            rationale_parts.append("Expert-level proficiency")
                        elif 'advanced' in rating.lower() or '3' in rating:
                            rationale_parts.append("Advanced skill level")
                        
                        # Experience years
                        if experience_years != "N/A":
                            rationale_parts.append(f"{experience_years} experience")
                        
                        # Location advantages
                        if city in ['Bangalore', 'Pune']:
                            rationale_parts.append("Key tech hub location")
                        elif city in ['Dalian', 'Wuhan']:
                            rationale_parts.append("Asia-Pacific expertise")
                        
                        # Domain expertise
                        if 'cloud' in domain.lower():
                            rationale_parts.append("Cloud domain specialist")
                        elif 'data' in domain.lower():
                            rationale_parts.append("Data engineering focus")
                        elif 'analytics' in domain.lower():
                            rationale_parts.append("Analytics expertise")
                        
                        if rationale_parts:
                            st.info(f"💡 **Why this resource:** {', '.join(rationale_parts)}")
                        else:
                            st.info(f"💡 **Match reason:** Relevant experience in {domain}")
                        
                        st.markdown("---")
                else:
                    st.warning("No matching resources found. Try different keywords.")

def show_resources(data):
    """Resource explorer."""
    st.header("👥 Resource Explorer")
    
    if 'resource_DETAILS_28_Export_clean_clean' not in data:
        st.warning("Resource data not available.")
        return
    
    df = data['resource_DETAILS_28_Export_clean_clean']
    st.write(f"**Total Resources:** {len(df):,}")
    
    # Simple sample
    sample_df = df[['resource_name', 'Skill_Certification_Name', 'Rating', 'domain']].head(10)
    sample_df.columns = ['Name', 'Skill', 'Rating', 'Domain']
    st.dataframe(sample_df, use_container_width=True)

def show_forecasting(data):
    """Clean, professional resource forecasting interface."""
    
    # Clean header with better spacing
    st.markdown("""
    <div style='background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                padding: 2rem; border-radius: 15px; margin: 1rem 0; text-align: center;'>
        <h2 style='color: #495057; margin: 0;'>🔮 Resource Forecasting & Capacity Planning</h2>
        <p style='color: #6c757d; margin: 0.5rem 0 0 0; font-size: 1.1rem;'>
            Strategic workforce planning based on real data and service requirements
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Data quality indicator
    if 'resources_corrected' in data:
        corrected_df = data['resources_corrected']
        st.success(f"✅ **Data Quality**: Using corrected data with {len(corrected_df)} unique people")
    else:
        st.warning("⚠️ **Data Quality**: Using original data (may contain duplicates)")
        return
    
    # Clean, organized controls
    st.markdown("### ⚙️ Forecasting Parameters")
    
    col1, col2, col3 = st.columns([2, 2, 3])
    
    with col1:
        forecast_months = st.selectbox(
            "Forecast Horizon:", 
            [3, 6, 12, 18, 24],
            index=2,
            help="How far ahead to forecast resource needs"
        )
    
    with col2:
        analysis_type = st.selectbox(
            "Analysis Type:",
            ["Capacity vs Demand", "Skill Gap Analysis", "Geographic Distribution"],
            help="Type of forecasting analysis to perform"
        )
    
    with col3:
        st.markdown("<br/>", unsafe_allow_html=True)
        if st.button("🚀 Generate Professional Forecast", type="primary", use_container_width=True):
            with st.spinner("🔮 Generating professional forecast analysis..."):
                try:
                    # Use CORRECTED data for realistic forecasting
                    if 'resources_corrected' in data:
                        corrected_df = data['resources_corrected']
                        service_df = data.get('service_skillset_Services_to_skillsets_Mapping_Master_v5_clean_clean')
                        
                        # Professional results display
                        st.markdown("""
                        <div style='background: #d4edda; padding: 1.5rem; border-radius: 10px; 
                                    border-left: 5px solid #28a745; margin: 1rem 0;'>
                            <h4 style='color: #155724; margin: 0;'>✅ Professional Forecast Generated</h4>
                            <p style='color: #155724; margin: 0.5rem 0 0 0;'>
                                Analysis complete using corrected data and actual service requirements
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Calculate realistic metrics
                        total_people = len(corrected_df)
                        avg_skills_per_person = corrected_df['skill_count'].mean()
                        
                        # Executive metrics with better styling
                        st.markdown("### 📊 Executive Capacity Overview")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                label="👥 Available Talent", 
                                value=f"{total_people}",
                                help="Unique people in workforce (deduplicated)"
                            )
                        
                        with col2:
                            st.metric(
                                label="🛠️ Skills Depth", 
                                value=f"{avg_skills_per_person:.1f}",
                                help="Average skills per person"
                            )
                        
                        with col3:
                            if service_df is not None:
                                service_complexity = service_df.groupby('New Service Name')['Skill Set'].count()
                                complex_services = len(service_complexity[service_complexity >= 20])
                                st.metric(
                                    label="🎯 Complex Services", 
                                    value=f"{complex_services}",
                                    help="Services requiring 20+ skills"
                                )
                        
                        with col4:
                            if service_df is not None:
                                total_service_demand = (service_complexity / avg_skills_per_person).sum()
                                capacity_ratio = total_people / total_service_demand if total_service_demand > 0 else 1
                                st.metric(
                                    label="📈 Capacity Ratio", 
                                    value=f"{capacity_ratio:.1f}x",
                                    help="Available capacity vs estimated demand"
                                )
                            
                            # Domain capacity analysis
                            st.markdown("### 🏢 Domain Capacity Analysis")
                            
                            domain_capacity = corrected_df.groupby('primary_domain').agg({
                                'resource_name': 'count',
                                'skill_count': 'mean',
                                'avg_rating': 'mean'
                            }).round(1)
                            domain_capacity.columns = ['People_Available', 'Avg_Skills', 'Avg_Rating']
                            domain_capacity = domain_capacity.sort_values('People_Available', ascending=False)
                            
                            # Create capacity chart
                            fig = px.bar(
                                domain_capacity.reset_index(),
                                x='primary_domain',
                                y='People_Available',
                                title='Available People by Domain',
                                labels={'primary_domain': 'Domain', 'People_Available': 'Available People'}
                            )
                            fig.update_layout(xaxis_tickangle=-45, height=400)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Service requirements analysis
                            st.markdown("### ⚙️ Service Requirements vs Capacity")
                            
                            # Top services by complexity
                            top_services = service_complexity.nlargest(10).reset_index()
                            top_services.columns = ['Service_Name', 'Skills_Required']
                            top_services['Est_People_Needed'] = (top_services['Skills_Required'] / avg_skills_per_person).round().astype(int)
                            
                            # Create requirements chart
                            fig2 = px.bar(
                                top_services,
                                x='Service_Name',
                                y='Est_People_Needed',
                                title='Estimated People Required for Top Services',
                                labels={'Service_Name': 'Service', 'Est_People_Needed': 'People Needed'}
                            )
                            fig2.update_layout(xaxis_tickangle=-45, height=400)
                            st.plotly_chart(fig2, use_container_width=True)
                            
                            # Show detailed requirements table
                            st.markdown("### 📋 Detailed Service Requirements")
                            st.dataframe(top_services.head(15), use_container_width=True)
                            
                            # Capacity recommendations
                            st.markdown("### 💡 Realistic Capacity Recommendations")
                            
                            # Calculate gaps
                            cybersecurity_people = len(corrected_df[corrected_df['primary_domain'].str.contains('Cyber', case=False, na=False)])
                            ai_people = len(corrected_df[corrected_df['primary_domain'].str.contains('Data|AI', case=False, na=False)])
                            
                            if cybersecurity_people < 20:
                                st.error(f"🚨 **Cybersecurity Gap**: Only {cybersecurity_people} people available for high-demand security services")
                                st.markdown("**Action Required:** Hire 10-15 cybersecurity specialists")
                            
                            if ai_people < 30:
                                st.warning(f"⚠️ **AI/Data Gap**: Only {ai_people} people for growing AI service demand")
                                st.markdown("**Recommendation:** Expand AI/Data team by 15-20 people")
                            
                            # Geographic capacity analysis
                            geographic_dist = corrected_df['city'].value_counts()
                            if len(geographic_dist) > 0 and geographic_dist.iloc[0] / total_people > 0.8:
                                st.info("💡 **Geographic Concentration**: Consider expanding to additional locations for risk mitigation")
                        
                        else:
                            st.warning("Service mapping data not available for detailed analysis")
                    
                except Exception as e:
                    st.error(f"❌ Forecasting failed: {str(e)}")
                    st.markdown("Please ensure all required data files are available.")

def show_export():
    """Enhanced export with business intelligence."""
    st.header("📊 Enhanced Data Export")
    
    st.markdown("### 🎯 Comprehensive Business Intelligence Export")
    st.markdown("Generate Excel export with enhanced columns and business insights.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🆕 Enhanced Features:**
        - **21 New Business Columns** (Skill categories, Experience levels, etc.)
        - **Geographic Intelligence** (Region categories, Timezone data)
        - **Business Value Scoring** (Resource ranking and scarcity analysis)
        - **Executive Summary** (Key metrics and insights)
        - **Professional Formatting** (Charts and business-ready presentation)
        """)
    
    with col2:
        if st.button("🚀 Generate Enhanced Export", type="primary"):
            with st.spinner("Creating enhanced Excel export..."):
                try:
                    from src.enhanced_excel_export import create_enhanced_excel_export
                    output_file = create_enhanced_excel_export()
                    
                    st.success("✅ Enhanced export created!")
                    st.code(f"File: {output_file}")
                    
                    # Show what's included
                    st.markdown("**📊 Worksheets Created:**")
                    st.markdown("• Enhanced_Resources (18K+ records with 31 columns)")
                    st.markdown("• Skill_Intelligence (Categorized skill analysis)")
                    st.markdown("• Geographic_Intelligence (Regional distribution)")
                    st.markdown("• Top_100_Resources (Highest value resources)")
                    st.markdown("• Scarcity_Analysis (Skills supply/demand)")
                    st.markdown("• Executive_Summary (Key business metrics)")
                    
                except Exception as e:
                    st.error(f"❌ Export failed: {str(e)}")

def main():
    """Main app."""
    show_header()
    
    # Load data
    data = load_data()
    
    # Enhanced tabs with forecasting
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🤖 Smart Recommendations",
        "🔮 Resource Forecasting",
        "👥 Resources", 
        "📋 Export"
    ])
    
    with tab1:
        show_overview(data)
    
    with tab2:
        show_smart_recommendations(data)
    
    with tab3:
        show_forecasting(data)
    
    with tab4:
        show_resources(data)
    
    with tab5:
        show_export()
    
    st.markdown("---")
    st.markdown("*HPE Talent Intelligence Platform v2.0*")

if __name__ == "__main__":
    main()
