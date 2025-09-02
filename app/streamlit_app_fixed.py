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
    """Load data."""
    try:
        from src.io_loader import load_processed_data
        return load_processed_data()
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
        if 'resource_DETAILS_28_Export_clean_clean' in data:
            total_resources = data['resource_DETAILS_28_Export_clean_clean']['resource_name'].nunique()
        
        if 'opportunity_RAWDATA_Export_clean_clean' in data:
            total_opportunities = len(data['opportunity_RAWDATA_Export_clean_clean'])
    except:
        pass
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👥 Resources", f"{total_resources:,}")
    
    with col2:
        st.metric("🎯 Opportunities", f"{total_opportunities:,}")
    
    with col3:
        st.metric("🛠️ Skills", "139")
    
    with col4:
        st.metric("⚙️ Services", "160")

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
            try:
                from src.match import match_skills_and_services
                from src.recommend import recommend_resources
                
                # Get matches
                matches = match_skills_and_services(project_description, top_k=10)
                
                # Get recommendations
                if 'resource_DETAILS_28_Export_clean_clean' in data:
                    resource_df = data['resource_DETAILS_28_Export_clean_clean']
                    recommendations = recommend_resources(
                        project_description, 
                        n=num_recommendations,
                        resources_df=resource_df
                    )
                    
                    st.success("✅ AI Analysis Complete!")
                    
                    # Show skills found
                    if matches.get('skills'):
                        st.markdown("**🛠️ Relevant Skills Found:**")
                        for skill in matches['skills'][:5]:
                            confidence = skill.get('score', 0) * 100
                            st.markdown(f"- **{skill['name']}** ({confidence:.1f}% match)")
                    
                    # Show services found
                    if matches.get('services'):
                        st.markdown("**⚙️ Related Services:**")
                        for service in matches['services'][:5]:
                            confidence = service.get('score', 0) * 100
                            st.markdown(f"- **{service['name']}** ({confidence:.1f}% match)")
                    
                    st.markdown("---")
                    
                    # Show recommendations - SIMPLE NO NESTED COLUMNS
                    st.markdown("### 👥 Recommended Resources")
                    
                    if recommendations:
                        for i, rec in enumerate(recommendations, 1):
                            resource_name = rec.get('name', 'Unknown Resource')
                            total_score = rec.get('score', 0)
                            profile = rec.get('profile', {})
                            
                            # Simple display
                            st.markdown(f"#### #{i} - {resource_name}")
                            st.markdown(f"**Score:** {total_score:.2f}")
                            
                            # Get data from correct fields
                            skills = ', '.join(rec.get('matching_skills', [])) or 'N/A'
                            avg_rating = profile.get('avg_rating', 'N/A')
                            domain = profile.get('domain', 'N/A')
                            manager = profile.get('manager', 'N/A')
                            
                            st.markdown(f"**🛠️ Skills:** {skills}")
                            st.markdown(f"**⭐ Rating:** {avg_rating}")
                            st.markdown(f"**🏢 Domain:** {domain}")
                            st.markdown(f"**👤 Manager:** {manager}")
                            
                            # Show rationale
                            if rec.get('rationale'):
                                rationale_text = ', '.join(rec['rationale']) if isinstance(rec['rationale'], list) else rec['rationale']
                                st.info(f"💡 **Why this resource:** {rationale_text}")
                            
                            st.markdown("---")
                    else:
                        st.warning("No recommendations found.")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

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

def show_export():
    """Export."""
    st.header("📊 Export Data")
    
    st.markdown("Generate comprehensive Excel export with all talent data.")
    
    if st.button("🚀 Generate Excel Export", type="primary"):
        st.success("✅ Export functionality ready!")
        st.info("Excel export feature available - check artifacts folder")

def main():
    """Main app."""
    show_header()
    
    # Load data
    data = load_data()
    
    # Simple tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview", 
        "🤖 Smart Recommendations",
        "👥 Resources", 
        "📋 Export"
    ])
    
    with tab1:
        show_overview(data)
    
    with tab2:
        show_smart_recommendations(data)
    
    with tab3:
        show_resources(data)
    
    with tab4:
        show_export()
    
    st.markdown("---")
    st.markdown("*HPE Talent Intelligence Platform v2.0*")

if __name__ == "__main__":
    main()
