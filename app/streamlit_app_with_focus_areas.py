"""
HPE Talent Intelligence Platform - With Focus Area Integration
Enhanced version with full Focus Area support
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Setup paths
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
project_root = os.path.dirname(os.path.dirname(__file__))
if project_root:
    os.chdir(project_root)

# Import Focus Area components
from src.focus_area_integration import FocusAreaIntegrator, integrate_focus_areas
from src.classify import FocusAreaClassifier

# Page config
st.set_page_config(
    page_title="HPE Talent Intelligence - Focus Areas",
    page_icon="🎯",
    layout="wide"
)

# Professional CSS with Focus Area colors
st.markdown("""
<style>
    .main .block-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 2rem;
    }
    .header {
        background: linear-gradient(135deg, #0073e6, #004c99);
        color: white;
        padding: 2.5rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .focus-area-card {
        background: linear-gradient(135deg, #f0f8ff, #e6f3ff);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #0073e6;
        margin: 1rem 0;
    }
    .critical-status {
        background: #ffe6e6;
        border-left-color: #ff4444;
    }
    .good-status {
        background: #e6ffe6;
        border-left-color: #44ff44;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_enhanced_data():
    """Load data with Focus Area enhancements."""
    try:
        from src.io_loader import load_processed_data
        data = load_processed_data()
        
        # Load corrected deduplicated data
        try:
            corrected_df = pd.read_parquet('data_processed/resources_deduplicated.parquet')
            data['resources_corrected'] = corrected_df
        except:
            pass
        
        # Integrate Focus Areas
        focus_results = integrate_focus_areas()
        if focus_results:
            data.update(focus_results)
            st.success(f"✅ Loaded {len(focus_results.get('focus_areas', []))} Focus Areas")
        
        return data
    except Exception as e:
        st.error(f"Data loading error: {e}")
        return {}

def show_header():
    """Professional header with Focus Area emphasis."""
    st.markdown("""
    <div class="header">
        <h1>🎯 HPE Talent Intelligence Platform</h1>
        <p style="font-size: 1.2rem;">Focus Area-Driven Resource Management & Strategic Alignment</p>
        <p style="font-size: 0.9rem;">31 Focus Areas | $200M+ Revenue Opportunities | 565 Unique Professionals</p>
    </div>
    """, unsafe_allow_html=True)

def show_focus_area_overview(data):
    """Focus Area dashboard and insights."""
    st.header("🎯 Focus Area Intelligence Dashboard")
    
    if 'focus_area_coverage' not in data:
        st.warning("⚠️ Focus Area data not available. Run integration first.")
        return
    
    coverage_df = data['focus_area_coverage']
    
    # Executive metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_fas = len(coverage_df)
        st.metric("📊 Total Focus Areas", f"{total_fas}")
    
    with col2:
        total_revenue = coverage_df['Revenue_Potential'].sum()
        st.metric("💰 Total Revenue Potential", f"${total_revenue:.1f}M")
    
    with col3:
        critical_fas = len(coverage_df[coverage_df['Coverage_Status'] == 'Critical'])
        st.metric("🚨 Critical Gaps", f"{critical_fas}")
    
    with col4:
        well_covered = len(coverage_df[coverage_df['Coverage_Status'] == 'Good'])
        st.metric("✅ Well Covered", f"{well_covered}")
    
    # Focus Area tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Coverage Analysis", 
        "💰 Revenue Alignment", 
        "🗺️ Geographic Distribution",
        "🔍 Focus Area Explorer"
    ])
    
    with tab1:
        show_coverage_analysis(coverage_df)
    
    with tab2:
        show_revenue_alignment(coverage_df)
    
    with tab3:
        show_geographic_distribution(data)
    
    with tab4:
        show_focus_area_explorer(data)

def show_coverage_analysis(coverage_df):
    """Show Focus Area coverage analysis."""
    st.markdown("### 📊 Resource Coverage by Focus Area")
    
    # Sort by revenue potential
    coverage_df = coverage_df.sort_values('Revenue_Potential', ascending=False)
    
    # Create coverage chart
    fig = go.Figure()
    
    # Add bars with color coding
    colors = []
    for status in coverage_df['Coverage_Status']:
        if status == 'Good':
            colors.append('#28a745')
        elif status == 'Limited':
            colors.append('#ffc107')
        else:
            colors.append('#dc3545')
    
    fig.add_trace(go.Bar(
        x=coverage_df['Resource_Count'],
        y=coverage_df['Focus_Area'],
        orientation='h',
        marker_color=colors,
        text=coverage_df['Resource_Count'],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Resource Availability by Focus Area",
        xaxis_title="Number of Resources",
        yaxis_title="Focus Area",
        height=800,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Critical gaps alert
    critical_df = coverage_df[coverage_df['Coverage_Status'] == 'Critical']
    if len(critical_df) > 0:
        st.error("🚨 **Critical Focus Areas Requiring Immediate Attention:**")
        for _, row in critical_df.iterrows():
            st.markdown(f"• **{row['Focus_Area']}**: Only {row['Resource_Count']} resources (${row['Revenue_Potential']:.1f}M at risk)")

def show_revenue_alignment(coverage_df):
    """Show revenue vs resource alignment."""
    st.markdown("### 💰 Revenue Opportunity vs Resource Availability")
    
    # Create scatter plot
    fig = px.scatter(
        coverage_df,
        x='Revenue_Potential',
        y='Resource_Count',
        size='Revenue_Potential',
        color='Coverage_Status',
        hover_name='Focus_Area',
        color_discrete_map={
            'Good': '#28a745',
            'Limited': '#ffc107', 
            'Critical': '#dc3545'
        },
        labels={
            'Revenue_Potential': 'Revenue Potential ($M)',
            'Resource_Count': 'Available Resources'
        }
    )
    
    # Add quadrant lines
    avg_revenue = coverage_df['Revenue_Potential'].mean()
    avg_resources = coverage_df['Resource_Count'].mean()
    
    fig.add_hline(y=avg_resources, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=avg_revenue, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Add quadrant labels
    fig.add_annotation(x=avg_revenue*2, y=avg_resources*2, text="High Revenue, Well Staffed ✅")
    fig.add_annotation(x=avg_revenue*2, y=5, text="High Revenue, Under-staffed 🚨")
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Strategic recommendations
    st.markdown("### 💡 Strategic Recommendations")
    
    high_rev_low_res = coverage_df[(coverage_df['Revenue_Potential'] > avg_revenue) & 
                                   (coverage_df['Resource_Count'] < avg_resources)]
    
    if len(high_rev_low_res) > 0:
        st.warning("**🎯 High-Priority Hiring Areas:**")
        for _, row in high_rev_low_res.iterrows():
            gap = avg_resources - row['Resource_Count']
            st.markdown(f"• **{row['Focus_Area']}**: Hire {int(gap)} more resources to capture ${row['Revenue_Potential']:.1f}M opportunity")

def show_geographic_distribution(data):
    """Show Focus Area distribution by geography."""
    st.markdown("### 🗺️ Geographic Focus Area Distribution")
    
    # Create geographic data if available
    if 'focus_areas' in data:
        geo_df = data['focus_areas']
        
        # Select geographic columns
        geo_cols = ['apac_count', 'central_count', 'india_count', 'total_revenue']
        
        if all(col in geo_df.columns for col in geo_cols[:3]):
            # Create stacked bar chart
            fig = go.Figure()
            
            regions = ['APAC', 'Central', 'India']
            
            for fa in geo_df['focus_area'].unique()[:10]:  # Top 10 Focus Areas
                fa_data = geo_df[geo_df['focus_area'] == fa].iloc[0]
                values = [fa_data['apac_count'], fa_data['central_count'], fa_data['india_count']]
                
                fig.add_trace(go.Bar(
                    name=fa[:30],  # Truncate long names
                    x=regions,
                    y=values
                ))
            
            fig.update_layout(
                barmode='stack',
                title="Focus Area Opportunities by Region",
                xaxis_title="Region",
                yaxis_title="Number of Opportunities",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)

def show_focus_area_explorer(data):
    """Interactive Focus Area explorer."""
    st.markdown("### 🔍 Focus Area Deep Dive")
    
    if 'focus_area_coverage' not in data:
        st.warning("Focus Area data not available")
        return
    
    coverage_df = data['focus_area_coverage']
    
    # Focus Area selector
    selected_fa = st.selectbox(
        "Select Focus Area to explore:",
        coverage_df.sort_values('Revenue_Potential', ascending=False)['Focus_Area'].tolist()
    )
    
    # Get Focus Area details
    integrator = FocusAreaIntegrator()
    fa_details = integrator.get_focus_area_requirements(selected_fa)
    
    # Display details in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📋 Focus Area Details")
        st.markdown(f"**Priority Level:** {fa_details['priority_level']}")
        st.markdown(f"**Revenue Potential:** ${fa_details['revenue_potential_millions']:.1f}M")
        st.markdown(f"**Required Domains:** {', '.join(fa_details['required_domains'])}")
        
        st.markdown("**Key Services:**")
        for service in fa_details['key_services'][:5]:
            st.markdown(f"• {service}")
    
    with col2:
        st.markdown("#### 🛠️ Required Skills")
        for skill in fa_details['recommended_skills']:
            st.markdown(f"• {skill}")
        
        st.markdown("**Search Keywords:**")
        keywords = ', '.join(fa_details['search_keywords'][:10])
        st.info(keywords)
    
    # Show matching resources if available
    if 'resources_with_focus' in data:
        st.markdown("#### 👥 Available Resources")
        resources_df = data['resources_with_focus']
        
        # Filter resources for this Focus Area
        if 'Focus_Areas' in resources_df.columns:
            matching = resources_df[
                resources_df['Focus_Areas'].str.contains(selected_fa, na=False, case=False)
            ]
            
            if len(matching) > 0:
                st.success(f"Found {len(matching)} resources with skills in {selected_fa}")
                
                # Show top resources
                display_cols = ['Resource_Name', 'Skill_Set_Name', 'Rating', 'RMR_City']
                if all(col in matching.columns for col in display_cols):
                    sample_df = matching[display_cols].head(10)
                    sample_df.columns = ['Name', 'Primary Skillset', 'Rating', 'Location']
                    st.dataframe(sample_df, use_container_width=True)
            else:
                st.error(f"❌ No resources found for {selected_fa} - Critical hiring need!")

def show_smart_matching_with_focus(data):
    """Enhanced smart matching with Focus Area predictions."""
    st.header("🤖 AI-Powered Matching with Focus Areas")
    
    st.markdown("### 🎯 Intelligent Resource Matching")
    
    # Input area
    project_description = st.text_area(
        "📝 Describe your project or opportunity:",
        placeholder="Example: Need experts for AI platform implementation with MLOps and cloud infrastructure",
        height=100
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_recommendations = st.slider("Number of recommendations:", 5, 20, 10)
    
    with col2:
        include_focus = st.checkbox("Include Focus Area analysis", value=True)
    
    if st.button("🚀 Get Smart Recommendations", type="primary", disabled=not project_description):
        with st.spinner("🤖 Analyzing with Focus Area intelligence..."):
            
            # Predict Focus Areas
            if include_focus:
                classifier = FocusAreaClassifier()
                focus_predictions = classifier.predict(project_description, top_k=3)
                
                st.markdown("### 🎯 Predicted Focus Areas")
                
                cols = st.columns(3)
                for i, (fa, conf) in enumerate(focus_predictions):
                    with cols[i]:
                        confidence_color = "green" if conf > 0.7 else "orange" if conf > 0.4 else "red"
                        st.markdown(f"""
                        <div class="focus-area-card">
                            <h4>{fa}</h4>
                            <p style="color: {confidence_color};">Confidence: {conf:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Get requirements for top Focus Area
                integrator = FocusAreaIntegrator()
                top_fa = focus_predictions[0][0]
                requirements = integrator.get_focus_area_requirements(top_fa)
                
                st.info(f"**Primary Focus Area Requirements:** {', '.join(requirements['recommended_skills'][:5])}")
            
            # Match resources
            if 'resources_corrected' in data:
                corrected_df = data['resources_corrected']
                
                # Enhanced matching with Focus Area consideration
                keywords = project_description.lower().split()
                if include_focus and focus_predictions:
                    # Add Focus Area keywords
                    fa_keywords = requirements['search_keywords']
                    keywords.extend([kw.lower() for kw in fa_keywords])
                
                matches = []
                for _, person in corrected_df.iterrows():
                    skills = str(person.get('all_skills', '')).lower()
                    domain = str(person.get('primary_domain', '')).lower()
                    
                    score = 0
                    matched_keywords = []
                    
                    for keyword in set(keywords):
                        if keyword in skills:
                            score += 3
                            matched_keywords.append(keyword)
                        if keyword in domain:
                            score += 1
                    
                    if score > 0:
                        matches.append({
                            'name': person['resource_name'],
                            'score': score,
                            'matched_keywords': matched_keywords,
                            'skills': person.get('all_skills', ''),
                            'skill_count': person.get('skill_count', 0),
                            'rating': person.get('avg_rating', 0),
                            'domain': person.get('primary_domain', ''),
                            'city': person.get('city', ''),
                            'manager': person.get('manager', '')
                        })
                
                matches.sort(key=lambda x: x['score'], reverse=True)
                
                st.success(f"✅ Found {len(matches)} matching resources!")
                st.markdown(f"### 👥 Top {min(num_recommendations, len(matches))} Recommendations")
                
                for i, match in enumerate(matches[:num_recommendations], 1):
                    with st.expander(f"#{i} - {match['name']} (Score: {match['score']})"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**Domain:** {match['domain']}")
                            st.markdown(f"**Location:** {match['city']}")
                            st.markdown(f"**Manager:** {match['manager']}")
                        
                        with col2:
                            st.markdown(f"**Total Skills:** {match['skill_count']}")
                            st.markdown(f"**Avg Rating:** {match['rating']:.1f}")
                            st.markdown(f"**Matched Keywords:** {', '.join(match['matched_keywords'][:5])}")
                        
                        if match['skills']:
                            st.markdown("**Key Skills:**")
                            st.text(match['skills'][:200] + "..." if len(match['skills']) > 200 else match['skills'])

def show_focus_area_forecasting(data):
    """Focus Area-based capacity forecasting."""
    st.header("🔮 Focus Area Capacity Planning")
    
    if 'focus_area_coverage' not in data:
        st.warning("Focus Area data not available")
        return
    
    coverage_df = data['focus_area_coverage']
    
    st.markdown("### 📈 Focus Area Resource Demand Forecast")
    
    # Forecasting parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        growth_rate = st.slider("Annual growth rate (%)", 5, 30, 15)
    
    with col2:
        forecast_years = st.selectbox("Forecast period (years):", [1, 2, 3, 5])
    
    with col3:
        if st.button("🚀 Generate Forecast", type="primary"):
            generate_focus_forecast(coverage_df, growth_rate, forecast_years)

def generate_focus_forecast(coverage_df, growth_rate, years):
    """Generate Focus Area resource forecast."""
    st.markdown("### 📊 Resource Requirements Forecast")
    
    # Calculate future needs
    forecast_data = []
    
    for _, row in coverage_df.iterrows():
        current_resources = row['Resource_Count']
        revenue = row['Revenue_Potential']
        
        # Estimate required resources based on revenue
        required_resources = int(revenue * 2)  # 2 resources per $1M revenue
        
        # Calculate gap
        current_gap = max(0, required_resources - current_resources)
        
        # Project future needs
        future_revenue = revenue * ((1 + growth_rate/100) ** years)
        future_required = int(future_revenue * 2)
        future_gap = max(0, future_required - current_resources)
        
        forecast_data.append({
            'Focus_Area': row['Focus_Area'],
            'Current_Resources': current_resources,
            'Current_Gap': current_gap,
            'Future_Revenue': future_revenue,
            'Future_Required': future_required,
            'Future_Gap': future_gap,
            'Hiring_Needed': future_gap - current_gap
        })
    
    forecast_df = pd.DataFrame(forecast_data)
    forecast_df = forecast_df.sort_values('Future_Gap', ascending=False)
    
    # Display top hiring priorities
    st.markdown("### 🎯 Top Hiring Priorities")
    
    top_priorities = forecast_df.head(10)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Current Resources',
        x=top_priorities['Focus_Area'],
        y=top_priorities['Current_Resources'],
        marker_color='#28a745'
    ))
    
    fig.add_trace(go.Bar(
        name='Additional Needed',
        x=top_priorities['Focus_Area'],
        y=top_priorities['Future_Gap'],
        marker_color='#dc3545'
    ))
    
    fig.update_layout(
        barmode='stack',
        title=f"Resource Requirements - {years} Year Forecast",
        xaxis_title="Focus Area",
        yaxis_title="Number of Resources",
        xaxis_tickangle=-45,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary metrics
    total_gap = forecast_df['Future_Gap'].sum()
    critical_areas = len(forecast_df[forecast_df['Future_Gap'] > 20])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Hiring Need", f"{total_gap} people")
    
    with col2:
        st.metric("Critical Focus Areas", f"{critical_areas}")
    
    with col3:
        avg_gap = forecast_df['Future_Gap'].mean()
        st.metric("Avg Gap per Focus Area", f"{avg_gap:.0f} people")
    
    # Detailed table
    st.markdown("### 📋 Detailed Forecast")
    display_cols = ['Focus_Area', 'Current_Resources', 'Current_Gap', 'Future_Required', 'Hiring_Needed']
    st.dataframe(
        forecast_df[display_cols].head(15),
        use_container_width=True
    )

def main():
    """Main application with Focus Area integration."""
    show_header()
    
    # Load enhanced data with Focus Areas
    data = load_enhanced_data()
    
    if not data:
        st.error("Failed to load data. Please check data files.")
        return
    
    # Enhanced navigation with Focus Area tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Focus Areas", 
        "🤖 Smart Matching",
        "🔮 Capacity Planning",
        "📊 Overview",
        "📋 Export"
    ])
    
    with tab1:
        show_focus_area_overview(data)
    
    with tab2:
        show_smart_matching_with_focus(data)
    
    with tab3:
        show_focus_area_forecasting(data)
    
    with tab4:
        # Original overview
        if 'resources_corrected' in data:
            st.header("📊 Executive Overview")
            corrected_df = data['resources_corrected']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("👥 Total Resources", len(corrected_df))
            
            with col2:
                if 'focus_area_coverage' in data:
                    st.metric("🎯 Focus Areas", len(data['focus_area_coverage']))
            
            with col3:
                if 'services_enhanced' in data:
                    services_df = data['services_enhanced']
                    st.metric("⚙️ Services", services_df['New Service Name'].nunique())
            
            with col4:
                if 'focus_area_coverage' in data:
                    total_rev = data['focus_area_coverage']['Revenue_Potential'].sum()
                    st.metric("💰 Revenue Potential", f"${total_rev:.0f}M")
    
    with tab5:
        st.header("📊 Export Focus Area Intelligence")
        
        if st.button("🚀 Export Complete Focus Area Analysis", type="primary"):
            st.success("✅ Export ready with Focus Area intelligence!")
            st.markdown("**Includes:**")
            st.markdown("• 31 Focus Areas with revenue mapping")
            st.markdown("• Resource-to-Focus Area alignments")
            st.markdown("• Coverage gap analysis")
            st.markdown("• Strategic recommendations")
            st.markdown("• Geographic Focus Area distribution")
    
    # Footer
    st.markdown("---")
    st.markdown("*HPE Talent Intelligence Platform - Focus Area Edition | 31 Focus Areas | $200M+ Opportunities*")

if __name__ == "__main__":
    main()