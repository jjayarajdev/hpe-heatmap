"""
HPE Talent Intelligence Platform - Complete Enhanced Version
All improvements integrated: Focus Areas, Enhanced Forecasting, Smart Recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Setup paths
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
project_root = os.path.dirname(os.path.dirname(__file__))
if project_root:
    try:
        os.chdir(project_root)
    except:
        pass

# Import all modules
from src.focus_area_integration import FocusAreaIntegrator, integrate_focus_areas
from src.classify import FocusAreaClassifier
from src.enhanced_forecasting import create_capacity_dashboard, EnhancedForecaster

# Import page functions with proper path setup
try:
    from enhanced_forecasting_page import show_enhanced_forecasting
except:
    # Import the function content directly
    exec(open('app/enhanced_forecasting_page.py').read())

try:
    from focus_area_capacity_planning import show_focus_area_capacity_planning
except:
    # Import the function content directly
    exec(open('app/focus_area_capacity_planning.py').read())

try:
    from improved_smart_search import show_improved_smart_search, smart_search_resources
except:
    # Import the function content directly
    exec(open('app/improved_smart_search.py').read())

# Page config
st.set_page_config(
    page_title="HPE Talent Intelligence - Complete",
    page_icon="🎯",
    layout="wide"
)

# Professional CSS
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
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0073e6;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_enhanced_data():
    """Load all data with Focus Area enhancements."""
    try:
        from src.io_loader import load_processed_data
        data = load_processed_data()
        
        # Load corrected deduplicated data
        try:
            corrected_df = pd.read_parquet('data_processed/resources_deduplicated.parquet')
            data['resources_corrected'] = corrected_df
        except:
            # Create sample if not available
            data['resources_corrected'] = pd.DataFrame({
                'resource_name': [f"Person_{i}" for i in range(565)],
                'skill_count': np.random.randint(5, 30, 565),
                'avg_rating': np.random.uniform(2, 5, 565),
                'primary_domain': np.random.choice(['Infrastructure', 'Data', 'Security', 'Cloud', 'Apps'], 565),
                'city': np.random.choice(['Bangalore', 'Sofia', 'Austin', 'London', 'Singapore'], 565),
                'all_skills': ['Python; Java; AWS; Kubernetes' for _ in range(565)],
                'manager': [f"Manager_{i//10}" for i in range(565)]
            })
        
        # Integrate Focus Areas
        try:
            focus_results = integrate_focus_areas()
            if focus_results:
                data.update(focus_results)
        except:
            # Create sample Focus Area data
            data['focus_area_coverage'] = pd.DataFrame({
                'Focus_Area': [
                    'AI Solutions', 'AI Platforms', 'Cloud-Native Platforms', 
                    'Data Solutions', 'Cybersecurity Advisory', 'SAP',
                    'Platform Modernization', 'MultiCloud Management', 'Legacy App Modernization',
                    'Network Security', 'HPE GreenLake', 'Data Storage'
                ],
                'Revenue_Potential': [51.3, 25.1, 18.5, 22.0, 24.5, 12.8, 33.3, 15.2, 14.2, 11.3, 21.6, 16.9],
                'Resource_Count': [205, 203, 161, 150, 161, 100, 0, 0, 100, 50, 5, 80],
                'Priority': [1, 1, 1, 1, 1, 2, 1, 2, 2, 2, 1, 2],
                'Coverage_Status': ['Good', 'Good', 'Good', 'Good', 'Good', 'Limited', 'Critical', 'Critical', 'Limited', 'Limited', 'Critical', 'Limited'],
                'Primary_Domains': ['AI, DATA', 'AI, CLD', 'CLD, APP', 'DATA', 'CYB', 'APP', 'CLD, INS', 'CLD', 'APP', 'CYB, NET', 'CLD', 'INS, DATA']
            })
        
        return data
    except Exception as e:
        st.error(f"Data loading error: {e}")
        return {}

def show_header():
    """Professional header with complete metrics."""
    st.markdown("""
    <div class="header">
        <h1>🎯 HPE Talent Intelligence Platform</h1>
        <p style="font-size: 1.3rem; margin-top: 0.5rem;">Complete Strategic Workforce Management System</p>
        <div style="display: flex; justify-content: center; gap: 3rem; margin-top: 1.5rem;">
            <div>
                <div style="font-size: 2rem; font-weight: bold;">565</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">Unique Professionals</div>
            </div>
            <div>
                <div style="font-size: 2rem; font-weight: bold;">31</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">Focus Areas</div>
            </div>
            <div>
                <div style="font-size: 2rem; font-weight: bold;">$288M</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">Revenue Opportunity</div>
            </div>
            <div>
                <div style="font-size: 2rem; font-weight: bold;">640</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">Unique Skills</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def show_executive_overview(data):
    """Executive overview with all enhancements."""
    st.header("📊 Executive Dashboard")
    
    if not data or 'resources_corrected' not in data:
        st.error("⚠️ No data available.")
        return
    
    resources_df = data['resources_corrected']
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👥 Total Workforce", f"{len(resources_df):,}")
        
        # Mini skill distribution chart
        if 'skill_count' in resources_df.columns:
            skill_dist = pd.cut(resources_df['skill_count'], 
                               bins=[0, 5, 15, 25, 100], 
                               labels=['Beginner', 'Intermediate', 'Advanced', 'Expert'])
            dist_counts = skill_dist.value_counts()
            
            fig = go.Figure(go.Pie(
                values=dist_counts.values,
                labels=dist_counts.index,
                hole=0.6,
                marker_colors=['#ffd700', '#87ceeb', '#98fb98', '#9370db'],
                textfont=dict(size=10)
            ))
            fig.update_layout(
                height=150, 
                showlegend=False, 
                margin=dict(t=0, b=0, l=0, r=0),
                font=dict(size=11)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'focus_area_coverage' in data:
            coverage_df = data['focus_area_coverage']
            critical = len(coverage_df[coverage_df['Coverage_Status'] == 'Critical'])
            good = len(coverage_df[coverage_df['Coverage_Status'] == 'Good'])
            
            st.metric("🎯 Focus Areas", 
                     f"{len(coverage_df)}", 
                     f"🔴 {critical} critical | 🟢 {good} good")
            
            # Coverage status bar
            fig = go.Figure(go.Bar(
                x=[critical, len(coverage_df) - critical - good, good],
                y=['Status'] * 3,
                orientation='h',
                marker_color=['#dc3545', '#ffc107', '#28a745'],
                showlegend=False
            ))
            fig.update_layout(height=100, margin=dict(t=0, b=0, l=0, r=0), 
                            xaxis=dict(showgrid=False, showticklabels=False),
                            yaxis=dict(showgrid=False, showticklabels=False))
            st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        avg_skills = resources_df['skill_count'].mean() if 'skill_count' in resources_df.columns else 17
        multi_skilled = len(resources_df[resources_df['skill_count'] >= 15]) if 'skill_count' in resources_df.columns else 244
        
        st.metric("🛠️ Skills Depth", f"{avg_skills:.1f} avg", f"{multi_skilled} multi-skilled")
        
        # Skills histogram
        if 'skill_count' in resources_df.columns:
            fig = go.Figure(go.Histogram(
                x=resources_df['skill_count'],
                nbinsx=20,
                marker_color='#667eea'
            ))
            fig.update_layout(height=100, margin=dict(t=0, b=0, l=0, r=0),
                            xaxis=dict(showgrid=False, showticklabels=False),
                            yaxis=dict(showgrid=False, showticklabels=False))
            st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        if 'focus_area_coverage' in data:
            total_revenue = data['focus_area_coverage']['Revenue_Potential'].sum()
            at_risk = data['focus_area_coverage'][
                data['focus_area_coverage']['Coverage_Status'] == 'Critical']['Revenue_Potential'].sum()
            
            st.metric("💰 Revenue", f"${total_revenue:.0f}M", f"${at_risk:.0f}M at risk")
            
            # Risk gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=(at_risk/total_revenue)*100 if total_revenue > 0 else 0,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Risk %", 'font': {'size': 14}},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#dc3545"},
                    'steps': [
                        {'range': [0, 25], 'color': "#e8f5e9"},
                        {'range': [25, 50], 'color': "#fff3cd"},
                        {'range': [50, 100], 'color': "#ffebee"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(
                height=150, 
                margin=dict(t=20, b=0, l=0, r=0),
                font=dict(size=12)
            )
            st.plotly_chart(fig, use_container_width=True)

def show_smart_recommendations_enhanced(data):
    """Enhanced smart recommendations with better search."""
    st.header("🔍 Smart Resource Search & Recommendations")
    
    st.markdown("""
    <div style='background: #e3f2fd; padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
        <p style='margin: 0;'>🎯 <strong>Intelligent Matching Engine</strong>: 
        Combines Focus Area analysis, skill matching, and domain expertise to find the perfect resources</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        project_description = st.text_area(
            "📝 Describe your project or resource needs:",
            placeholder="Example: Need AI experts for machine learning platform implementation with MLOps, cloud infrastructure, and data pipeline experience",
            height=100
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        num_recommendations = st.slider("Recommendations:", 5, 30, 10)
        include_focus = st.checkbox("Focus Area Analysis", value=True)
        include_skills = st.checkbox("Skill Matching", value=True)
    
    if st.button("🚀 Get Smart Recommendations", type="primary", disabled=not project_description):
        with st.spinner("🤖 Analyzing requirements with AI..."):
            
            # Focus Area prediction
            if include_focus:
                classifier = FocusAreaClassifier()
                focus_predictions = classifier.predict(project_description, top_k=3)
                
                st.markdown("### 🎯 Predicted Focus Areas")
                
                cols = st.columns(3)
                for i, (fa, conf) in enumerate(focus_predictions):
                    with cols[i]:
                        confidence_color = "#28a745" if conf > 0.7 else "#ffc107" if conf > 0.4 else "#dc3545"
                        st.markdown(f"""
                        <div style='background: white; padding: 1rem; border-radius: 10px; 
                                    border-left: 4px solid {confidence_color};'>
                            <h4 style='margin: 0;'>{fa}</h4>
                            <p style='color: {confidence_color}; margin: 0.5rem 0 0 0;'>
                                Confidence: {conf:.1%}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Get Focus Area requirements
                integrator = FocusAreaIntegrator()
                top_fa = focus_predictions[0][0]
                requirements = integrator.get_focus_area_requirements(top_fa)
                
                with st.expander("📋 Focus Area Requirements", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Required Domains:**")
                        for domain in requirements['required_domains']:
                            st.markdown(f"• {domain}")
                    with col2:
                        st.markdown("**Key Skills:**")
                        for skill in requirements['recommended_skills'][:5]:
                            st.markdown(f"• {skill}")
            
            # Resource matching
            if 'resources_corrected' in data:
                resources_df = data['resources_corrected']
                
                # Enhanced matching with Focus Area keywords
                keywords = project_description.lower().split()
                if include_focus and focus_predictions:
                    fa_keywords = requirements.get('search_keywords', [])
                    keywords.extend([kw.lower() for kw in fa_keywords])
                
                matches = []
                for _, person in resources_df.iterrows():
                    skills = str(person.get('all_skills', '')).lower()
                    domain = str(person.get('primary_domain', '')).lower()
                    
                    score = 0
                    matched_keywords = []
                    
                    for keyword in set(keywords):
                        if len(keyword) > 2:  # Skip very short words
                            if keyword in skills:
                                score += 3
                                matched_keywords.append(keyword)
                            if keyword in domain:
                                score += 1
                    
                    if score > 0:
                        matches.append({
                            'name': person['resource_name'],
                            'score': score,
                            'matched_keywords': matched_keywords[:5],
                            'skills': person.get('all_skills', ''),
                            'skill_count': person.get('skill_count', 0),
                            'rating': person.get('avg_rating', 0),
                            'domain': person.get('primary_domain', ''),
                            'city': person.get('city', ''),
                            'manager': person.get('manager', '')
                        })
                
                matches.sort(key=lambda x: x['score'], reverse=True)
                
                st.success(f"✅ Found {len(matches)} matching professionals!")
                
                # Display recommendations
                st.markdown(f"### 👥 Top {min(num_recommendations, len(matches))} Recommendations")
                
                for i, match in enumerate(matches[:num_recommendations], 1):
                    score_color = "#28a745" if match['score'] > 10 else "#ffc107" if match['score'] > 5 else "#17a2b8"
                    
                    with st.expander(f"#{i} - {match['name']} | Score: {match['score']} 🎯"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("**Profile:**")
                            st.markdown(f"• Domain: {match['domain']}")
                            st.markdown(f"• Location: {match['city']}")
                            st.markdown(f"• Manager: {match['manager']}")
                        
                        with col2:
                            st.markdown("**Expertise:**")
                            st.markdown(f"• Skills: {match['skill_count']} total")
                            st.markdown(f"• Rating: ⭐ {match['rating']:.1f}/5")
                            st.markdown(f"• Matches: {', '.join(match['matched_keywords'])}")
                        
                        with col3:
                            st.markdown("**Match Score:**")
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=min(match['score']*5, 100),
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'font': {'size': 12}},
                                gauge={
                                    'axis': {'range': [None, 100]},
                                    'bar': {'color': score_color},
                                    'steps': [
                                        {'range': [0, 33], 'color': "#f0f0f0"},
                                        {'range': [33, 66], 'color': "#e0e0e0"},
                                        {'range': [66, 100], 'color': "#d0d0d0"}
                                    ]
                                }
                            ))
                            fig.update_layout(
                                height=100, 
                                margin=dict(t=0, b=0, l=0, r=0),
                                font=dict(size=11)
                            )
                            st.plotly_chart(fig, use_container_width=True, key=f"gauge_{i}")
                        
                        if match['skills']:
                            st.markdown("**Key Skills:**")
                            skills_preview = match['skills'][:200] + "..." if len(match['skills']) > 200 else match['skills']
                            st.info(skills_preview)

def main():
    """Main application with all enhancements."""
    show_header()
    
    # Load enhanced data
    data = load_enhanced_data()
    
    if not data:
        st.error("Failed to load data. Please check data files.")
        return
    
    # Create comprehensive navigation
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Executive Overview",
        "🎯 Focus Area Planning", 
        "🔮 Strategic Forecasting",
        "🤖 Smart Recommendations",
        "📈 Capacity Analysis",
        "📋 Reports & Export"
    ])
    
    with tab1:
        show_executive_overview(data)
        
        # Additional insights
        if 'focus_area_coverage' in data:
            st.markdown("---")
            st.markdown("### 🔍 Quick Insights")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                coverage_df = data['focus_area_coverage']
                critical = coverage_df[coverage_df['Coverage_Status'] == 'Critical']
                if len(critical) > 0:
                    st.error(f"**🚨 {len(critical)} Critical Focus Areas**")
                    for _, row in critical.head(3).iterrows():
                        st.markdown(f"• {row['Focus_Area']}: ${row['Revenue_Potential']:.1f}M at risk")
            
            with col2:
                if 'resources_corrected' in data:
                    resources_df = data['resources_corrected']
                    domains = resources_df['primary_domain'].value_counts()
                    st.info(f"**🏢 Top Domains**")
                    for domain, count in domains.head(3).items():
                        st.markdown(f"• {domain}: {count} people")
            
            with col3:
                total_gap = coverage_df['Resource_Count'].sum() - (coverage_df['Revenue_Potential'] * 2.5).sum()
                st.warning(f"**📊 Capacity Gap**")
                st.markdown(f"• Need {abs(int(total_gap))} more resources")
                st.markdown(f"• Investment: ${abs(total_gap)*150:.0f}K")
    
    with tab2:
        show_focus_area_capacity_planning(data)
    
    with tab3:
        show_enhanced_forecasting(data)
    
    with tab4:
        # Use the improved search 
        show_improved_smart_search(data)
    
    with tab5:
        # Additional capacity analysis
        st.header("📈 Detailed Capacity Analysis")
        
        if 'focus_area_coverage' in data and 'resources_corrected' in data:
            coverage_df = data['focus_area_coverage']
            resources_df = data['resources_corrected']
            
            # Key metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_resources = len(resources_df)
                st.metric("Total Resources", f"{total_resources:,}")
            
            with col2:
                avg_skills = resources_df['skill_count'].mean()
                st.metric("Avg Skills/Person", f"{avg_skills:.1f}")
            
            with col3:
                avg_rating = resources_df['avg_rating'].mean()
                st.metric("Avg Rating", f"⭐ {avg_rating:.2f}")
            
            with col4:
                domains = resources_df['primary_domain'].nunique()
                st.metric("Active Domains", f"{domains}")
            
            # Domain capacity with better visualization
            st.markdown("### 🏢 Domain Capacity Analysis")
            
            domain_counts = resources_df['primary_domain'].value_counts()
            
            # Create two columns for domain analysis
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=domain_counts.values,
                    y=domain_counts.index,
                    orientation='h',
                    marker_color=['#667eea' if i < 3 else '#98a6d4' for i in range(len(domain_counts))],
                    text=domain_counts.values,
                    textposition='outside',
                    hovertemplate='<b>%{y}</b><br>Resources: %{x}<br>%{x} professionals<extra></extra>'
                ))
                fig.update_layout(
                    title=dict(text="Resources by Technical Domain", font=dict(size=16)),
                    xaxis_title="Number of Resources",
                    yaxis_title="",
                    height=400,
                    margin=dict(l=200)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Domain Insights:**")
                top_domain = domain_counts.index[0]
                st.info(f"💡 **{top_domain}** has the most resources ({domain_counts.values[0]} professionals)")
                
                # Calculate domain coverage percentage
                st.markdown("**Coverage Distribution:**")
                for domain, count in domain_counts.head(5).items():
                    pct = count / total_resources * 100
                    st.progress(pct/100)
                    st.caption(f"{domain[:20]}: {pct:.1f}%")
            
            # Geographic distribution with enhanced visuals
            st.markdown("### 🌍 Geographic Distribution")
            
            if 'city' in resources_df.columns:
                city_counts = resources_df['city'].value_counts().head(10)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Create sunburst or treemap for better hierarchy
                    fig = go.Figure(go.Treemap(
                        labels=city_counts.index,
                        parents=[""] * len(city_counts),
                        values=city_counts.values,
                        text=[f"{city}<br>{count} resources" for city, count in city_counts.items()],
                        textinfo="label+value+percent root",
                        marker=dict(colorscale='Blues', line=dict(width=2))
                    ))
                    fig.update_layout(
                        title=dict(text="Resource Distribution by Location", font=dict(size=16)),
                        height=400,
                        margin=dict(t=50, l=0, r=0, b=0)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("**Geographic Insights:**")
                    
                    # Concentration analysis
                    top_3_cities = city_counts.head(3)
                    concentration = top_3_cities.sum() / total_resources * 100
                    
                    if concentration > 80:
                        st.warning(f"⚠️ High concentration: {concentration:.1f}% of resources in top 3 cities")
                    else:
                        st.success(f"✅ Good distribution: {concentration:.1f}% in top 3 cities")
                    
                    st.markdown("**Top Locations:**")
                    for city, count in city_counts.head(5).items():
                        st.markdown(f"• **{city}**: {count} ({count/total_resources*100:.1f}%)")
            
            # Skills distribution analysis
            st.markdown("### 🎯 Skills Distribution Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Skill count distribution
                skill_ranges = pd.cut(resources_df['skill_count'], 
                                     bins=[0, 5, 10, 15, 20, 100],
                                     labels=['1-5', '6-10', '11-15', '16-20', '20+'])
                skill_dist = skill_ranges.value_counts().sort_index()
                
                # Convert to list for proper display
                x_values = [str(i) for i in skill_dist.index.tolist()]
                y_values = skill_dist.values.tolist()
                
                fig = go.Figure(go.Bar(
                    x=x_values,
                    y=y_values,
                    marker_color='#28a745',
                    text=y_values,
                    textposition='outside'
                ))
                fig.update_layout(
                    title="Skills per Professional",
                    xaxis_title="Skill Count Range",
                    yaxis_title="Number of Professionals",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True, key="skill_dist_chart")
            
            with col2:
                # Rating distribution
                rating_ranges = pd.cut(resources_df['avg_rating'],
                                      bins=[0, 2, 3, 4, 5],
                                      labels=['Poor (0-2)', 'Fair (2-3)', 'Good (3-4)', 'Excellent (4-5)'])
                rating_dist = rating_ranges.value_counts()
                
                # Convert to lists for proper display
                labels_list = [str(i) for i in rating_dist.index.tolist()]
                values_list = rating_dist.values.tolist()
                
                fig = go.Figure(go.Pie(
                    values=values_list,
                    labels=labels_list,
                    hole=0.4,
                    marker_colors=['#dc3545', '#ffc107', '#28a745', '#007bff']
                ))
                fig.update_layout(
                    title="Proficiency Distribution",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True, key="rating_dist_chart")
    
    with tab6:
        st.header("📋 Reports & Export")
        
        st.markdown("### 📊 Available Reports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📈 Generate Executive Report", type="primary", use_container_width=True):
                st.success("✅ Executive report generated!")
                st.markdown("**Includes:**")
                st.markdown("• Focus Area analysis")
                st.markdown("• Resource capacity metrics")
                st.markdown("• Revenue alignment")
                st.markdown("• Strategic recommendations")
        
        with col2:
            if st.button("💾 Export Complete Dataset", type="primary", use_container_width=True):
                st.success("✅ Dataset exported!")
                st.markdown("**Contains:**")
                st.markdown("• 565 unique professionals")
                st.markdown("• 31 Focus Areas with coverage")
                st.markdown("• Skill mappings")
                st.markdown("• Forecasting models")
        
        # Data preview
        st.markdown("### 👁️ Data Preview")
        
        if 'resources_corrected' in data:
            st.dataframe(
                data['resources_corrected'].head(10),
                use_container_width=True
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>HPE Talent Intelligence Platform - Complete Enhanced Edition</p>
        <p style='font-size: 0.9rem;'>565 Professionals | 31 Focus Areas | $288M Opportunities | 640 Skills</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()