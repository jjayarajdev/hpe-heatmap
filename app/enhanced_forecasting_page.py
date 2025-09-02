"""
Enhanced Resource Forecasting & Capacity Planning Page
Professional, actionable, and insightful workforce analytics
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
from src.enhanced_forecasting import create_capacity_dashboard, EnhancedForecaster

def show_enhanced_forecasting(data):
    """Enhanced forecasting with comprehensive analytics."""
    
    # Professional header
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2.5rem; border-radius: 15px; margin: 1rem 0; color: white;'>
        <h2 style='margin: 0; font-size: 2.5rem;'>🔮 Strategic Workforce Intelligence</h2>
        <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.95;'>
            Advanced Capacity Planning & Predictive Analytics
        </p>
        <div style='display: flex; gap: 2rem; margin-top: 1.5rem;'>
            <div style='flex: 1;'>
                <h3 style='margin: 0; font-size: 2rem;'>565</h3>
                <p style='margin: 0; opacity: 0.9;'>Total Professionals</p>
            </div>
            <div style='flex: 1;'>
                <h3 style='margin: 0; font-size: 2rem;'>31</h3>
                <p style='margin: 0; opacity: 0.9;'>Focus Areas</p>
            </div>
            <div style='flex: 1;'>
                <h3 style='margin: 0; font-size: 2rem;'>$288M</h3>
                <p style='margin: 0; opacity: 0.9;'>Revenue Opportunity</p>
            </div>
            <div style='flex: 1;'>
                <h3 style='margin: 0; font-size: 2rem;'>640</h3>
                <p style='margin: 0; opacity: 0.9;'>Unique Skills</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Check data availability
    if 'resources_corrected' not in data:
        st.error("❌ Corrected resource data not available for enhanced forecasting.")
        return
    
    resources_df = data['resources_corrected']
    services_df = data.get('service_skillset_Services_to_skillsets_Mapping_Master_v5_clean_clean')
    
    # Generate comprehensive dashboard
    with st.spinner("🤖 Generating strategic insights..."):
        dashboard_data = create_capacity_dashboard(resources_df, services_df)
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Executive Summary",
        "🎯 Skill Gap Analysis", 
        "📈 Demand Forecasting",
        "🔄 Scenario Planning",
        "💡 Recommendations",
        "🗺️ Geographic Intel"
    ])
    
    with tab1:
        show_executive_summary(dashboard_data)
    
    with tab2:
        show_skill_gap_analysis(dashboard_data)
    
    with tab3:
        show_demand_forecasting(dashboard_data)
    
    with tab4:
        show_scenario_planning(dashboard_data, resources_df)
    
    with tab5:
        show_recommendations(dashboard_data)
    
    with tab6:
        show_geographic_intelligence(dashboard_data)

def show_executive_summary(dashboard_data):
    """Executive summary with key metrics and insights."""
    st.markdown("### 📊 Executive Capacity Dashboard")
    
    metrics = dashboard_data['metrics']
    
    # Key metrics in cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total = metrics['total_resources']
        st.metric(
            "👥 Total Workforce",
            f"{total}",
            help="Verified unique professionals"
        )
        
        # Skill distribution mini chart
        if 'skill_distribution' in metrics:
            dist = metrics['skill_distribution']
            if 'percentages' in dist:
                fig = go.Figure(go.Pie(
                    values=list(dist['percentages'].values()),
                    labels=['Beginner', 'Intermediate', 'Advanced', 'Expert'],
                    hole=0.6,
                    marker_colors=['#ffd700', '#87ceeb', '#98fb98', '#9370db']
                ))
                fig.update_layout(height=150, showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        unique_skills = metrics.get('unique_skills', 0)
        st.metric(
            "🛠️ Unique Skills",
            f"{unique_skills}",
            help="Total distinct capabilities"
        )
        
        # Top domains mini bar
        if 'domain_analysis' in metrics:
            domains = metrics['domain_analysis']
            top_3 = sorted(domains.items(), key=lambda x: x[1]['count'], reverse=True)[:3]
            for domain, info in top_3:
                st.progress(info['percentage']/100, text=f"{domain[:15]}: {info['count']}")
    
    with col3:
        if 'utilization' in metrics:
            util = metrics['utilization']
            st.metric(
                "📈 Utilization Rate",
                f"{util['estimated_utilization']}%",
                f"{util['available_immediately']} available now",
                help="Current resource utilization"
            )
        
        # Utilization gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=metrics['utilization']['estimated_utilization'],
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#667eea"},
                'steps': [
                    {'range': [0, 50], 'color': "#f0f0f0"},
                    {'range': [50, 75], 'color': "#e0e0e0"},
                    {'range': [75, 100], 'color': "#d0d0d0"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=150, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        if 'geographic_analysis' in metrics:
            geo = metrics['geographic_analysis']
            st.metric(
                "🌍 Locations",
                f"{geo['total_locations']}",
                f"{geo['concentration_risk']} concentration risk",
                help="Geographic distribution"
            )
        
        # Risk indicator
        risk_color = {
            'Low': '🟢',
            'Medium': '🟡', 
            'High': '🔴'
        }.get(geo.get('concentration_risk', 'Medium'), '🟡')
        
        st.markdown(f"""
        <div style='text-align: center; font-size: 2rem;'>
            {risk_color}
        </div>
        """, unsafe_allow_html=True)
    
    # Insights section
    st.markdown("### 🔍 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Critical gaps
        if 'skill_gaps' in metrics:
            critical = [g for g in metrics['skill_gaps'] if g['severity'] == 'Critical']
            if critical:
                st.error(f"🚨 **{len(critical)} Critical Skill Gaps Detected**")
                for gap in critical[:3]:
                    st.markdown(f"• **{gap['skill']}**: Need {gap['gap']} more (have {gap['actual']}/{gap['required']})")
        
        # Growth opportunities
        if 'growth_areas' in metrics:
            high_growth = [g for g in metrics['growth_areas'] if g['priority'] == 'High']
            if high_growth:
                st.info(f"📈 **{len(high_growth)} High-Growth Areas Identified**")
                for area in high_growth[:3]:
                    st.markdown(f"• **{area['technology']}**: Need {area['growth_required']} resources in {area['timeline']}")
    
    with col2:
        # Domain balance
        if 'domain_analysis' in metrics:
            domains = metrics['domain_analysis']
            
            # Create treemap
            labels = []
            values = []
            colors = []
            
            for domain, info in domains.items():
                labels.append(f"{domain}<br>{info['count']} people<br>{info['avg_skills']:.1f} avg skills")
                values.append(info['count'])
                colors.append(info['avg_rating'])
            
            fig = go.Figure(go.Treemap(
                labels=labels,
                values=values,
                marker=dict(
                    colorscale='Viridis',
                    cmid=3,
                    colorbar=dict(title="Avg Rating")
                ),
                text=[f"{v}" for v in values],
                textposition="middle center"
            ))
            fig.update_layout(height=300, title="Domain Distribution & Expertise")
            st.plotly_chart(fig, use_container_width=True)

def show_skill_gap_analysis(dashboard_data):
    """Detailed skill gap analysis."""
    st.markdown("### 🎯 Critical Skill Gap Analysis")
    
    metrics = dashboard_data['metrics']
    
    if 'skill_gaps' not in metrics:
        st.warning("Skill gap data not available")
        return
    
    gaps = metrics['skill_gaps']
    
    if not gaps:
        st.success("✅ No critical skill gaps identified!")
        return
    
    # Create comprehensive gap visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Skill Gap Severity", "Gap Size Distribution", 
                       "Coverage Percentage", "Hiring Priority Matrix"),
        specs=[[{"type": "bar"}, {"type": "pie"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # Gap severity bar chart
    gap_df = pd.DataFrame(gaps)
    fig.add_trace(
        go.Bar(
            x=gap_df['skill'].tolist(),
            y=gap_df['gap'].tolist(),
            marker_color=['#dc3545' if s == 'Critical' else '#ffc107' if s == 'High' else '#28a745' 
                         for s in gap_df['severity']],
            text=gap_df['gap'].tolist(),
            textposition='outside',
            name="Gap Size"
        ),
        row=1, col=1
    )
    
    # Severity distribution pie
    severity_counts = gap_df['severity'].value_counts()
    fig.add_trace(
        go.Pie(
            labels=severity_counts.index.tolist(),
            values=severity_counts.values.tolist(),
            marker_colors=['#dc3545', '#ffc107', '#28a745'],
            hole=0.4
        ),
        row=1, col=2
    )
    
    # Coverage percentage
    gap_df['coverage_pct'] = (gap_df['actual'] / gap_df['required'] * 100).round(1)
    fig.add_trace(
        go.Bar(
            x=gap_df['skill'].tolist(),
            y=gap_df['coverage_pct'].tolist(),
            marker_color='#667eea',
            text=[f"{pct:.1f}%" for pct in gap_df['coverage_pct']],
            textposition='outside',
            name="Coverage %"
        ),
        row=2, col=1
    )
    
    # Priority matrix (Impact vs Effort)
    gap_df['impact'] = gap_df['gap'] * (4 - gap_df['severity'].map({'Critical': 1, 'High': 2, 'Medium': 3}))
    gap_df['effort'] = gap_df['gap'] * 50000  # Estimated cost per resource
    
    fig.add_trace(
        go.Scatter(
            x=gap_df['effort'].tolist(),
            y=gap_df['impact'].tolist(),
            mode='markers+text',
            marker=dict(
                size=(gap_df['gap']*2).tolist(),
                color=gap_df['gap'].tolist(),
                colorscale='Reds',
                showscale=True
            ),
            text=gap_df['skill'].tolist(),
            textposition="top center",
            name="Skills"
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False)
    fig.update_xaxes(title_text="Skill", row=1, col=1)
    fig.update_xaxes(title_text="Skill", row=2, col=1)
    fig.update_xaxes(title_text="Hiring Cost (Est.)", row=2, col=2)
    fig.update_yaxes(title_text="Gap Size", row=1, col=1)
    fig.update_yaxes(title_text="Coverage %", row=2, col=1)
    fig.update_yaxes(title_text="Business Impact", row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True, key="skill_gap_analysis_chart")
    
    # Action plan table
    st.markdown("### 📋 Skill Gap Action Plan")
    
    action_df = gap_df[['skill', 'required', 'actual', 'gap', 'severity', 'coverage_pct']].copy()
    action_df['Action'] = action_df.apply(
        lambda x: f"Hire {x['gap']} immediately" if x['severity'] == 'Critical'
        else f"Hire {x['gap']} within 3 months" if x['severity'] == 'High'
        else f"Plan for {x['gap']} hires", axis=1
    )
    action_df['Est. Cost'] = action_df['gap'] * 150000  # Average salary estimate
    action_df['Est. Cost'] = action_df['Est. Cost'].apply(lambda x: f"${x:,.0f}")
    
    st.dataframe(
        action_df.style.background_gradient(subset=['coverage_pct'], cmap='RdYlGn'),
        use_container_width=True
    )

def show_demand_forecasting(dashboard_data):
    """Advanced demand forecasting."""
    st.markdown("### 📈 Predictive Demand Forecasting")
    
    forecast = dashboard_data['forecast']
    
    # Forecast controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        timeline = st.selectbox("Forecast Period:", ["6 months", "12 months", "24 months"], index=1)
    
    with col2:
        growth_scenario = st.select_slider(
            "Growth Scenario:",
            options=["Conservative (10%)", "Baseline (15%)", "Aggressive (25%)"],
            value="Baseline (15%)"
        )
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Recalculate Forecast", type="primary"):
            st.rerun()
    
    # Main forecast chart
    projections = forecast['monthly_projections']
    proj_df = pd.DataFrame(projections)
    
    fig = go.Figure()
    
    # Current baseline
    fig.add_trace(go.Scatter(
        x=proj_df['month'].tolist(),
        y=[forecast['current_resources']] * len(proj_df),
        mode='lines',
        line=dict(dash='dash', color='gray'),
        name='Current Capacity'
    ))
    
    # Projected need
    fig.add_trace(go.Scatter(
        x=proj_df['month'].tolist(),
        y=proj_df['projected_need'].tolist(),
        mode='lines+markers',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8),
        name='Projected Need',
        fill='tonexty',
        fillcolor='rgba(102, 126, 234, 0.1)'
    ))
    
    # Add annotations for key milestones
    for i in [3, 6, 9, 12]:
        if i <= len(proj_df):
            row = proj_df.iloc[i-1]
            fig.add_annotation(
                x=row['month'],
                y=row['projected_need'],
                text=f"+{row['additional_required']}",
                showarrow=True,
                arrowhead=2
            )
    
    fig.update_layout(
        title="Resource Demand Projection",
        xaxis_title="Months",
        yaxis_title="Total Resources Required",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Domain-specific forecasts
    st.markdown("### 🏢 Domain-Specific Requirements")
    
    if 'domain_forecasts' in forecast:
        domain_data = []
        for domain, data in forecast['domain_forecasts'].items():
            domain_data.append({
                'Domain': domain,
                'Current': data['current'],
                'Projected': data['projected'],
                'Gap': data['gap'],
                'Growth %': f"{(data['gap']/data['current']*100):.1f}%"
            })
        
        domain_df = pd.DataFrame(domain_data).sort_values('Gap', ascending=False)
        
        # Create grouped bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Current',
            x=domain_df['Domain'].tolist(),
            y=domain_df['Current'].tolist(),
            marker_color='#98fb98'
        ))
        
        fig.add_trace(go.Bar(
            name='Additional Needed',
            x=domain_df['Domain'].tolist(),
            y=domain_df['Gap'].tolist(),
            marker_color='#ffd700'
        ))
        
        fig.update_layout(
            barmode='stack',
            title="Domain Hiring Requirements",
            xaxis_title="Domain",
            yaxis_title="Number of Resources",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_scenario_planning(dashboard_data, resources_df):
    """Interactive scenario planning."""
    st.markdown("### 🔄 Strategic Scenario Planning")
    
    st.markdown("""
    <div style='background: #f0f8ff; padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
        <p style='margin: 0;'>🎯 <strong>Interactive What-If Analysis</strong>: 
        Adjust parameters to see how different scenarios impact your workforce capacity</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Scenario parameters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        growth_rate = st.slider("Annual Growth %", 5, 30, 15)
    
    with col2:
        attrition_rate = st.slider("Attrition Rate %", 5, 20, 10)
    
    with col3:
        months = st.selectbox("Timeline (months)", [6, 12, 18, 24], index=1)
    
    with col4:
        hiring_rate = st.slider("Monthly Hiring", 0, 20, 5)
    
    # Calculate scenario
    forecaster = EnhancedForecaster(resources_df)
    
    scenarios = [
        {
            'name': 'Your Scenario',
            'growth_rate': growth_rate/100,
            'attrition_rate': attrition_rate/100,
            'months': months
        },
        {
            'name': 'Best Case',
            'growth_rate': 0.25,
            'attrition_rate': 0.05,
            'months': months
        },
        {
            'name': 'Worst Case',
            'growth_rate': 0.05,
            'attrition_rate': 0.20,
            'months': months
        }
    ]
    
    scenario_results = forecaster.create_scenario_analysis(scenarios)
    
    # Visualization
    fig = go.Figure()
    
    colors = {'Your Scenario': '#667eea', 'Best Case': '#28a745', 'Worst Case': '#dc3545'}
    
    for name, result in scenario_results.items():
        projections = result['monthly_projections']
        months_list = [p['month'] for p in projections]
        resources_list = [p['resources'] for p in projections]
        
        fig.add_trace(go.Scatter(
            x=months_list,
            y=resources_list,
            mode='lines+markers',
            name=name,
            line=dict(color=colors.get(name, '#gray'), width=3 if name == 'Your Scenario' else 2)
        ))
    
    # Add current baseline
    fig.add_hline(
        y=len(resources_df),
        line_dash="dash",
        line_color="gray",
        annotation_text="Current Capacity"
    )
    
    fig.update_layout(
        title="Scenario Comparison",
        xaxis_title="Months",
        yaxis_title="Total Resources",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Scenario outcomes
    st.markdown("### 📊 Scenario Outcomes")
    
    outcomes = []
    for name, result in scenario_results.items():
        outcomes.append({
            'Scenario': name,
            'Starting': result['initial'],
            'Ending': result['final'],
            'Net Change': result['net_change'],
            'Change %': f"{(result['net_change']/result['initial']*100):.1f}%"
        })
    
    outcomes_df = pd.DataFrame(outcomes)
    
    # Style the dataframe
    def style_change(val):
        if isinstance(val, str):
            return ''
        return 'color: green' if val > 0 else 'color: red' if val < 0 else ''
    
    st.dataframe(
        outcomes_df.style.applymap(style_change, subset=['Net Change']),
        use_container_width=True
    )

def show_recommendations(dashboard_data):
    """Actionable recommendations."""
    st.markdown("### 💡 Strategic Recommendations")
    
    recommendations = dashboard_data['recommendations']
    
    if not recommendations:
        st.info("No critical recommendations at this time")
        return
    
    # Group by priority
    priority_groups = {}
    for rec in recommendations:
        priority = rec['priority']
        if priority not in priority_groups:
            priority_groups[priority] = []
        priority_groups[priority].append(rec)
    
    # Display recommendations
    for priority in ['Critical', 'High', 'Medium']:
        if priority in priority_groups:
            st.markdown(f"#### {get_priority_icon(priority)} {priority} Priority Actions")
            
            for rec in priority_groups[priority]:
                with st.expander(f"{rec['title']} ({rec['category']})", expanded=(priority == 'Critical')):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Situation:** {rec['description']}")
                        st.markdown("**Action Items:**")
                        for item in rec['action_items']:
                            st.markdown(f"• {item}")
                    
                    with col2:
                        st.metric("Impact", rec['impact'])
                        st.metric("Timeline", rec['timeline'])
                    
                    # Add implementation button
                    if st.button(f"📋 Create Action Plan", key=f"plan_{rec['title']}"):
                        st.success(f"✅ Action plan created for: {rec['title']}")

def show_geographic_intelligence(dashboard_data):
    """Geographic distribution analysis."""
    st.markdown("### 🗺️ Geographic Intelligence")
    
    metrics = dashboard_data['metrics']
    
    if 'geographic_analysis' not in metrics:
        st.warning("Geographic data not available")
        return
    
    geo = metrics['geographic_analysis']
    
    # Risk assessment
    risk_level = geo['concentration_risk']
    risk_color = {'Low': 'success', 'Medium': 'warning', 'High': 'error'}.get(risk_level, 'info')
    
    getattr(st, risk_color)(f"**Geographic Concentration Risk: {risk_level}** - "
                           f"{geo['top_city_percentage']:.1f}% of resources in top location")
    
    # Location distribution
    if 'top_locations' in geo:
        locations = geo['top_locations']
        
        # Create map-like visualization
        fig = go.Figure()
        
        # Convert to dataframe for easier plotting
        loc_df = pd.DataFrame([
            {'Location': loc, 'Count': count}
            for loc, count in locations.items()
        ])
        
        # Bubble chart
        fig.add_trace(go.Scatter(
            x=list(range(len(loc_df))),
            y=[1] * len(loc_df),
            mode='markers+text',
            marker=dict(
                size=loc_df['Count']*2,
                color=loc_df['Count'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Resources")
            ),
            text=loc_df['Location'],
            textposition="top center",
            hovertemplate='<b>%{text}</b><br>Resources: %{marker.color}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Resource Distribution by Location",
            showlegend=False,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        st.markdown("### 📍 Location Details")
        
        loc_df['Percentage'] = (loc_df['Count'] / loc_df['Count'].sum() * 100).round(1)
        loc_df['Risk'] = loc_df['Percentage'].apply(
            lambda x: '🔴 High' if x > 30 else '🟡 Medium' if x > 15 else '🟢 Low'
        )
        
        st.dataframe(loc_df, use_container_width=True)
        
        # Diversity score
        if 'geographic_diversity_score' in geo:
            st.metric(
                "Geographic Diversity Score",
                f"{geo['geographic_diversity_score']:.1f}/100",
                help="Higher scores indicate better geographic distribution"
            )

def get_priority_icon(priority):
    """Get icon for priority level."""
    return {
        'Critical': '🚨',
        'High': '⚠️',
        'Medium': '📌'
    }.get(priority, '📌')

# Example integration into main app
if __name__ == "__main__":
    st.set_page_config(page_title="Enhanced Forecasting", page_icon="🔮", layout="wide")
    
    # Load sample data
    @st.cache_data
    def load_data():
        # This would load your actual data
        return {
            'resources_corrected': pd.DataFrame({
                'resource_name': [f"Person_{i}" for i in range(565)],
                'skill_count': np.random.randint(5, 30, 565),
                'avg_rating': np.random.uniform(2, 5, 565),
                'primary_domain': np.random.choice(['Infrastructure', 'Data', 'Security', 'Cloud', 'Apps'], 565),
                'city': np.random.choice(['Bangalore', 'Sofia', 'Austin', 'London', 'Singapore'], 565),
                'all_skills': ['Python; Java; AWS' for _ in range(565)]
            })
        }
    
    data = load_data()
    show_enhanced_forecasting(data)