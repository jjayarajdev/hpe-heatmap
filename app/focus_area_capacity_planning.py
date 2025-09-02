"""
Enhanced Focus Area Capacity Planning
Smart, data-driven capacity planning based on actual Focus Areas and revenue
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def show_focus_area_capacity_planning(data):
    """Intelligent Focus Area-based capacity planning."""
    
    # Professional header with context
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 15px; margin: 1rem 0; color: white;'>
        <h2 style='margin: 0; font-size: 2rem;'>🎯 Focus Area Strategic Capacity Planning</h2>
        <p style='margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;'>
            Revenue-driven workforce planning across 31 Focus Areas
        </p>
        <div style='display: flex; gap: 2rem; margin-top: 1rem;'>
            <div>
                <span style='font-size: 1.5rem; font-weight: bold;'>$288M</span>
                <span style='opacity: 0.8;'> Total Opportunity</span>
            </div>
            <div>
                <span style='font-size: 1.5rem; font-weight: bold;'>31</span>
                <span style='opacity: 0.8;'> Focus Areas</span>
            </div>
            <div>
                <span style='font-size: 1.5rem; font-weight: bold;'>565</span>
                <span style='opacity: 0.8;'> Available Resources</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Check for Focus Area data
    if 'focus_area_coverage' not in data:
        st.error("❌ Focus Area data not available. Please run Focus Area integration first.")
        return
    
    coverage_df = data['focus_area_coverage']
    resources_df = data.get('resources_corrected', pd.DataFrame())
    
    # Create analysis tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💰 Revenue-Based Planning",
        "📊 Current vs Required", 
        "🎯 Focus Area Deep Dive",
        "🔮 Predictive Modeling",
        "📋 Action Plan"
    ])
    
    with tab1:
        show_revenue_based_planning(coverage_df, resources_df)
    
    with tab2:
        show_current_vs_required(coverage_df)
    
    with tab3:
        show_focus_area_deep_dive(coverage_df, data)
    
    with tab4:
        show_predictive_modeling(coverage_df)
    
    with tab5:
        show_action_plan(coverage_df)

def show_revenue_based_planning(coverage_df, resources_df):
    """Revenue-driven capacity planning."""
    st.markdown("### 💰 Revenue-Driven Resource Allocation")
    
    # Calculate resources needed per million dollar of revenue
    resources_per_million = 2.5  # Industry benchmark
    
    # Add calculated fields
    coverage_df['Resources_Needed'] = (coverage_df['Revenue_Potential'] * resources_per_million).round().astype(int)
    coverage_df['Gap'] = coverage_df['Resources_Needed'] - coverage_df['Resource_Count']
    coverage_df['Gap_Percentage'] = ((coverage_df['Gap'] / coverage_df['Resources_Needed']) * 100).round(1)
    
    # Sort by revenue potential
    coverage_df = coverage_df.sort_values('Revenue_Potential', ascending=False)
    
    # Create comprehensive visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "<b>Revenue vs Current Resources</b>",
            "<b>Resource Gap by Focus Area</b>",
            "<b>Revenue at Risk</b>",
            "<b>ROI Potential</b>"
        ),
        specs=[[{"type": "scatter"}, {"type": "bar"}],
               [{"type": "pie"}, {"type": "bar"}]]
    )
    
    # 1. Revenue vs Resources scatter
    fig.add_trace(
        go.Scatter(
            x=coverage_df['Revenue_Potential'],
            y=coverage_df['Resource_Count'],
            mode='markers+text',
            marker=dict(
                size=coverage_df['Revenue_Potential'],
                color=coverage_df['Gap'],
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title=dict(text="Gap", font=dict(size=12)), x=0.45, y=0.75, tickfont=dict(size=11))
            ),
            text=coverage_df['Focus_Area'].str[:20],
            textposition="top center",
            textfont=dict(size=11, color='black', family='Arial'),
            hovertemplate='<b>%{text}</b><br>Revenue: $%{x:.1f}M<br>Resources: %{y}<br>Gap: %{marker.color}<extra></extra>',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Add optimal line
    x_range = np.linspace(0, coverage_df['Revenue_Potential'].max(), 100)
    y_optimal = x_range * resources_per_million
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_optimal,
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='Optimal Ratio',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 2. Resource Gap bars
    top_gaps = coverage_df.nlargest(10, 'Gap')
    colors = ['#dc3545' if gap > 10 else '#ffc107' if gap > 5 else '#28a745' for gap in top_gaps['Gap']]
    
    # Truncate and clean Focus Area names for display
    display_names = top_gaps['Focus_Area'].apply(lambda x: str(x)[:25] + '...' if len(str(x)) > 25 else str(x))
    
    fig.add_trace(
        go.Bar(
            x=top_gaps['Gap'],
            y=display_names,
            orientation='h',
            marker_color=colors,
            text=top_gaps['Gap'],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Gap: %{x} resources<extra></extra>',
            name='Resource Gap',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # 3. Revenue at Risk pie
    risk_categories = {
        'Well Staffed': coverage_df[coverage_df['Gap'] <= 0]['Revenue_Potential'].sum(),
        'Minor Gap (1-5)': coverage_df[(coverage_df['Gap'] > 0) & (coverage_df['Gap'] <= 5)]['Revenue_Potential'].sum(),
        'Moderate Gap (6-15)': coverage_df[(coverage_df['Gap'] > 5) & (coverage_df['Gap'] <= 15)]['Revenue_Potential'].sum(),
        'Critical Gap (>15)': coverage_df[coverage_df['Gap'] > 15]['Revenue_Potential'].sum()
    }
    
    fig.add_trace(
        go.Pie(
            labels=list(risk_categories.keys()),
            values=list(risk_categories.values()),
            marker_colors=['#28a745', '#ffc107', '#ff8c00', '#dc3545'],
            hole=0.4,
            textinfo='label+value',
            texttemplate='%{label}<br>$%{value:.1f}M',
            textfont=dict(size=12, color='white', family='Arial Bold'),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 4. ROI Potential
    coverage_df['ROI_Score'] = (coverage_df['Revenue_Potential'] / (coverage_df['Gap'].abs() + 1)).round(1)
    top_roi = coverage_df.nlargest(10, 'ROI_Score')
    
    # Truncate and clean Focus Area names for display
    roi_display_names = top_roi['Focus_Area'].apply(lambda x: str(x)[:25] + '...' if len(str(x)) > 25 else str(x))
    
    fig.add_trace(
        go.Bar(
            x=top_roi['ROI_Score'],
            y=roi_display_names,
            orientation='h',
            marker_color='#667eea',
            text=top_roi['ROI_Score'],
            textposition='outside',
            texttemplate='%{text:.1f}',
            textfont=dict(size=11),
            hovertemplate='<b>%{y}</b><br>ROI Score: %{x:.1f}<extra></extra>',
            name='ROI Score',
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Update layout with larger fonts for better readability
    fig.update_xaxes(title=dict(text="Revenue Potential ($M)", font=dict(size=14)), tickfont=dict(size=12), row=1, col=1)
    fig.update_yaxes(title=dict(text="Current Resources", font=dict(size=14)), tickfont=dict(size=12), row=1, col=1)
    fig.update_xaxes(title=dict(text="Resource Gap", font=dict(size=14)), tickfont=dict(size=12), row=1, col=2)
    fig.update_yaxes(tickfont=dict(size=11), row=1, col=2)
    fig.update_xaxes(title=dict(text="ROI Score", font=dict(size=14)), tickfont=dict(size=12), row=2, col=2)
    fig.update_yaxes(tickfont=dict(size=11), row=2, col=2)
    
    fig.update_layout(
        height=900,
        showlegend=False,
        font=dict(size=13),
        title_font=dict(size=16),
        margin=dict(t=80, b=80, l=100, r=80)
    )
    fig.update_annotations(font_size=15)
    st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.markdown("### 🔍 Key Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_gap = coverage_df['Gap'].sum()
        st.metric(
            "Total Resource Gap",
            f"{total_gap} people",
            f"${total_gap * 150:.0f}K hiring cost",
            delta_color="inverse"
        )
    
    with col2:
        revenue_at_risk = coverage_df[coverage_df['Gap'] > 0]['Revenue_Potential'].sum()
        st.metric(
            "Revenue at Risk",
            f"${revenue_at_risk:.1f}M",
            f"{(revenue_at_risk/coverage_df['Revenue_Potential'].sum()*100):.0f}% of total",
            delta_color="inverse"
        )
    
    with col3:
        critical_count = len(coverage_df[coverage_df['Gap'] > 15])
        st.metric(
            "Critical Focus Areas",
            f"{critical_count}",
            "Need immediate attention",
            delta_color="inverse"
        )

def show_current_vs_required(coverage_df):
    """Show current vs required resources by Focus Area."""
    st.markdown("### 📊 Current vs Required Resources")
    
    # Prepare data
    coverage_df['Resources_Needed'] = (coverage_df['Revenue_Potential'] * 2.5).round().astype(int)
    top_15 = coverage_df.nlargest(15, 'Revenue_Potential')
    
    # Create stacked bar chart
    fig = go.Figure()
    
    # Current resources
    fig.add_trace(go.Bar(
        name='Current Resources',
        x=top_15['Focus_Area'].str[:25],
        y=top_15['Resource_Count'],
        marker_color='#28a745',
        text=top_15['Resource_Count'],
        textposition='inside',
        textfont=dict(size=11, color='white'),
        hovertemplate='%{x}<br>Current: %{y}<extra></extra>'
    ))
    
    # Additional needed
    fig.add_trace(go.Bar(
        name='Additional Needed',
        x=top_15['Focus_Area'].str[:25],
        y=top_15['Gap'].clip(lower=0),
        marker_color='#dc3545',
        text=top_15['Gap'].clip(lower=0),
        textposition='inside',
        textfont=dict(size=11, color='white'),
        hovertemplate='%{x}<br>Gap: %{y}<extra></extra>'
    ))
    
    # Add revenue line
    fig.add_trace(go.Scatter(
        name='Revenue ($M)',
        x=top_15['Focus_Area'].str[:25],
        y=top_15['Revenue_Potential'],
        mode='lines+markers',
        marker_color='#667eea',
        yaxis='y2',
        line=dict(width=3),
        hovertemplate='%{x}<br>Revenue: $%{y:.1f}M<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text="Resource Requirements vs Revenue Opportunity", font=dict(size=18)),
        barmode='stack',
        yaxis=dict(title=dict(text="Number of Resources", font=dict(size=14)), tickfont=dict(size=12)),
        yaxis2=dict(title=dict(text="Revenue ($M)", font=dict(size=14)), tickfont=dict(size=12), overlaying='y', side='right'),
        xaxis=dict(tickangle=-45, tickfont=dict(size=11)),
        height=600,
        hovermode='x unified',
        margin=dict(t=100, b=120, l=80, r=80),
        legend=dict(font=dict(size=12))
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    st.markdown("### 📋 Detailed Requirements")
    
    display_df = coverage_df[['Focus_Area', 'Revenue_Potential', 'Resource_Count', 'Resources_Needed', 'Gap', 'Coverage_Status']].copy()
    display_df['Fulfillment %'] = ((display_df['Resource_Count'] / display_df['Resources_Needed']) * 100).round(1)
    display_df = display_df.sort_values('Revenue_Potential', ascending=False).head(15)
    
    # Apply styling
    def style_gap(val):
        if val > 15:
            return 'background-color: #ffcccc'
        elif val > 5:
            return 'background-color: #fff3cd'
        elif val > 0:
            return 'background-color: #ffffcc'
        else:
            return 'background-color: #ccffcc'
    
    styled_df = display_df.style.applymap(style_gap, subset=['Gap'])
    st.dataframe(styled_df, use_container_width=True)

def show_focus_area_deep_dive(coverage_df, data):
    """Deep dive into specific Focus Area."""
    st.markdown("### 🎯 Focus Area Deep Analysis")
    
    # Select Focus Area
    selected_fa = st.selectbox(
        "Select Focus Area for detailed analysis:",
        coverage_df.sort_values('Revenue_Potential', ascending=False)['Focus_Area'].tolist(),
        key="fa_deep_dive"
    )
    
    # Get Focus Area details
    fa_data = coverage_df[coverage_df['Focus_Area'] == selected_fa].iloc[0]
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Revenue Potential", f"${fa_data['Revenue_Potential']:.1f}M")
    
    with col2:
        st.metric("Current Resources", f"{fa_data['Resource_Count']}")
    
    with col3:
        needed = int(fa_data['Revenue_Potential'] * 2.5)
        st.metric("Resources Needed", f"{needed}")
    
    with col4:
        gap = needed - fa_data['Resource_Count']
        delta_color = "inverse" if gap > 0 else "normal"
        st.metric("Gap", f"{gap}", delta_color=delta_color)
    
    # Skills required for this Focus Area
    st.markdown("#### 🛠️ Required Skills & Capabilities")
    
    # Get skills from Focus Area mapping
    from src.focus_area_integration import FocusAreaIntegrator
    integrator = FocusAreaIntegrator()
    requirements = integrator.get_focus_area_requirements(selected_fa)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Key Services:**")
        for service in requirements['key_services'][:5]:
            st.markdown(f"• {service}")
        
        st.markdown("**Required Domains:**")
        for domain in requirements['required_domains']:
            st.markdown(f"• {domain}")
    
    with col2:
        st.markdown("**Recommended Skills:**")
        for skill in requirements['recommended_skills']:
            st.markdown(f"• {skill}")
        
        st.markdown(f"**Priority Level:** {requirements['priority_level']}")
    
    # Resource matching
    if 'resources_with_focus' in data:
        st.markdown("#### 👥 Available Resources for this Focus Area")
        
        resources_df = data['resources_with_focus']
        
        # Filter for this Focus Area
        if 'Focus_Areas' in resources_df.columns:
            matching = resources_df[
                resources_df['Focus_Areas'].str.contains(selected_fa, na=False, case=False, regex=False)
            ]
            
            # Get unique resources
            if 'resource_name' in matching.columns:
                unique_resources = matching['resource_name'].nunique()
            else:
                unique_resources = len(matching.drop_duplicates())
            
            if len(matching) > 0:
                st.success(f"✅ Found {unique_resources} unique professionals with relevant skills")
                
                # Show skill distribution
                if 'Rating' in matching.columns:
                    rating_dist = matching['Rating'].value_counts()
                    
                    fig = go.Figure(go.Bar(
                        x=rating_dist.index,
                        y=rating_dist.values,
                        marker_color='#667eea',
                        text=rating_dist.values,
                        textposition='outside'
                    ))
                    fig.update_layout(
                        title="Skill Level Distribution",
                        xaxis_title="Rating",
                        yaxis_title="Count",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"❌ No resources found - Critical hiring need!")
                st.markdown("**Immediate Action Required:**")
                st.markdown(f"• Hire {needed} specialists immediately")
                st.markdown(f"• Estimated cost: ${needed * 150}K")
                st.markdown(f"• Timeline: 3-6 months")

def show_predictive_modeling(coverage_df):
    """Predictive modeling for Focus Area growth."""
    st.markdown("### 🔮 Predictive Capacity Modeling")
    
    # Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        scenario = st.selectbox(
            "Business Scenario:",
            ["Conservative (10% growth)", "Baseline (20% growth)", "Aggressive (35% growth)"],
            index=1
        )
    
    with col2:
        timeline = st.selectbox(
            "Timeline:",
            ["Q1 2025", "Q2 2025", "H2 2025", "FY 2025"],
            index=2
        )
    
    with col3:
        focus_filter = st.multiselect(
            "Focus Areas:",
            ["All"] + coverage_df['Focus_Area'].tolist(),
            default=["All"]
        )
    
    # Calculate projections
    growth_rate = {"Conservative (10% growth)": 0.10, "Baseline (20% growth)": 0.20, "Aggressive (35% growth)": 0.35}[scenario]
    
    if "All" in focus_filter or len(focus_filter) == 0:
        projection_df = coverage_df.copy()
    else:
        projection_df = coverage_df[coverage_df['Focus_Area'].isin(focus_filter)].copy()
    
    # Project future needs
    projection_df['Future_Revenue'] = projection_df['Revenue_Potential'] * (1 + growth_rate)
    projection_df['Future_Resources_Needed'] = (projection_df['Future_Revenue'] * 2.5).round().astype(int)
    projection_df['Future_Gap'] = projection_df['Future_Resources_Needed'] - projection_df['Resource_Count']
    projection_df['Additional_Hiring'] = projection_df['Future_Gap'] - projection_df.get('Gap', 0)
    
    # Visualization
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Current vs Projected Gaps", "Hiring Requirements by Quarter"),
        specs=[[{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # Current vs Projected
    top_10 = projection_df.nlargest(10, 'Future_Gap')
    
    fig.add_trace(
        go.Bar(
            name='Current Gap',
            x=top_10['Focus_Area'].str[:15],
            y=top_10.get('Gap', 0),
            marker_color='#ffc107',
            showlegend=True
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            name='Projected Gap',
            x=top_10['Focus_Area'].str[:15],
            y=top_10['Future_Gap'],
            marker_color='#dc3545',
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Quarterly hiring plan
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    quarterly_hiring = [
        int(projection_df['Additional_Hiring'].sum() * 0.3),
        int(projection_df['Additional_Hiring'].sum() * 0.3),
        int(projection_df['Additional_Hiring'].sum() * 0.2),
        int(projection_df['Additional_Hiring'].sum() * 0.2)
    ]
    cumulative = np.cumsum(quarterly_hiring)
    
    fig.add_trace(
        go.Scatter(
            x=quarters,
            y=quarterly_hiring,
            mode='lines+markers',
            name='Quarterly Hiring',
            marker=dict(size=10),
            line=dict(width=3)
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=quarters,
            y=cumulative,
            mode='lines+markers',
            name='Cumulative',
            marker=dict(size=10),
            line=dict(width=3, dash='dash')
        ),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary metrics
    st.markdown("### 📊 Projection Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_future_gap = projection_df['Future_Gap'].sum()
        st.metric("Total Future Gap", f"{total_future_gap} people")
    
    with col2:
        hiring_needed = projection_df['Additional_Hiring'].sum()
        st.metric("Additional Hiring Needed", f"{hiring_needed} people")
    
    with col3:
        hiring_cost = hiring_needed * 150000
        st.metric("Estimated Cost", f"${hiring_cost/1e6:.1f}M")
    
    with col4:
        future_revenue = projection_df['Future_Revenue'].sum()
        st.metric("Projected Revenue", f"${future_revenue:.1f}M")

def show_action_plan(coverage_df):
    """Generate actionable hiring and development plan."""
    st.markdown("### 📋 Strategic Action Plan")
    
    # Calculate priorities
    coverage_df['Priority_Score'] = (
        coverage_df['Revenue_Potential'] * 0.4 +
        coverage_df['Gap'].fillna(0) * 0.3 +
        (coverage_df['Priority'].map({1: 30, 2: 20, 3: 10}).fillna(10))
    )
    
    # Sort by priority
    priority_df = coverage_df.sort_values('Priority_Score', ascending=False).head(10)
    
    # Create action plan
    st.markdown("#### 🎯 Priority Hiring Plan")
    
    for idx, row in priority_df.iterrows():
        gap = int(row['Revenue_Potential'] * 2.5 - row['Resource_Count'])
        if gap > 0:
            with st.expander(f"🔴 {row['Focus_Area']} - Hire {gap} people"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Revenue at Risk:** ${row['Revenue_Potential']:.1f}M")
                    st.markdown(f"**Current Resources:** {row['Resource_Count']}")
                    st.markdown(f"**Resources Needed:** {int(row['Revenue_Potential'] * 2.5)}")
                    st.markdown(f"**Gap:** {gap} people")
                
                with col2:
                    st.markdown("**Action Items:**")
                    st.markdown(f"• Post {gap} job requisitions")
                    st.markdown(f"• Budget: ${gap * 150}K")
                    st.markdown(f"• Timeline: {'Immediate' if gap > 10 else '3 months'}")
                    st.markdown(f"• Domains: {row.get('Primary_Domains', 'Multiple')}")
                
                # Hiring timeline
                if st.button(f"Generate Hiring Timeline", key=f"timeline_{idx}"):
                    months = ['Month 1', 'Month 2', 'Month 3', 'Month 4', 'Month 5', 'Month 6']
                    monthly_hires = [gap//6 + (1 if i < gap%6 else 0) for i in range(6)]
                    
                    fig = go.Figure(go.Bar(
                        x=months,
                        y=monthly_hires,
                        marker_color='#667eea',
                        text=monthly_hires,
                        textposition='outside'
                    ))
                    fig.update_layout(
                        title=f"Hiring Timeline for {row['Focus_Area']}",
                        yaxis_title="Hires per Month",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # Development plan for existing resources
    st.markdown("#### 🎓 Skill Development Plan")
    
    development_areas = [
        {"Focus Area": "AI Solutions", "Current": 205, "Target": 300, "Training": "AI/ML Certification Program"},
        {"Focus Area": "Cloud-Native Platforms", "Current": 161, "Target": 250, "Training": "Kubernetes & Container Training"},
        {"Focus Area": "Cybersecurity", "Current": 9, "Target": 50, "Training": "Security Certification Fast-track"},
        {"Focus Area": "Data Solutions", "Current": 150, "Target": 200, "Training": "Data Engineering Bootcamp"}
    ]
    
    dev_df = pd.DataFrame(development_areas)
    dev_df['Gap'] = dev_df['Target'] - dev_df['Current']
    
    for _, row in dev_df.iterrows():
        if row['Gap'] > 0:
            st.info(f"📚 **{row['Focus Area']}**: Train {row['Gap']} people - {row['Training']}")
    
    # Budget summary
    st.markdown("#### 💰 Budget Summary")
    
    total_gap = coverage_df['Gap'].sum()
    hiring_cost = total_gap * 150000
    training_cost = len(coverage_df) * 5000  # Average training cost
    total_cost = hiring_cost + training_cost
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Hiring Budget", f"${hiring_cost/1e6:.1f}M")
    
    with col2:
        st.metric("Training Budget", f"${training_cost/1e6:.1f}M")
    
    with col3:
        st.metric("Total Investment", f"${total_cost/1e6:.1f}M")
    
    with col4:
        roi = (coverage_df['Revenue_Potential'].sum() / (total_cost/1e6)) if total_cost > 0 else 0
        st.metric("Expected ROI", f"{roi:.1f}x")

# Example usage
if __name__ == "__main__":
    st.set_page_config(page_title="Focus Area Capacity Planning", page_icon="🎯", layout="wide")
    
    # Load sample data
    sample_coverage = pd.DataFrame({
        'Focus_Area': ['AI Solutions', 'Cloud-Native Platforms', 'Data Solutions', 'Cybersecurity', 'SAP'],
        'Revenue_Potential': [51.3, 18.5, 22.0, 24.5, 12.8],
        'Resource_Count': [205, 161, 150, 9, 100],
        'Priority': [1, 1, 1, 1, 2],
        'Primary_Domains': ['AI, DATA', 'CLD, APP', 'DATA', 'CYB', 'APP'],
        'Coverage_Status': ['Good', 'Good', 'Good', 'Critical', 'Limited']
    })
    
    sample_resources = pd.DataFrame({
        'resource_name': [f"Person_{i}" for i in range(565)],
        'Focus_Areas': ['AI Solutions; Data Solutions' for _ in range(565)]
    })
    
    data = {
        'focus_area_coverage': sample_coverage,
        'resources_corrected': sample_resources,
        'resources_with_focus': sample_resources
    }
    
    show_focus_area_capacity_planning(data)