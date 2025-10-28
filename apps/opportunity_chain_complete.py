"""
HPE Complete Opportunity-to-Resource Chain Platform
Full implementation of the chain:
1. Opportunity → Product Line
2. Product Line → Services
3. Services → Skillsets
4. Skillsets → Skills
5. Skills → Resources
6. Resource Double-Click (Detailed View)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from collections import defaultdict, Counter
import warnings

warnings.filterwarnings('ignore')

# Product Line Mapping - Direct matches only
PL_MAPPING = {
    '60': '60 (IJ)',  # Cloud-Native Platforms
    '1Z': '1Z (PN)',  # Network
    '5V': '5V (II)',  # Hybrid Workplace
    '4J': '4J (SX)',  # Education Services
    'G4': 'G4 (PK)',  # Private Platforms
    'PD': 'PD (C8)',  # HPE POD Modular DC
}

# Page Configuration
st.set_page_config(
    page_title="Complete Opportunity Chain | HPE",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS with enhanced resource cards
st.markdown("""
<style>
    .chain-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
    }

    .chain-step {
        background: white;
        border-left: 5px solid #01a982;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .resource-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 2px solid #01a982;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        cursor: pointer;
        transition: transform 0.3s;
    }

    .resource-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }

    .skill-badge {
        background: #01a982;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        display: inline-block;
        margin: 0.25rem;
        font-size: 0.9rem;
    }

    .rating-star {
        color: #ffc107;
        font-size: 1.2rem;
    }

    .chain-flow {
        background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        font-weight: bold;
    }

    .resource-detail-modal {
        background: white;
        border: 3px solid #01a982;
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }

    /* Style popover buttons to match st.info() blue boxes */
    [data-testid="stPopover"] button {
        background-color: #d4edff !important;
        border: 1px solid #0096D6 !important;
        color: #0c5273 !important;
        width: 100% !important;
        padding: 0.75rem 1rem !important;
        border-radius: 0.5rem !important;
        font-weight: 500 !important;
        text-align: center !important;
    }

    [data-testid="stPopover"] button:hover {
        background-color: #b8e0ff !important;
        border-color: #0078b3 !important;
    }
</style>
""", unsafe_allow_html=True)


class CompleteOpportunityChain:
    """Complete implementation of the opportunity to resource chain"""

    def __init__(self):
        """Initialize with all data files"""
        self.load_all_data()
        self.create_pl_mapping()  # Must come before build_complete_chain
        self.build_complete_chain()

    @staticmethod
    def format_currency_millions(value):
        """Format currency value in millions with proper rounding"""
        if pd.isna(value) or value == 0:
            return "$0.00M"
        millions = value / 1e6
        return f"${millions:.2f}M"

    def load_all_data(self):
        """Load all required data files"""
        try:
            # 1. OPPORTUNITY DATA
            self.opportunity_df = pd.read_excel('data/Opportunioty and PL sample.xlsx')

            # 2. SERVICES MAPPING
            self.services_mapping = pd.read_excel(
                'data/Services_to_skillsets Mapping.xlsx',
                sheet_name='Master v5'
            )

            # 3. SKILLSETS TO SKILLS
            self.skills_mapping = self.load_skills_mapping()

            # 4. EMPLOYEE SKILLS
            self.employees_df = pd.read_excel('data/DETAILS (28).xlsx', sheet_name='Export')

            # Process data
            self.process_opportunity_data()
            self.process_employee_data()

        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()

    def load_skills_mapping(self):
        """Load and combine all skills mapping sheets"""
        excel_file = pd.ExcelFile('data/Skillsets_to_Skills_mapping.xlsx')
        all_skills = []

        for sheet in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet)
            all_skills.append(df)

        return all_skills

    def process_opportunity_data(self):
        """Process and summarize opportunity data"""
        # Clean columns
        self.opportunity_df.columns = self.opportunity_df.columns.str.strip()

        # Convert revenue and rename to TCV USD
        self.opportunity_df['TCV USD'] = pd.to_numeric(
            self.opportunity_df['Schedule Amount (converted)'], errors='coerce'
        )

        # Create summary
        if self.opportunity_df.empty:
            self.opportunity_summary = pd.DataFrame(
                columns=[
                    'HPE Opportunity Id',
                    'Opportunity Name',
                    'Account Name',
                    'Product Line',
                    'TCV USD',
                    'Sales Stage',
                    'Forecast Category'
                ]
            )
        else:
            self.opportunity_summary = self.opportunity_df.groupby(
                ['HPE Opportunity Id', 'Opportunity Name', 'Account Name', 'Product Line']
            ).agg({
                'TCV USD': 'sum',
                'Sales Stage': 'first',
                'Forecast Category': 'first'
            }).reset_index()

            # Add PL Code and Description columns
            def extract_pl_code(pl):
                """Extract PL code from Product Line"""
                if ' - ' in str(pl):
                    return str(pl).split(' - ')[0].strip()
                return str(pl).strip()

            def extract_pl_description(pl):
                """Extract PL description from Product Line"""
                if ' - ' in str(pl):
                    return str(pl).split(' - ')[1].strip()
                return ''

            self.opportunity_summary['Product Line Code'] = self.opportunity_summary['Product Line'].apply(extract_pl_code)
            self.opportunity_summary['Product Line Description'] = self.opportunity_summary['Product Line'].apply(extract_pl_description)

        # Calculate metrics
        self.total_opportunities = self.opportunity_df['HPE Opportunity Id'].nunique()
        self.total_value = self.opportunity_df['TCV USD'].sum()
        self.unique_pls = self.opportunity_df['Product Line'].nunique()

    def process_employee_data(self):
        """Process employee data for efficient lookup"""
        # Create employee profiles
        self.employee_profiles = {}

        for _, row in self.employees_df.iterrows():
            emp_name = row.get('Resource_Name')
            if pd.notna(emp_name):
                if emp_name not in self.employee_profiles:
                    self.employee_profiles[emp_name] = {
                        'skills': [],
                        'location': row.get('RMR_City', 'Unknown'),
                        'manager': row.get('Resource_Manager_Name', 'Unknown'),
                        'domain': row.get('RMR_Domain', 'Unknown'),
                        'unit': row.get('RMR_MRU', 'Unknown')
                    }

                if pd.notna(row.get('Skill_Certification_Name')):
                    # Use Proficieny_Rating column which has numeric values
                    proficiency_rating = row.get('Proficieny_Rating', 0)
                    # Also get the text rating for display purposes
                    text_rating = row.get('Rating', '')

                    # Ensure proficiency_rating is numeric
                    try:
                        proficiency_rating = float(proficiency_rating) if proficiency_rating else 0
                    except:
                        proficiency_rating = 0

                    self.employee_profiles[emp_name]['skills'].append({
                        'skill': row['Skill_Certification_Name'],
                        'skillset': row.get('Skill_Set_Name', ''),
                        'rating': proficiency_rating,
                        'rating_text': text_rating  # Keep the descriptive text too
                    })

    def create_pl_mapping(self):
        """Create mapping between opportunity PLs and service PLs"""
        self.pl_mapping = PL_MAPPING.copy()

    def build_complete_chain(self):
        """Build the complete chain mappings"""

        # STEP 1: OPPORTUNITY → PRODUCT LINE
        self.opportunity_to_pl = {}
        if self.opportunity_summary.empty:
            return
        for _, row in self.opportunity_summary.iterrows():
            opp_id = row['HPE Opportunity Id']
            # Extract just the PL code (before the " - " if present)
            pl_full = row['Product Line']
            if ' - ' in str(pl_full):
                pl_code = str(pl_full).split(' - ')[0].strip()
            else:
                pl_code = str(pl_full).strip()

            self.opportunity_to_pl[opp_id] = {
                'pl_code': pl_code,
                'pl_full': pl_full,
                'opportunity_name': row['Opportunity Name'],
                'account': row['Account Name'],
                'value': row['TCV USD'],
                'stage': row.get('Sales Stage', 'Unknown'),
                'forecast': row.get('Forecast Category', 'Unknown')
            }

        # STEP 2: PRODUCT LINE → SERVICES
        self.pl_to_services = defaultdict(set)
        self.service_to_pl = defaultdict(set)

        for _, row in self.services_mapping.iterrows():
            service_pl = row.get('FY25 PL')
            service_name = row.get('New Service Name')

            if pd.notna(service_pl) and pd.notna(service_name):
                self.service_to_pl[service_name].add(service_pl)

                # Map to opportunity PLs
                for opp_pl, mapped_pls in self.pl_mapping.items():
                    # Skip N/A mappings
                    if 'N/A' in mapped_pls:
                        continue
                    if service_pl in mapped_pls:
                        self.pl_to_services[opp_pl].add(service_name)

        # STEP 3: SERVICES → SKILLSETS
        self.service_to_skillsets = defaultdict(set)
        for _, row in self.services_mapping.iterrows():
            service = row.get('New Service Name')
            skillset = row.get('Skill Set')
            if pd.notna(service) and pd.notna(skillset):
                self.service_to_skillsets[service].add(skillset)

        # STEP 4: SKILLSETS → SKILLS
        self.skillset_to_skills = defaultdict(set)
        self.skill_to_skillsets = defaultdict(set)

        # Process each sheet separately
        for df in self.skills_mapping:
            for _, row in df.iterrows():
                # Get skillset name - prefer FY25 unless it's "No change"
                fy25_skillset = row.get("FY'25 Skillset Name")
                fy24_skillset = row.get("FY'24 Skillset Name")

                # Determine which skillset to use
                if pd.notna(fy25_skillset) and str(fy25_skillset).lower() not in ['no change', 'no changes']:
                    skillset = fy25_skillset
                elif pd.notna(fy24_skillset):
                    skillset = fy24_skillset
                else:
                    continue

                # Get skill (try multiple column names)
                skill = None
                for col in ['Skill or Certification Name', 'Skill/Certification Name',
                           'Skill Name', 'Skill / Certification: Skill / Certification Name']:
                    if col in row and pd.notna(row[col]):
                        skill = row[col]
                        break

                if pd.notna(skillset) and pd.notna(skill):
                    # Add mapping for the actual skillset name
                    self.skillset_to_skills[skillset].add(skill)
                    self.skill_to_skillsets[skill].add(skillset)

                    # IMPORTANT: Also add mapping for FY24 name when FY25 is "No change"
                    # This ensures skillsets from services file can find their skills
                    if pd.notna(fy24_skillset) and pd.notna(fy25_skillset) and str(fy25_skillset).lower() in ['no change', 'no changes']:
                        self.skillset_to_skills[fy24_skillset].add(skill)
                        self.skill_to_skillsets[skill].add(fy24_skillset)

        # STEP 5: SKILLS → RESOURCES
        self.skill_to_resources = defaultdict(list)
        for emp_name, profile in self.employee_profiles.items():
            for skill_info in profile['skills']:
                skill_name = skill_info['skill']
                self.skill_to_resources[skill_name].append({
                    'name': emp_name,
                    'rating': skill_info['rating'],
                    'location': profile['location'],
                    'manager': profile['manager']
                })

    def get_complete_chain(self, opportunity_id):
        """Get the complete chain for an opportunity"""
        if opportunity_id not in self.opportunity_to_pl:
            return None

        opp_data = self.opportunity_to_pl[opportunity_id]
        pl_code = opp_data['pl_code']

        chain = {
            'opportunity': opp_data,
            'product_line': pl_code,
            'services': list(self.pl_to_services.get(pl_code, set()))[:10],
            'skillsets': set(),
            'skills': set(),
            'resources': {}
        }

        # Get skillsets from services
        for service in chain['services']:
            chain['skillsets'].update(self.service_to_skillsets.get(service, set()))

        chain['skillsets'] = list(chain['skillsets'])[:20]

        # Get skills from skillsets
        for skillset in chain['skillsets']:
            chain['skills'].update(self.skillset_to_skills.get(skillset, set()))

        chain['skills'] = list(chain['skills'])[:50]

        # Get resources from skills
        resource_scores = defaultdict(lambda: {'count': 0, 'max_rating': 0, 'skills': []})

        for skill in chain['skills']:
            for resource in self.skill_to_resources.get(skill, []):
                resource_scores[resource['name']]['count'] += 1
                # Track max rating (peak proficiency) instead of averaging
                current_max = resource_scores[resource['name']]['max_rating']
                resource_scores[resource['name']]['max_rating'] = max(current_max, resource['rating'])
                resource_scores[resource['name']]['skills'].append({
                    'skill': skill,
                    'rating': resource['rating']
                })

        # Sort resources by match count and max rating
        sorted_resources = sorted(
            resource_scores.items(),
            key=lambda x: (x[1]['count'], x[1]['max_rating']),
            reverse=True
        )

        chain['resources'] = dict(sorted_resources[:20])

        return chain

    @st.dialog("👤 Resource Detail View", width="large")
    def render_resource_detail(self, resource_name):
        """Render detailed view for a specific resource in a modal dialog"""
        if resource_name not in self.employee_profiles:
            st.error(f"Resource {resource_name} not found")
            return

        profile = self.employee_profiles[resource_name]

        st.markdown(f"### {resource_name}")
        st.markdown("---")

        # Basic Information
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Location", profile['location'])
        with col2:
            st.metric("Manager", profile['manager'])
        with col3:
            st.metric("Domain", profile['domain'])
        with col4:
            st.metric("Unit", profile['unit'])

        # Skills Summary
        st.subheader("📚 Skills Profile")

        total_skills = len(profile['skills'])
        expert_skills = sum(1 for s in profile['skills'] if s['rating'] >= 4)
        advanced_skills = sum(1 for s in profile['skills'] if s['rating'] == 3)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Skills", total_skills)
        with col2:
            st.metric("Expert (4)", expert_skills)
        with col3:
            st.metric("Advanced (3)", advanced_skills)
        with col4:
            avg_rating = np.mean([s['rating'] for s in profile['skills'] if s['rating'] > 0])
            st.metric("Avg Rating", f"{avg_rating:.1f}" if avg_rating > 0 else "N/A")

        # Skills by Category
        st.subheader("🎯 Skills by Proficiency")

        # Group skills by rating
        skills_by_rating = defaultdict(list)
        for skill_info in profile['skills']:
            rating = int(skill_info['rating']) if skill_info['rating'] > 0 else 0
            skills_by_rating[rating].append(skill_info)

        # Display skills by rating level
        rating_labels = {
            4: ("Expert", "🌟"),
            3: ("Advanced", "⭐"),
            2: ("Intermediate", "📘"),
            1: ("Beginner", "📗"),
            0: ("Unrated", "❓")
        }

        for rating in [4, 3, 2, 1, 0]:
            if rating in skills_by_rating:
                label, icon = rating_labels[rating]
                skills = skills_by_rating[rating]

                with st.expander(f"{icon} {label} ({len(skills)} skills)"):
                    # Display in columns
                    cols = st.columns(2)
                    for idx, skill_info in enumerate(skills[:20]):  # Limit display
                        with cols[idx % 2]:
                            st.markdown(f"""
                            <div class="skill-badge">
                                {skill_info['skill'][:40]}
                            </div>
                            """, unsafe_allow_html=True)
                            if skill_info['skillset']:
                                st.caption(f"Skillset: {skill_info['skillset'][:30]}")

        # Opportunity Matches
        st.subheader("💰 Matching Opportunities")

        # Find opportunities this resource could support
        matching_opps = []
        resource_skills = set(s['skill'] for s in profile['skills'])

        for opp_id, opp_data in self.opportunity_to_pl.items():
            chain = self.get_complete_chain(opp_id)
            if chain:
                matching_skills = resource_skills.intersection(set(chain['skills']))
                if len(matching_skills) >= 3:  # At least 3 matching skills
                    matching_opps.append({
                        'id': opp_id,
                        'name': opp_data['opportunity_name'],
                        'value': opp_data['value'],
                        'match_count': len(matching_skills),
                        'coverage': len(matching_skills) / len(chain['skills']) * 100 if chain['skills'] else 0
                    })

        # Sort by match count and value
        matching_opps.sort(key=lambda x: (x['match_count'], x['value']), reverse=True)

        if matching_opps:
            st.info(f"This resource matches {len(matching_opps)} opportunities")

            for opp in matching_opps[:5]:  # Show top 5
                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    st.markdown(f"**{opp['name'][:50]}...**")
                    st.caption(f"ID: {opp['id']}")

                with col2:
                    st.metric("Value", f"${opp['value']/1e6:.1f}M")

                with col3:
                    st.metric("Coverage", f"{opp['coverage']:.0f}%")
        else:
            st.warning("No matching opportunities found for this resource")

    def render_chain_visualization(self, opportunity_id):
        """Render interactive chain visualization"""
        chain = self.get_complete_chain(opportunity_id)

        if not chain:
            st.error("Chain data not available")
            return

        # Create Sankey diagram
        labels = []
        customdata = []  # Full text for hover
        sources = []
        targets = []
        values = []
        colors = []

        node_index = {}
        current_idx = 0

        # Add opportunity node (shortened label, full name in hover)
        opp_short = chain['opportunity']['opportunity_name'][:25] + "..." if len(chain['opportunity']['opportunity_name']) > 25 else chain['opportunity']['opportunity_name']
        labels.append(opp_short)
        customdata.append(chain['opportunity']['opportunity_name'])
        node_index[opp_short] = current_idx
        colors.append('#667eea')
        current_idx += 1

        # Add PL node
        pl_label = chain['product_line']
        labels.append(pl_label)
        customdata.append(f"Product Line: {chain['product_line']}")
        node_index[pl_label] = current_idx
        colors.append('#764ba2')
        current_idx += 1

        sources.append(node_index[opp_short])
        targets.append(node_index[pl_label])
        values.append(chain['opportunity']['value'])

        # Add services (limited)
        for service in chain['services'][:3]:
            # Shorten service name intelligently
            service_short = service[:30] + "..." if len(service) > 30 else service
            if service_short not in node_index:
                labels.append(service_short)
                customdata.append(f"Service: {service}")
                node_index[service_short] = current_idx
                colors.append('#01a982')
                current_idx += 1

            sources.append(node_index[pl_label])
            targets.append(node_index[service_short])
            values.append(chain['opportunity']['value'] / len(chain['services'][:3]))

            # Add skillsets for this service
            service_skillsets = list(self.service_to_skillsets.get(service, set()))[:2]
            for skillset in service_skillsets:
                skillset_short = skillset[:25] + "..." if len(skillset) > 25 else skillset
                if skillset_short not in node_index:
                    labels.append(skillset_short)
                    customdata.append(f"Skillset: {skillset}")
                    node_index[skillset_short] = current_idx
                    colors.append('#f5576c')
                    current_idx += 1

                sources.append(node_index[service_short])
                targets.append(node_index[skillset_short])
                values.append(chain['opportunity']['value'] / (len(chain['services'][:3]) * 2))

        # Create Sankey
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=20,
                thickness=25,
                line=dict(color="black", width=0.5),
                label=labels,
                color=colors,
                customdata=customdata,
                hovertemplate='%{customdata}<br>Value: %{value:,.0f}<extra></extra>'
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color='rgba(100, 100, 100, 0.2)'
            ),
            textfont=dict(
                color='black',
                size=16,
                family='Arial Black, Arial, sans-serif'
            )
        )])

        fig.update_layout(
            title=dict(
                text=f"Complete Chain for Opportunity: ${chain['opportunity']['value']/1e6:.2f}M",
                font=dict(size=18, family="Arial, sans-serif", color='black')
            ),
            height=600,
            font=dict(size=16, family="Arial, sans-serif", color='black'),
            margin=dict(l=10, r=10, t=60, b=10),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )

        st.plotly_chart(fig, use_container_width=True)

    def render(self):
        """Main render method with tabbed interface"""
        # Header
        st.markdown("""
        <div class="chain-header">
            <h1>🔄 Complete Opportunity-to-Resource Chain</h1>
            <p>Full chain: Opportunity → PL → Services → Skillsets → Skills → Resources (with drill-down)</p>
        </div>
        """, unsafe_allow_html=True)

        # Top-level metrics
        metrics_col1, metrics_col2, metrics_col3, metrics_col4, metrics_col5 = st.columns(5)
        with metrics_col1:
            st.metric("Opportunities", f"{self.total_opportunities:,}")
        with metrics_col2:
            st.metric("Total Value", f"${self.total_value/1e6:.1f}M")
        with metrics_col3:
            st.metric("Product Lines", self.unique_pls)
        with metrics_col4:
            st.metric("Resources", len(self.employee_profiles))
        with metrics_col5:
            st.metric("Skills", len(self.skill_to_resources))

        st.divider()

        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview",
            "🔗 Chain Analysis",
            "👥 Resources",
            "🔍 Search & Filter"
        ])

        with tab1:
            self.render_overview_tab()

        with tab2:
            self.render_chain_analysis_tab()

        with tab3:
            self.render_resources_tab()

        with tab4:
            self.render_search_filter_tab()

    def render_overview_tab(self):
        """Render the Overview tab"""
        st.header("📊 Platform Overview")

        # Summary statistics
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Opportunity Distribution")
            # Top opportunities by value
            top_opps = self.opportunity_summary.nlargest(10, 'TCV USD').copy()
            top_opps['TCV USD (M)'] = top_opps['TCV USD'].apply(self.format_currency_millions)
            st.dataframe(
                top_opps[['HPE Opportunity Id', 'Opportunity Name', 'Product Line Code', 'Product Line Description', 'TCV USD (M)']],
                use_container_width=True,
                hide_index=True
            )

        with col2:
            st.subheader("🎯 Product Line Summary")
            pl_summary = self.opportunity_summary.groupby('Product Line').agg({
                'HPE Opportunity Id': 'count',
                'TCV USD': 'sum'
            }).reset_index()
            pl_summary.columns = ['Product Line', 'Count', 'Total Value']
            pl_summary['Total Value (M)'] = pl_summary['Total Value'].apply(self.format_currency_millions)
            pl_summary = pl_summary.sort_values('Total Value', ascending=False)
            st.dataframe(pl_summary[['Product Line', 'Count', 'Total Value (M)']], use_container_width=True, hide_index=True)

        # Visualizations
        st.subheader("📊 Visual Analytics")
        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            # Pie chart for TCV USD by Product Line
            pl_value_summary = self.opportunity_summary.groupby('Product Line').agg({
                'TCV USD': 'sum'
            }).reset_index()
            pl_value_summary = pl_value_summary.sort_values('TCV USD', ascending=False)

            fig_pie = px.pie(
                pl_value_summary,
                values='TCV USD',
                names='Product Line',
                title='TCV USD Distribution by Product Line',
                hole=0.4
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)

        with viz_col2:
            # Bar chart for Opportunity Count by Product Line
            pl_count_summary = self.opportunity_summary.groupby('Product Line').agg({
                'HPE Opportunity Id': 'count'
            }).reset_index()
            pl_count_summary.columns = ['Product Line', 'Count']
            pl_count_summary = pl_count_summary.sort_values('Count', ascending=False)

            fig_bar = px.bar(
                pl_count_summary,
                x='Product Line',
                y='Count',
                title='Opportunity Count by Product Line',
                color='Count',
                color_continuous_scale='Blues'
            )
            fig_bar.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)

        # Coverage metrics
        st.subheader("📊 Mapping Coverage")
        coverage_col1, coverage_col2, coverage_col3 = st.columns(3)

        with coverage_col1:
            mapped_opps = sum(
                1 for _, data in self.opportunity_to_pl.items()
                if data['pl_code'] in self.pl_mapping
            )
            coverage = (
                (mapped_opps / self.total_opportunities) * 100
                if self.total_opportunities
                else 0
            )
            st.metric("Opportunities with Direct PL Mapping", f"{mapped_opps:,} ({coverage:.1f}%)")

        with coverage_col2:
            services_count = sum(len(services) for services in self.pl_to_services.values())
            st.metric("Total Services Mapped", services_count)

        with coverage_col3:
            skillsets_count = sum(len(skillsets) for skillsets in self.service_to_skillsets.values())
            st.metric("Total Skillsets Identified", skillsets_count)

    def render_chain_analysis_tab(self):
        """Render the Chain Analysis tab"""
        st.header("🔗 Opportunity Chain Analysis")

        # Opportunity selector
        if self.opportunity_summary.empty:
            st.warning("No opportunities available for analysis.")
            return

        top_opps = self.opportunity_summary.nlargest(50, 'TCV USD')

        opp_options = {
            f"{row['HPE Opportunity Id']}: {row['Opportunity Name'][:50]}... (${row['TCV USD']/1e6:.2f}M)":
            row['HPE Opportunity Id']
            for _, row in top_opps.iterrows()
        }

        if not opp_options:
            st.warning("Opportunities exist, but none meet the selection criteria.")
            return

        selected = st.selectbox(
            "Choose an opportunity to explore the complete chain",
            options=list(opp_options.keys())
        )

        if selected:
            opp_id = opp_options[selected]
            chain = self.get_complete_chain(opp_id)

            if chain:
                # Display the chain steps
                st.header("📊 Chain Analysis")

                # Step indicators
                steps = st.columns(6)

                with steps[0]:
                    st.markdown("""
                    <div class="chain-flow">
                    1️⃣ Opportunity
                    </div>
                    """, unsafe_allow_html=True)
                    st.info(opp_id)
                    st.caption(chain['opportunity']['opportunity_name'][:40])

                with steps[1]:
                    st.markdown("""
                    <div class="chain-flow">
                    2️⃣ PL Code | Description
                    </div>
                    """, unsafe_allow_html=True)
                    # Display both code and description side-by-side
                    pl_code = chain['opportunity'].get('pl_code', chain['product_line'])
                    pl_full = chain['opportunity'].get('pl_full', chain['product_line'])
                    # Extract description if available
                    if ' - ' in str(pl_full):
                        pl_desc = str(pl_full).split(' - ')[1].strip()
                        st.info(f"{pl_code} | {pl_desc}")
                    else:
                        st.info(pl_code)

                with steps[2]:
                    st.markdown("""
                    <div class="chain-flow">
                    3️⃣ Services
                    </div>
                    """, unsafe_allow_html=True)
                    with st.popover(f"{len(chain['services'])} services", use_container_width=True):
                        st.markdown("**Services Required:**")
                        for idx, service in enumerate(chain['services'][:10], 1):
                            st.write(f"{idx}. {service}")
                        if len(chain['services']) > 10:
                            st.caption(f"+ {len(chain['services']) - 10} more services")

                with steps[3]:
                    st.markdown("""
                    <div class="chain-flow">
                    4️⃣ Skillsets
                    </div>
                    """, unsafe_allow_html=True)
                    with st.popover(f"{len(chain['skillsets'])} skillsets", use_container_width=True):
                        st.markdown("**Skillsets Required:**")
                        for idx, skillset in enumerate(chain['skillsets'][:10], 1):
                            st.write(f"{idx}. {skillset}")
                        if len(chain['skillsets']) > 10:
                            st.caption(f"+ {len(chain['skillsets']) - 10} more skillsets")

                with steps[4]:
                    st.markdown("""
                    <div class="chain-flow">
                    5️⃣ Skills
                    </div>
                    """, unsafe_allow_html=True)
                    with st.popover(f"{len(chain['skills'])} skills", use_container_width=True):
                        st.markdown("**Skills Required:**")
                        # Group skills and show with resource counts
                        for idx, skill in enumerate(chain['skills'][:15], 1):
                            resource_count = len(self.skill_to_resources.get(skill, []))
                            st.write(f"{idx}. {skill[:50]} ({resource_count} resources)")
                        if len(chain['skills']) > 15:
                            st.caption(f"+ {len(chain['skills']) - 15} more skills")

                with steps[5]:
                    st.markdown("""
                    <div class="chain-flow">
                    6️⃣ Resources
                    </div>
                    """, unsafe_allow_html=True)
                    with st.popover(f"{len(chain['resources'])} matched", use_container_width=True):
                        st.markdown("**Top Matched Resources:**")
                        for idx, (resource_name, resource_data) in enumerate(list(chain['resources'].items())[:10], 1):
                            profile = self.employee_profiles.get(resource_name, {})
                            location = profile.get('location', 'Unknown')
                            match_count = resource_data['count']
                            st.write(f"{idx}. **{resource_name}** ({location}) - {match_count} skills")
                        if len(chain['resources']) > 10:
                            st.caption(f"+ {len(chain['resources']) - 10} more resources")

                # Visualization
                st.header("🔗 Chain Visualization")
                self.render_chain_visualization(opp_id)

                # Resources Section
                st.header("👥 Matched Resources")

                if chain['resources']:
                    st.success(f"Found {len(chain['resources'])} resources matching this opportunity")

                    # Display resources
                    for resource_name, resource_data in list(chain['resources'].items())[:10]:
                        profile = self.employee_profiles.get(resource_name, {})

                        col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])

                        with col1:
                            # Make resource clickable - opens modal
                            if st.button(f"👤 {resource_name}", key=f"res_{resource_name}"):
                                self.render_resource_detail(resource_name)

                            # Show top skills
                            top_skills = ", ".join([s['skill'][:20] for s in resource_data['skills'][:3]])
                            st.caption(f"Skills: {top_skills}...")

                        with col2:
                            st.metric("Matches", resource_data['count'])

                        with col3:
                            max_rating = resource_data['max_rating']
                            st.metric("Max Rating", f"{max_rating:.1f}")

                        with col4:
                            st.metric("Location", profile.get('location', 'Unknown'))

                        with col5:
                            # Availability indicator (mock)
                            avail = hash(resource_name) % 3
                            status = ["🟢 Available", "🔴 Busy", "🟡 Partial"][avail]
                            st.markdown(status)
                            st.caption('<span style="font-size: 8px; color: #999;">mock</span>', unsafe_allow_html=True)

                else:
                    st.warning("No matching resources found for this opportunity")

                # Chain Details Expander
                with st.expander("📝 View Complete Chain Details"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Services")
                        for service in chain['services']:
                            st.write(f"• {service}")

                        st.subheader("Skillsets")
                        for skillset in chain['skillsets'][:10]:
                            st.write(f"• {skillset}")

                    with col2:
                        st.subheader("Top Skills Required")
                        for skill in chain['skills'][:15]:
                            resource_count = len(self.skill_to_resources.get(skill, []))
                            st.write(f"• {skill[:40]} ({resource_count} resources)")

    def render_resources_tab(self):
        """Render the Resources tab"""
        st.header("👥 Resource Management")

        # Resource search
        search_col1, search_col2 = st.columns([2, 1])
        with search_col1:
            resource_search = st.text_input("🔍 Search for resources by name", placeholder="Enter employee name...")
        with search_col2:
            skill_filter = st.text_input("🎯 Filter by skill", placeholder="Enter skill keyword...")

        # Resource statistics
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            st.metric("Total Resources", len(self.employee_profiles))
        with stat_col2:
            total_resources = len(self.employee_profiles)
            avg_skills = (
                sum(len(p['skills']) for p in self.employee_profiles.values()) / total_resources
                if total_resources
                else 0
            )
            st.metric("Avg Skills per Resource", f"{avg_skills:.1f}")
        with stat_col3:
            st.metric("Unique Skills", len(self.skill_to_resources))

        # Resource list
        st.subheader("Resource Directory")

        # Filter resources
        filtered_resources = []
        for name, profile in self.employee_profiles.items():
            if resource_search and resource_search.lower() not in name.lower():
                continue
            if skill_filter:
                skills_text = " ".join([s['skill'] for s in profile['skills']])
                if skill_filter.lower() not in skills_text.lower():
                    continue
            filtered_resources.append((name, profile))

        # Display resources with improved UI
        if filtered_resources:
            for name, profile in filtered_resources[:20]:
                with st.expander(f"👤 {name} - {profile.get('location', 'Unknown')} ({len(profile['skills'])} skills)"):
                    # Header with key info
                    info_col1, info_col2, info_col3 = st.columns(3)
                    with info_col1:
                        st.markdown(f"**📍 Location:** {profile.get('location', 'N/A')}")
                    with info_col2:
                        st.markdown(f"**👔 Manager:** {profile.get('manager', 'N/A')[:25]}...")
                    with info_col3:
                        st.markdown(f"**🎯 Total Skills:** {len(profile['skills'])}")

                    st.divider()

                    # Top skills with visual indicators
                    st.markdown("**🏆 Top 5 Skills:**")
                    for skill_info in profile['skills'][:5]:
                        rating = skill_info['rating']
                        rating_text = skill_info.get('rating_text', f"Level {rating}")

                        # Color code by rating
                        if rating >= 4:
                            badge_color = "#28a745"  # Green for expert
                            emoji = "🌟"
                        elif rating >= 3:
                            badge_color = "#17a2b8"  # Blue for advanced
                            emoji = "⭐"
                        elif rating >= 2:
                            badge_color = "#ffc107"  # Yellow for intermediate
                            emoji = "📘"
                        else:
                            badge_color = "#6c757d"  # Gray for beginner
                            emoji = "📗"

                        st.markdown(f"""
                        <div style="background: {badge_color}15; border-left: 4px solid {badge_color}; padding: 8px; margin: 4px 0; border-radius: 4px;">
                            {emoji} <b>{skill_info['skill'][:45]}</b> <span style="color: {badge_color}; float: right;">({rating_text})</span>
                        </div>
                        """, unsafe_allow_html=True)

                    st.divider()

                    # Find matching opportunities button - more prominent
                    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
                    with col_btn2:
                        if st.button(f"🔍 Find Matching Opportunities", key=f"find_opp_{name}", use_container_width=True):
                            self.show_opportunities_modal(name)
        else:
            st.info("No resources found matching your criteria")

    @st.dialog("🎯 Matching Opportunities", width="large")
    def show_opportunities_modal(self, resource_name):
        """Show opportunities in a modal dialog"""
        st.markdown(f"### Finding opportunities for **{resource_name}**")
        st.markdown("---")

        self.render_resource_opportunities(resource_name)

    def render_resource_opportunities(self, resource_name):
        """Find and display opportunities matching a resource's skills"""
        if resource_name not in self.employee_profiles:
            st.error("Resource not found")
            return

        profile = self.employee_profiles[resource_name]
        resource_skills = set([s['skill'] for s in profile['skills']])

        matching_opps = []
        for opp_id, opp_data in self.opportunity_to_pl.items():
            chain = self.get_complete_chain(opp_id)
            if chain:
                matching_skills = resource_skills.intersection(set(chain['skills']))
                if len(matching_skills) >= 3:  # At least 3 matching skills
                    matching_opps.append({
                        'id': opp_id,
                        'name': opp_data['opportunity_name'],
                        'value': opp_data['value'],
                        'match_count': len(matching_skills),
                        'coverage': len(matching_skills) / len(chain['skills']) * 100 if chain['skills'] else 0
                    })

        # Sort by match count and value
        matching_opps.sort(key=lambda x: (x['match_count'], x['value']), reverse=True)

        if matching_opps:
            st.success(f"✅ Found {len(matching_opps)} matching opportunities")
            st.markdown("---")

            for idx, opp in enumerate(matching_opps[:10], 1):
                # Color code by coverage
                if opp['coverage'] >= 50:
                    border_color = "#28a745"  # Green for high coverage
                    bg_color = "#28a74510"
                elif opp['coverage'] >= 20:
                    border_color = "#17a2b8"  # Blue for medium coverage
                    bg_color = "#17a2b810"
                else:
                    border_color = "#ffc107"  # Yellow for low coverage
                    bg_color = "#ffc10710"

                st.markdown(f"""
                <div style="background: {bg_color}; border-left: 5px solid {border_color}; padding: 15px; margin: 10px 0; border-radius: 8px;">
                    <div style="font-size: 14px; color: #666; margin-bottom: 5px;">#{idx} · ID: {opp['id']}</div>
                    <div style="font-size: 18px; font-weight: bold; margin-bottom: 10px;">{opp['name'][:60]}</div>
                </div>
                """, unsafe_allow_html=True)

                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("💰 Value", self.format_currency_millions(opp['value']))
                with metric_col2:
                    st.metric("🎯 Skills Match", f"{opp['match_count']} skills")
                with metric_col3:
                    coverage_emoji = "🟢" if opp['coverage'] >= 50 else "🟡" if opp['coverage'] >= 20 else "🟠"
                    st.metric(f"{coverage_emoji} Coverage", f"{opp['coverage']:.1f}%")

                st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ No matching opportunities found")


    def render_search_filter_tab(self):
        """Render the Search & Filter tab"""
        st.header("🔍 Advanced Search & Filtering")

        # Search options
        search_type = st.radio(
            "What would you like to search for?",
            ["Opportunities", "Services", "Skills", "Resources"]
        )

        if search_type == "Opportunities":
            st.subheader("Search Opportunities")

            # Create three columns for filters
            col1, col2 = st.columns(2)
            with col1:
                opp_id_search = st.text_input("Opportunity ID contains:", placeholder="OPE-XX16XXXX41")
                opp_search = st.text_input("Opportunity name contains:", placeholder="Enter keywords...")
            with col2:
                pl_filter = st.selectbox("Product Line", ["All"] + sorted(self.opportunity_summary['Product Line'].unique().tolist()))

                # Add Sales Stage filter
                sales_stages = ["All"] + sorted([str(stage) for stage in self.opportunity_summary['Sales Stage'].unique() if pd.notna(stage)])
                sales_stage_filter = st.selectbox("Sales Stage", sales_stages)

            # Apply filters
            filtered = self.opportunity_summary.copy()

            # Filter by Opportunity ID
            if opp_id_search:
                filtered = filtered[filtered['HPE Opportunity Id'].str.contains(opp_id_search, case=False, na=False)]

            # Filter by Opportunity name
            if opp_search:
                filtered = filtered[filtered['Opportunity Name'].str.contains(opp_search, case=False, na=False)]

            # Filter by Product Line
            if pl_filter != "All":
                filtered = filtered[filtered['Product Line'] == pl_filter]

            # Filter by Sales Stage
            if sales_stage_filter != "All":
                filtered = filtered[filtered['Sales Stage'] == sales_stage_filter]

            st.write(f"Found {len(filtered)} opportunities")
            if len(filtered) > 0:
                filtered_display = filtered.head(50).copy()
                filtered_display['TCV USD (M)'] = filtered_display['TCV USD'].apply(self.format_currency_millions)
                st.dataframe(filtered_display[['HPE Opportunity Id', 'Opportunity Name', 'Product Line Code', 'Product Line Description',
                                      'Sales Stage', 'TCV USD (M)']],
                            use_container_width=True, hide_index=True)

        elif search_type == "Services":
            st.subheader("Search Services")

            col1, col2 = st.columns(2)
            with col1:
                service_search = st.text_input("Service name contains:", placeholder="Enter keywords...")
            with col2:
                # Get unique Product Line Codes for filtering - MULTI-SELECT
                pl_codes = sorted(list(set(self.opportunity_summary['Product Line Code'].unique())))
                pl_filter_services = st.multiselect(
                    "Filter by Product Line (select one or more)",
                    options=pl_codes,
                    default=[],
                    key="pl_filter_services",
                    help="Select multiple PLs to see services from all selected lines"
                )

            # Get services based on PL filter - SUPPORT MULTIPLE PLs
            if pl_filter_services:  # If any PLs selected
                all_services = set()
                for pl_code in pl_filter_services:
                    all_services.update(self.pl_to_services.get(pl_code, set()))
            else:  # No PLs selected = show all
                all_services = set()
                for services in self.pl_to_services.values():
                    all_services.update(services)

            # Apply search filter
            if service_search:
                matching = [s for s in all_services if service_search.lower() in s.lower()]
            else:
                matching = list(all_services)

            # Display result count with PL context
            if pl_filter_services:
                st.write(f"Found {len(matching)} services for PL: {', '.join(pl_filter_services)}")
            else:
                st.write(f"Found {len(matching)} services (all PLs)")

            if matching:
                for service in sorted(matching)[:50]:
                    skillsets = self.service_to_skillsets.get(service, set())
                    # Show which PLs this service belongs to
                    service_pls = self.service_to_pl.get(service, set())
                    pl_display = ", ".join(sorted(service_pls)[:3]) if service_pls else "N/A"
                    st.write(f"• **{service}** ({len(skillsets)} skillsets) - PL: {pl_display}")
            else:
                st.info("No services found matching your criteria")

        elif search_type == "Skills":
            st.subheader("Search Skills")

            col1, col2 = st.columns(2)
            with col1:
                skill_search = st.text_input("Skill name contains:", placeholder="Enter keywords...")
            with col2:
                # Product Line filter for cascading - MULTI-SELECT
                pl_codes = sorted(list(set(self.opportunity_summary['Product Line Code'].unique())))
                pl_filter_skills = st.multiselect(
                    "Filter by Product Line (select one or more)",
                    options=pl_codes,
                    default=[],
                    key="pl_filter_skills",
                    help="Select multiple PLs to see skills from all selected lines"
                )

            # Get skills based on PL filter (cascading through services -> skillsets -> skills) - SUPPORT MULTIPLE PLs
            if pl_filter_skills:  # If any PLs selected
                all_skills = set()
                for pl_code in pl_filter_skills:
                    # Get services for this PL
                    services_for_pl = self.pl_to_services.get(pl_code, set())
                    # Get skillsets for these services
                    skillsets_for_pl = set()
                    for service in services_for_pl:
                        skillsets_for_pl.update(self.service_to_skillsets.get(service, set()))
                    # Get skills for these skillsets
                    for skillset in skillsets_for_pl:
                        all_skills.update(self.skillset_to_skills.get(skillset, set()))
            else:  # No PLs selected = show all skills
                all_skills = set(self.skill_to_resources.keys())

            # Apply search filter
            if skill_search:
                matching = [(skill, self.skill_to_resources.get(skill, [])) for skill in all_skills
                          if skill_search.lower() in skill.lower()]
            else:
                matching = [(skill, self.skill_to_resources.get(skill, [])) for skill in all_skills]

            # Display result count with PL context
            if pl_filter_skills:
                st.write(f"Found {len(matching)} skills for PL: {', '.join(pl_filter_skills)}")
            else:
                st.write(f"Found {len(matching)} skills (all PLs)")

            if matching:
                # Sort by number of resources (descending)
                matching.sort(key=lambda x: len(x[1]), reverse=True)
                for skill, resources in matching[:50]:
                    # Show skillsets this skill belongs to
                    skillsets = self.skill_to_skillsets.get(skill, set())
                    skillset_display = ", ".join(sorted(list(skillsets))[:2]) if skillsets else "N/A"
                    st.write(f"• **{skill[:60]}** ({len(resources)} resources) - Skillsets: {skillset_display}")
            else:
                st.info("No skills found matching your criteria")

        elif search_type == "Resources":
            st.subheader("Search Resources")

            col1, col2 = st.columns(2)
            with col1:
                name_search = st.text_input("Resource name contains:", placeholder="Enter name...")
                location_filter = st.selectbox("Location",
                    ["All"] + sorted(list(set(p.get('location', 'Unknown') for p in self.employee_profiles.values()))))
            with col2:
                skill_search = st.text_input("Has skill:", placeholder="Enter skill...")
                # Product Line filter for cascading - MULTI-SELECT
                pl_codes = sorted(list(set(self.opportunity_summary['Product Line Code'].unique())))
                pl_filter_resources = st.multiselect(
                    "Filter by Product Line (select one or more)",
                    options=pl_codes,
                    default=[],
                    key="pl_filter_resources",
                    help="Select multiple PLs to see resources from all selected lines"
                )

            col3, col4 = st.columns(2)
            with col3:
                min_skills = st.number_input("Minimum skills", min_value=0, value=0)

            # Get relevant skills based on PL filter - SUPPORT MULTIPLE PLs
            if pl_filter_resources:  # If any PLs selected
                relevant_skills = set()
                for pl_code in pl_filter_resources:
                    # Get services for this PL
                    services_for_pl = self.pl_to_services.get(pl_code, set())
                    # Get skillsets for these services
                    skillsets_for_pl = set()
                    for service in services_for_pl:
                        skillsets_for_pl.update(self.service_to_skillsets.get(service, set()))
                    # Get skills for these skillsets
                    for skillset in skillsets_for_pl:
                        relevant_skills.update(self.skillset_to_skills.get(skillset, set()))
            else:
                relevant_skills = None  # No PL filter, consider all skills

            # Apply filters
            matching = []
            for name, profile in self.employee_profiles.items():
                # Name filter
                if name_search and name_search.lower() not in name.lower():
                    continue

                # Location filter
                if location_filter != "All" and profile.get('location') != location_filter:
                    continue

                # Skill search filter
                if skill_search:
                    skills_text = " ".join([s['skill'] for s in profile['skills']])
                    if skill_search.lower() not in skills_text.lower():
                        continue

                # Minimum skills filter
                if len(profile['skills']) < min_skills:
                    continue

                # Product Line filter (check if resource has skills relevant to the PL)
                if relevant_skills is not None:
                    resource_skills = set([s['skill'] for s in profile['skills']])
                    matching_pl_skills = resource_skills.intersection(relevant_skills)
                    if not matching_pl_skills:
                        continue
                    # Store matching skill count for display
                    pl_match_count = len(matching_pl_skills)
                else:
                    pl_match_count = 0

                matching.append((name, profile, pl_match_count))

            # Display result count with PL context
            if pl_filter_resources:
                st.write(f"Found {len(matching)} resources for PL: {', '.join(pl_filter_resources)}")
            else:
                st.write(f"Found {len(matching)} resources (all PLs)")

            if matching:
                # Sort by PL match count if filtering by PL, otherwise by total skills
                if pl_filter_resources:
                    matching.sort(key=lambda x: x[2], reverse=True)

                for name, profile, pl_match_count in matching[:50]:
                    if pl_filter_resources:
                        st.write(f"• **{name}** - {profile.get('location', 'Unknown')} ({len(profile['skills'])} total skills, {pl_match_count} matching PL)")
                    else:
                        st.write(f"• **{name}** - {profile.get('location', 'Unknown')} ({len(profile['skills'])} skills)")
            else:
                st.info("No resources found matching your criteria")


# Main execution
if __name__ == "__main__":
    platform = CompleteOpportunityChain()
    platform.render()