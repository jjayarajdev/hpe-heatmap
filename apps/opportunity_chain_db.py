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
import sqlite3
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
    page_title="Complete Opportunity Chain (DB) | HPE",
    page_icon="🗄️",
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


class CompleteOpportunityChainDB:
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
            # Connect to database
            conn = sqlite3.connect('data/heatmap.db')

            # 1. OPPORTUNITY DATA
            self.opportunity_df = pd.read_sql_query("SELECT * FROM opportunities", conn)

            # 2. SERVICES MAPPING
            self.services_mapping = pd.read_sql_query("SELECT * FROM services_skillsets", conn)

            # 3. SKILLSETS TO SKILLS
            self.skills_mapping = self.load_skills_mapping_db(conn)

            # 4. EMPLOYEE SKILLS
            self.employees_df = pd.read_sql_query("SELECT * FROM employee_skills", conn)

            # Close connection
            conn.close()

            # Process data
            self.process_opportunity_data()
            self.process_employee_data()

        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()

    def load_skills_mapping_db(self, conn):
        """Load skills mapping from database and group by sheet"""
        # Get all skills data
        skills_df = pd.read_sql_query("SELECT * FROM skillsets_skills", conn)

        # Group by sheet_name to maintain the same structure as Excel loading
        all_skills = []
        for sheet_name in skills_df['sheet_name'].unique():
            sheet_df = skills_df[skills_df['sheet_name'] == sheet_name].copy()
            sheet_df = sheet_df.drop('sheet_name', axis=1)  # Remove the sheet_name column
            all_skills.append(sheet_df)

        return all_skills

    def process_opportunity_data(self):
        """Process and summarize opportunity data"""
        # Clean columns
        self.opportunity_df.columns = self.opportunity_df.columns.str.strip()

        # Convert revenue and rename to TCV USD
        self.opportunity_df['TCV USD'] = pd.to_numeric(
            self.opportunity_df['Schedule Amount (converted)'], errors='coerce'
        )

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
                        'skills_dict': {},  # Use dict to deduplicate by skill name
                        'location': row.get('RMR_City', 'Unknown'),
                        'manager': row.get('Resource_Manager_Name', 'Unknown'),
                        'domain': row.get('RMR_Domain', 'Unknown'),
                        'unit': row.get('RMR_MRU', 'Unknown')
                    }

                if pd.notna(row.get('Skill_Certification_Name')):
                    skill_name = row['Skill_Certification_Name']

                    # Use Proficieny_Rating column which has numeric values
                    proficiency_rating = row.get('Proficieny_Rating', 0)
                    # Also get the text rating for display purposes
                    text_rating = row.get('Rating', '')

                    # Ensure proficiency_rating is numeric
                    try:
                        proficiency_rating = float(proficiency_rating) if proficiency_rating else 0
                    except:
                        proficiency_rating = 0

                    # Deduplicate: If skill already exists, keep the higher rating
                    if skill_name not in self.employee_profiles[emp_name]['skills_dict']:
                        self.employee_profiles[emp_name]['skills_dict'][skill_name] = {
                            'skill': skill_name,
                            'skillset': row.get('Skill_Set_Name', ''),
                            'rating': proficiency_rating,
                            'rating_text': text_rating
                        }
                    else:
                        # Keep the higher rating if duplicate
                        existing_rating = self.employee_profiles[emp_name]['skills_dict'][skill_name]['rating']
                        if proficiency_rating > existing_rating:
                            self.employee_profiles[emp_name]['skills_dict'][skill_name]['rating'] = proficiency_rating
                            self.employee_profiles[emp_name]['skills_dict'][skill_name]['rating_text'] = text_rating

        # Convert skills_dict to skills list for backward compatibility
        for emp_name, profile in self.employee_profiles.items():
            profile['skills'] = list(profile['skills_dict'].values())
            del profile['skills_dict']  # Remove the temporary dict

    def calculate_matching_resources(self, skills_list, limit=30):
        """
        Calculate matching resources from a list of skills.
        Returns a dict of {resource_name: {count, max_rating, skills}} sorted by match quality.

        Args:
            skills_list: List of skill names to match resources against
            limit: Maximum number of resources to return (default 30)

        Returns:
            Dictionary of top matching resources
        """
        resource_scores = defaultdict(lambda: {'count': 0, 'max_rating': 0, 'skills': []})

        for skill in skills_list:
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

        return dict(sorted_resources[:limit])

    def create_pl_mapping(self):
        """Create mapping between opportunity PLs and service PLs"""
        self.pl_mapping = PL_MAPPING.copy()

    def build_complete_chain(self):
        """Build the complete chain mappings"""

        # STEP 1: OPPORTUNITY → PRODUCT LINE (Supporting Multiple PLs)
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

            # If opportunity already exists, append the PL to the list
            if opp_id not in self.opportunity_to_pl:
                self.opportunity_to_pl[opp_id] = {
                    'pl_codes': [],  # Changed to list
                    'pl_fulls': [],  # Changed to list
                    'opportunity_name': row['Opportunity Name'],
                    'account': row['Account Name'],
                    'value': row['TCV USD'],
                    'stage': row.get('Sales Stage', 'Unknown'),
                    'forecast': row.get('Forecast Category', 'Unknown')
                }

            # Add this PL to the opportunity's list of PLs
            self.opportunity_to_pl[opp_id]['pl_codes'].append(pl_code)
            self.opportunity_to_pl[opp_id]['pl_fulls'].append(pl_full)
            # Accumulate TCV USD if same opportunity has multiple PLs
            if len(self.opportunity_to_pl[opp_id]['pl_codes']) > 1:
                self.opportunity_to_pl[opp_id]['value'] += row['TCV USD']

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
        """Get the complete chain for an opportunity (supports multiple PLs)"""
        if opportunity_id not in self.opportunity_to_pl:
            return None

        opp_data = self.opportunity_to_pl[opportunity_id]
        pl_codes = opp_data['pl_codes']  # Now a list

        chain = {
            'opportunity': opp_data,
            'product_lines': pl_codes,  # Changed to list
            'services': set(),
            'skillsets': set(),
            'skills': set(),
            'resources': {}
        }

        # Aggregate services from ALL Product Lines
        for pl_code in pl_codes:
            pl_services = self.pl_to_services.get(pl_code, set())
            chain['services'].update(pl_services)

        # Limit services for performance
        chain['services'] = list(chain['services'])[:20]

        # Get skillsets from services
        for service in chain['services']:
            chain['skillsets'].update(self.service_to_skillsets.get(service, set()))

        chain['skillsets'] = list(chain['skillsets'])[:30]

        # Get skills from skillsets
        for skillset in chain['skillsets']:
            chain['skills'].update(self.skillset_to_skills.get(skillset, set()))

        chain['skills'] = list(chain['skills'])[:75]

        # Get resources from skills using the helper method
        chain['resources'] = self.calculate_matching_resources(chain['skills'], limit=30)

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

        # Add PL nodes (multiple PLs supported)
        pl_nodes = []
        for pl_code in chain['product_lines']:
            pl_label = pl_code
            labels.append(pl_label)
            customdata.append(f"Product Line: {pl_code}")
            node_index[pl_label] = current_idx
            colors.append('#764ba2')
            pl_nodes.append(pl_label)
            current_idx += 1

            # Link opportunity to this PL
            sources.append(node_index[opp_short])
            targets.append(node_index[pl_label])
            # Distribute value across all PLs
            values.append(chain['opportunity']['value'] / len(chain['product_lines']))

        # Add services (limited) - connect to all PLs
        for service in chain['services'][:3]:
            # Shorten service name intelligently
            service_short = service[:30] + "..." if len(service) > 30 else service
            if service_short not in node_index:
                labels.append(service_short)
                customdata.append(f"Service: {service}")
                node_index[service_short] = current_idx
                colors.append('#01a982')
                current_idx += 1

            # Connect each service to all PL nodes (distributed)
            for pl_label in pl_nodes:
                sources.append(node_index[pl_label])
                targets.append(node_index[service_short])
                values.append(chain['opportunity']['value'] / (len(chain['services'][:3]) * len(pl_nodes)))

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
                values.append(chain['opportunity']['value'] / (len(chain['services'][:3]) * len(pl_nodes) * 2))

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
            <h1>🗄️ Complete Opportunity-to-Resource Chain (Database Edition)</h1>
            <p>Full chain: Opportunity → PL → Services → Skillsets → Skills → Resources (with drill-down)</p>
            <p style="color: #00d26a; font-size: 14px;">⚡ Powered by SQLite Database for Enhanced Performance</p>
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

        # Initialize session state for active tab if not exists
        if 'active_tab' not in st.session_state:
            st.session_state.active_tab = 0

        # Create tabs for different views with on_change callback
        tab_names = ["📊 Overview", "🔗 Chain Analysis", "👥 Resources", "🔍 Search & Filter"]

        # Use radio buttons styled as tabs for better control
        selected_tab = st.radio(
            "Navigation",
            options=range(len(tab_names)),
            format_func=lambda x: tab_names[x],
            index=st.session_state.active_tab,
            horizontal=True,
            label_visibility="collapsed",
            key="tab_selector"
        )

        st.session_state.active_tab = selected_tab

        if selected_tab == 0:
            self.render_overview_tab()
        elif selected_tab == 1:
            self.render_chain_analysis_tab()
        elif selected_tab == 2:
            self.render_resources_tab()
        elif selected_tab == 3:
            self.render_search_filter_tab()

    def render_overview_tab(self):
        """Render the Overview tab"""
        st.header("📊 Platform Overview")

        # Summary statistics
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Opportunity Distribution")
            st.caption("💡 Click any row to view its complete chain")

            # Top opportunities by value
            top_opps = self.opportunity_summary.nlargest(10, 'TCV USD').copy()
            top_opps['TCV USD (M)'] = top_opps['TCV USD'].apply(self.format_currency_millions)

            # Display as interactive dataframe
            event = st.dataframe(
                top_opps[['HPE Opportunity Id', 'Opportunity Name', 'Product Line Code', 'Product Line Description', 'TCV USD (M)']],
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key="overview_opportunities_table"
            )

            # Check if a row was selected
            if event.selection and event.selection.rows:
                selected_row_idx = event.selection.rows[0]
                # Get the actual dataframe index
                selected_opp_id = top_opps.iloc[selected_row_idx]['HPE Opportunity Id']

                # Store selected opportunity and switch to Chain Analysis tab
                st.session_state.selected_opportunity_id = selected_opp_id
                st.session_state.active_tab = 1
                st.rerun()

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
                if any(pl_code in self.pl_mapping for pl_code in data['pl_codes'])
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

    def get_filtered_chain(self, opportunity_id, selected_services=None, selected_skillsets=None, selected_skills=None):
        """Get chain with cascading filters applied"""
        # Get the base chain
        chain = self.get_complete_chain(opportunity_id)
        if not chain:
            return None

        # Apply cascading filters
        if selected_services:
            # Filter to selected services only
            chain['services'] = [s for s in chain['services'] if s in selected_services]

            # Cascade: Update skillsets based on selected services
            chain['skillsets'] = set()
            for service in chain['services']:
                chain['skillsets'].update(self.service_to_skillsets.get(service, set()))
            chain['skillsets'] = list(chain['skillsets'])

        if selected_skillsets:
            # Filter to selected skillsets only
            chain['skillsets'] = [s for s in chain['skillsets'] if s in selected_skillsets]

            # Cascade: Update skills based on selected skillsets
            chain['skills'] = set()
            for skillset in chain['skillsets']:
                chain['skills'].update(self.skillset_to_skills.get(skillset, set()))
            chain['skills'] = list(chain['skills'])

        if selected_skills:
            # Filter to selected skills only
            chain['skills'] = [s for s in chain['skills'] if s in selected_skills]

            # Cascade: Update resources based on selected skills using helper method
            chain['resources'] = self.calculate_matching_resources(chain['skills'], limit=30)

        return chain

    def render_chain_analysis_tab(self):
        """Render the Chain Analysis tab with cascading filters"""
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

        # Determine default index based on pre-selected opportunity from Overview
        default_index = 0
        if 'selected_opportunity_id' in st.session_state:
            # Find the index of the pre-selected opportunity
            for idx, (label, opp_id) in enumerate(opp_options.items()):
                if opp_id == st.session_state.selected_opportunity_id:
                    default_index = idx
                    break
            # Clear the selected opportunity from session state after using it
            del st.session_state.selected_opportunity_id

        selected = st.selectbox(
            "Choose an opportunity to explore the complete chain",
            options=list(opp_options.keys()),
            index=default_index,
            key="opp_selector"
        )

        if selected:
            opp_id = opp_options[selected]

            # Initialize session state for this opportunity if changed
            if 'current_opp_id' not in st.session_state or st.session_state.current_opp_id != opp_id:
                st.session_state.current_opp_id = opp_id
                st.session_state.selected_services = []
                st.session_state.selected_skillsets = []
                st.session_state.selected_skills = []

            # Get the base chain (unfiltered)
            base_chain = self.get_complete_chain(opp_id)

            if base_chain:
                # Display the chain steps - COMPACT 6-COLUMN LAYOUT
                st.header("📊 Chain Analysis")

                # Initialize session state with all PLs selected by default
                if 'selected_pls' not in st.session_state or st.session_state.get('current_opp_id') != opp_id:
                    st.session_state.selected_pls = base_chain['product_lines']

                # Calculate cascading data
                # Services from selected PLs
                available_services = set()
                for pl_code in st.session_state.selected_pls:
                    available_services.update(self.pl_to_services.get(pl_code, set()))
                available_services = sorted(list(available_services))

                # Skillsets from selected services (or all if none selected)
                if st.session_state.selected_services:
                    available_skillsets = set()
                    for service in st.session_state.selected_services:
                        available_skillsets.update(self.service_to_skillsets.get(service, set()))
                    available_skillsets = sorted(list(available_skillsets))
                else:
                    available_skillsets = set()
                    for service in available_services:
                        available_skillsets.update(self.service_to_skillsets.get(service, set()))
                    available_skillsets = sorted(list(available_skillsets))

                # Skills from selected skillsets (or all if none selected)
                if st.session_state.selected_skillsets:
                    available_skills = set()
                    for skillset in st.session_state.selected_skillsets:
                        available_skills.update(self.skillset_to_skills.get(skillset, set()))
                    available_skills = sorted(list(available_skills))
                else:
                    available_skills = set()
                    for skillset in available_skillsets:
                        available_skills.update(self.skillset_to_skills.get(skillset, set()))
                    available_skills = sorted(list(available_skills))

                # Resources from selected skills - IMPORTANT: Actually filter!
                if st.session_state.selected_skills:
                    # Filter to only resources with selected skills
                    skills_to_match = set(st.session_state.selected_skills)
                elif st.session_state.selected_skillsets:
                    # Use skills from selected skillsets
                    skills_to_match = set(available_skills)
                elif st.session_state.selected_services:
                    # Use skills from selected services
                    skills_to_match = set(available_skills)
                else:
                    # Use all skills from selected PLs
                    skills_to_match = set(available_skills)

                # Calculate matching resources using helper method
                matched_resources = self.calculate_matching_resources(list(skills_to_match), limit=30)

                # COMPACT 6-STEP HEADER ROW
                steps = st.columns(6)

                with steps[0]:
                    st.markdown("""
                    <div class="chain-flow">
                    1️⃣ Opportunity
                    </div>
                    """, unsafe_allow_html=True)
                    st.info(opp_id)
                    st.caption(base_chain['opportunity']['opportunity_name'][:40])

                with steps[1]:
                    st.markdown("""
                    <div class="chain-flow">
                    2️⃣ Product Lines
                    </div>
                    """, unsafe_allow_html=True)
                    pl_fulls = base_chain['opportunity'].get('pl_fulls', base_chain['product_lines'])

                    # Interactive dropdown for PLs
                    with st.popover(f"{len(st.session_state.selected_pls)} PL{'s' if len(st.session_state.selected_pls) > 1 else ''}", use_container_width=True):
                        pl_options = {f"{pl_code} - {pl_full.split(' - ')[1] if ' - ' in pl_full else pl_code}": pl_code
                                      for pl_code, pl_full in zip(base_chain['product_lines'], pl_fulls)}

                        selected_pl_labels = st.multiselect(
                            f"Select PLs ({len(base_chain['product_lines'])} total)",
                            options=list(pl_options.keys()),
                            default=[k for k, v in pl_options.items() if v in st.session_state.selected_pls],
                            key="pl_multiselect"
                        )
                        st.session_state.selected_pls = [pl_options[label] for label in selected_pl_labels] if selected_pl_labels else base_chain['product_lines']

                with steps[2]:
                    st.markdown("""
                    <div class="chain-flow">
                    3️⃣ Services
                    </div>
                    """, unsafe_allow_html=True)

                    with st.popover(f"{len(available_services)} services", use_container_width=True):
                        st.session_state.selected_services = st.multiselect(
                            f"Select Services ({len(available_services)} from PLs)",
                            options=available_services,
                            default=st.session_state.selected_services if st.session_state.selected_services else [],
                            key="services_multiselect"
                        )

                with steps[3]:
                    st.markdown("""
                    <div class="chain-flow">
                    4️⃣ Skillsets
                    </div>
                    """, unsafe_allow_html=True)

                    with st.popover(f"{len(available_skillsets)} skillsets", use_container_width=True):
                        st.session_state.selected_skillsets = st.multiselect(
                            f"Select Skillsets ({len(available_skillsets)} from services)",
                            options=available_skillsets,
                            default=st.session_state.selected_skillsets if st.session_state.selected_skillsets else [],
                            key="skillsets_multiselect"
                        )

                with steps[4]:
                    st.markdown("""
                    <div class="chain-flow">
                    5️⃣ Skills
                    </div>
                    """, unsafe_allow_html=True)

                    with st.popover(f"{len(available_skills)} skills", use_container_width=True):
                        st.session_state.selected_skills = st.multiselect(
                            f"Select Skills ({len(available_skills)} from skillsets)",
                            options=available_skills,
                            default=st.session_state.selected_skills if st.session_state.selected_skills else [],
                            key="skills_multiselect"
                        )

                with steps[5]:
                    st.markdown("""
                    <div class="chain-flow">
                    6️⃣ Resources
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption(f"{len(matched_resources)} matched")

                # Reset button
                if st.session_state.selected_pls != base_chain['product_lines'] or st.session_state.selected_services or st.session_state.selected_skillsets or st.session_state.selected_skills:
                    if st.button("🔄 Reset All Filters"):
                        st.session_state.selected_pls = base_chain['product_lines']
                        st.session_state.selected_services = []
                        st.session_state.selected_skillsets = []
                        st.session_state.selected_skills = []
                        st.rerun()

                # Build the final chain object for downstream use
                chain = {
                    'opportunity': base_chain['opportunity'],
                    'product_lines': st.session_state.selected_pls,
                    'services': st.session_state.selected_services if st.session_state.selected_services else available_services,
                    'skillsets': st.session_state.selected_skillsets if st.session_state.selected_skillsets else available_skillsets,
                    'skills': st.session_state.selected_skills if st.session_state.selected_skills else available_skills,
                    'resources': matched_resources
                }

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

            col1, col2 = st.columns(2)
            with col1:
                opp_search = st.text_input("Opportunity name contains:", placeholder="Enter keywords...")
            with col2:
                pl_filter = st.selectbox("Product Line", ["All"] + self.opportunity_summary['Product Line'].unique().tolist())

            # Apply filters
            filtered = self.opportunity_summary.copy()
            if opp_search:
                filtered = filtered[filtered['Opportunity Name'].str.contains(opp_search, case=False, na=False)]
            if pl_filter != "All":
                filtered = filtered[filtered['Product Line'] == pl_filter]

            st.write(f"Found {len(filtered)} opportunities")
            if len(filtered) > 0:
                filtered_display = filtered.head(50).copy()
                filtered_display['TCV USD (M)'] = filtered_display['TCV USD'].apply(self.format_currency_millions)
                st.dataframe(filtered_display[['HPE Opportunity Id', 'Opportunity Name', 'Product Line Code', 'Product Line Description',
                                      'TCV USD (M)']],
                            use_container_width=True, hide_index=True)

        elif search_type == "Services":
            st.subheader("Search Services")
            service_search = st.text_input("Service name contains:", placeholder="Enter keywords...")

            all_services = set()
            for services in self.pl_to_services.values():
                all_services.update(services)

            if service_search:
                matching = [s for s in all_services if service_search.lower() in s.lower()]
                st.write(f"Found {len(matching)} services")
                for service in matching[:20]:
                    skillsets = self.service_to_skillsets.get(service, set())
                    st.write(f"• **{service}** ({len(skillsets)} skillsets)")
            else:
                st.info("Enter a search term to find services")

        elif search_type == "Skills":
            st.subheader("Search Skills")
            skill_search = st.text_input("Skill name contains:", placeholder="Enter keywords...")

            if skill_search:
                matching = [(skill, resources) for skill, resources in self.skill_to_resources.items()
                          if skill_search.lower() in skill.lower()]
                st.write(f"Found {len(matching)} skills")
                for skill, resources in matching[:20]:
                    st.write(f"• **{skill[:60]}** ({len(resources)} resources)")
            else:
                st.info("Enter a search term to find skills")

        elif search_type == "Resources":
            st.subheader("Search Resources")

            col1, col2 = st.columns(2)
            with col1:
                name_search = st.text_input("Resource name contains:", placeholder="Enter name...")
                location_filter = st.selectbox("Location",
                    ["All"] + sorted(list(set(p.get('location', 'Unknown') for p in self.employee_profiles.values()))))
            with col2:
                skill_search = st.text_input("Has skill:", placeholder="Enter skill...")
                min_skills = st.number_input("Minimum skills", min_value=0, value=0)

            # Apply filters
            matching = []
            for name, profile in self.employee_profiles.items():
                if name_search and name_search.lower() not in name.lower():
                    continue
                if location_filter != "All" and profile.get('location') != location_filter:
                    continue
                if skill_search:
                    skills_text = " ".join([s['skill'] for s in profile['skills']])
                    if skill_search.lower() not in skills_text.lower():
                        continue
                if len(profile['skills']) < min_skills:
                    continue
                matching.append((name, profile))

            st.write(f"Found {len(matching)} resources")
            for name, profile in matching[:20]:
                st.write(f"• **{name}** - {profile.get('location', 'Unknown')} ({len(profile['skills'])} skills)")


# Main execution
if __name__ == "__main__":
    platform = CompleteOpportunityChainDB()
    platform.render()