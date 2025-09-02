"""
Improved Smart Search - More Accurate Skill Matching
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os

# Setup paths
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def smart_search_resources(data, search_query, search_type='skills', top_k=10):
    """
    Improved search that focuses on actual skill matching.
    
    Args:
        data: Dictionary with resources data
        search_query: User's search query
        search_type: 'skills', 'domain', 'focus_area', or 'combined'
        top_k: Number of results to return
    """
    
    if 'resources_corrected' not in data:
        st.error("Resource data not available")
        return []
    
    resources_df = data['resources_corrected']
    search_terms = search_query.lower().split()
    
    # Remove common words that shouldn't affect search
    stop_words = {'for', 'and', 'the', 'with', 'need', 'want', 'looking', 'find', 'get', 'have'}
    search_terms = [term for term in search_terms if term not in stop_words and len(term) > 2]
    
    matches = []
    
    for _, person in resources_df.iterrows():
        # Get person's data
        skills = str(person.get('all_skills', '')).lower()
        domain = str(person.get('primary_domain', '')).lower()
        name = person.get('resource_name', 'Unknown')
        
        # Initialize scoring
        score = 0
        matched_terms = []
        skill_matches = []
        
        # More flexible skill matching
        for term in search_terms:
            term_lower = term.lower()
            
            # Count occurrences in skills
            skill_occurrences = skills.lower().count(term_lower)
            
            if skill_occurrences > 0:
                # Give points based on how many times the term appears
                score += min(skill_occurrences * 5, 20)  # Cap at 20 points per term
                matched_terms.append(term)
                skill_matches.append(term)
            
            # Check domain match (lower priority)
            if term_lower in domain:
                score += 2
                if term not in matched_terms:
                    matched_terms.append(f"{term} (domain)")
        
        # Only include if there's a match
        if score > 0:
            # Extract specific matching skills
            skill_list = skills.split(';') if ';' in skills else skills.split(',')
            matching_skills = []
            for skill in skill_list:
                skill_clean = skill.strip().lower()
                for term in search_terms:
                    if term in skill_clean:
                        matching_skills.append(skill.strip())
                        break
            
            matches.append({
                'name': name,
                'score': score,
                'matched_terms': list(set(matched_terms)),
                'matching_skills': matching_skills[:5],  # Top 5 matching skills
                'all_skills': person.get('all_skills', ''),
                'skill_count': person.get('skill_count', 0),
                'rating': person.get('avg_rating', 0),
                'domain': person.get('primary_domain', ''),
                'city': person.get('city', ''),
                'manager': person.get('manager', ''),
                'skill_matches': skill_matches
            })
    
    # Sort by score (highest first)
    matches.sort(key=lambda x: x['score'], reverse=True)
    
    return matches[:top_k]

def show_improved_smart_search(data):
    """Enhanced smart search interface."""
    
    st.header("🔍 Smart Resource Search")
    
    # Search interface
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        search_query = st.text_input(
            "Search for skills, technologies, or expertise:",
            placeholder="e.g., Python, AWS, Machine Learning, Kubernetes",
            help="Enter one or more skills separated by spaces"
        )
    
    with col2:
        search_type = st.selectbox(
            "Search Type:",
            ["Skills Only", "Include Domains", "Combined"],
            help="Choose how to search"
        )
    
    with col3:
        num_results = st.number_input(
            "Results:",
            min_value=5,
            max_value=50,
            value=10,
            step=5
        )
    
    if search_query:
        # Show what we're searching for
        st.info(f"🔍 Searching for: **{search_query}**")
        
        # Perform search
        with st.spinner("Searching resources..."):
            matches = smart_search_resources(
                data, 
                search_query,
                search_type=search_type.lower().replace(' ', '_'),
                top_k=num_results
            )
        
        if matches:
            st.success(f"✅ Found {len(matches)} matching professionals!")
            
            # Display results
            st.markdown("### 🎯 Search Results")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_rating = np.mean([m['rating'] for m in matches])
                st.metric("Avg Rating", f"⭐ {avg_rating:.1f}")
            
            with col2:
                avg_skills = np.mean([m['skill_count'] for m in matches])
                st.metric("Avg Skills", f"{avg_skills:.0f}")
            
            with col3:
                domains = list(set([m['domain'] for m in matches]))
                st.metric("Domains", f"{len(domains)}")
            
            with col4:
                cities = list(set([m['city'] for m in matches]))
                st.metric("Locations", f"{len(cities)}")
            
            # Detailed results
            for i, match in enumerate(matches, 1):
                # Determine match quality
                if match['score'] >= 30:
                    quality = "🟢 Excellent Match"
                    color = "#d4edda"
                elif match['score'] >= 15:
                    quality = "🟡 Good Match"
                    color = "#fff3cd"
                else:
                    quality = "🔵 Partial Match"
                    color = "#d1ecf1"
                
                with st.expander(f"#{i} - {match['name']} | Score: {match['score']} | {quality}"):
                    # Use columns for better layout
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("### 👤 Professional Profile")
                        st.markdown(f"**Domain:** {match['domain']}")
                        st.markdown(f"**Location:** {match['city']}")
                        st.markdown(f"**Manager:** {match['manager']}")
                        st.markdown(f"**Total Skills:** {match['skill_count']}")
                        st.markdown(f"**Rating:** ⭐ {match['rating']:.1f}/5")
                        
                        # Show matching skills specifically
                        if match['matching_skills']:
                            st.markdown("### 🎯 Matching Skills")
                            for skill in match['matching_skills']:
                                if any(term in skill.lower() for term in match['skill_matches']):
                                    st.markdown(f"• **{skill}** ✅")
                                else:
                                    st.markdown(f"• {skill}")
                        
                        # Show why they matched
                        st.markdown("### 🔍 Match Details")
                        st.markdown(f"**Matched on:** {', '.join(match['matched_terms'])}")
                    
                    with col2:
                        # Visual score representation
                        st.markdown("### 📊 Match Score")
                        
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=match['score'],
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Match Strength"},
                            delta={'reference': 15},  # Good match threshold
                            gauge={
                                'axis': {'range': [None, 50]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 10], 'color': "lightgray"},
                                    {'range': [10, 20], 'color': "gray"},
                                    {'range': [20, 50], 'color': "lightgreen"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 30  # Excellent match threshold
                                }
                            }
                        ))
                        fig.update_layout(height=200, margin=dict(t=30, b=0, l=0, r=0))
                        st.plotly_chart(fig, use_container_width=True, key=f"gauge_{i}")
                        
                        # Match breakdown
                        st.markdown("### 📈 Score Breakdown")
                        skill_score = len(match['skill_matches']) * 10
                        domain_score = match['score'] - skill_score
                        
                        st.progress(skill_score / match['score'] if match['score'] > 0 else 0)
                        st.caption(f"Skill matches: {skill_score} pts")
                        
                        if domain_score > 0:
                            st.progress(domain_score / match['score'] if match['score'] > 0 else 0)
                            st.caption(f"Domain match: {domain_score} pts")
                    
                    # Full skills preview
                    st.markdown("### 📄 All Skills")
                    with st.container():
                        skills_text = match['all_skills']
                        if len(skills_text) > 500:
                            st.text_area("", value=skills_text, height=100, disabled=True, key=f"skills_{i}")
                        else:
                            st.info(skills_text)
        else:
            st.warning(f"No matches found for '{search_query}'. Try different search terms.")
            
            # Suggestions
            st.markdown("### 💡 Search Tips")
            st.markdown("""
            - Try single skills: `Python`, `Java`, `AWS`
            - Use technology names: `Kubernetes`, `Docker`, `Jenkins`
            - Search for certifications: `CISSP`, `PMP`, `AWS Certified`
            - Look for domains: `Security`, `Cloud`, `Data`
            """)

# Standalone test
if __name__ == "__main__":
    st.set_page_config(page_title="Smart Search", page_icon="🔍", layout="wide")
    
    # Load or create sample data
    @st.cache_data
    def load_data():
        try:
            # Try to load real data
            corrected_df = pd.read_parquet('data_processed/resources_deduplicated.parquet')
            return {'resources_corrected': corrected_df}
        except:
            # Create sample data
            return {
                'resources_corrected': pd.DataFrame({
                    'resource_name': ['John Python Expert', 'Jane Cloud Architect', 'Bob Data Scientist'],
                    'all_skills': [
                        'Python; Django; Flask; Machine Learning; TensorFlow; Pandas; NumPy',
                        'AWS; Azure; Kubernetes; Docker; Terraform; Python; CI/CD',
                        'Python; R; SQL; Machine Learning; Statistics; Data Analysis; Spark'
                    ],
                    'skill_count': [7, 7, 7],
                    'avg_rating': [4.5, 4.2, 4.8],
                    'primary_domain': ['Data Engineering', 'Cloud Platform', 'Data Science'],
                    'city': ['Bangalore', 'Austin', 'London'],
                    'manager': ['Manager A', 'Manager B', 'Manager C']
                })
            }
    
    data = load_data()
    show_improved_smart_search(data)