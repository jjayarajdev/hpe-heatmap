"""
Real-time resource recommendation engine.

This module provides intelligent resource recommendations based on skills,
experience, availability, and other factors with transparent scoring.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity

from .utils import config, logger, Timer
from .match import SmartMatcher
from .taxonomy import TaxonomyQuery


class ResourceRecommender:
    """Intelligent resource recommendation system."""
    
    def __init__(self, taxonomy_query: TaxonomyQuery, matcher: SmartMatcher):
        self.taxonomy = taxonomy_query
        self.matcher = matcher
        self.resources_df = None
        self.resource_skills_df = None
        self.resource_profiles = {}
        
    def initialize(self, resources_df: pd.DataFrame, 
                  resource_skills_df: pd.DataFrame = None):
        """Initialize with resource data."""
        logger.info("Initializing resource recommender")
        
        self.resources_df = resources_df.copy()
        self.resource_skills_df = resource_skills_df.copy() if resource_skills_df is not None else None
        
        # Build resource profiles
        self._build_resource_profiles()
        
        logger.info(f"Initialized with {len(self.resource_profiles)} resource profiles")
    
    def _build_resource_profiles(self):
        """Build comprehensive profiles for each resource."""
        logger.info("Building resource profiles")
        
        # Group by resource
        if 'resource_id' in self.resources_df.columns:
            resource_groups = self.resources_df.groupby('resource_id')
        elif 'resource_name' in self.resources_df.columns:
            resource_groups = self.resources_df.groupby('resource_name')
        else:
            logger.error("No resource identifier column found")
            return
        
        for resource_id, group in resource_groups:
            profile = self._create_resource_profile(resource_id, group)
            self.resource_profiles[resource_id] = profile
    
    def _create_resource_profile(self, resource_id: str, resource_data: pd.DataFrame) -> Dict[str, Any]:
        """Create comprehensive profile for a single resource."""
        # Basic information
        first_row = resource_data.iloc[0]
        
        profile = {
            "resource_id": resource_id,
            "name": first_row.get("resource_name", str(resource_id)),
            "email": first_row.get("email", ""),
            "manager": first_row.get("manager", ""),
            "practice": first_row.get("practice", ""),
            "location": first_row.get("location", ""),
            "domain": first_row.get("domain", ""),
            "subdomain": first_row.get("subdomain", "")
        }
        
        # Skills analysis
        skills_info = self._analyze_resource_skills(resource_data)
        profile.update(skills_info)
        
        # Experience and performance metrics
        experience_info = self._calculate_experience_metrics(resource_data)
        profile.update(experience_info)
        
        # Availability and utilization (placeholder)
        profile["availability"] = self._estimate_availability(resource_data)
        
        return profile
    
    def _analyze_resource_skills(self, resource_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze skills for a resource."""
        skills_info = {
            "skills": [],
            "skill_count": 0,
            "avg_rating": 0.0,
            "max_rating": 0.0,
            "skill_categories": [],
            "expertise_areas": []
        }
        
        # Extract skills from resource data
        skill_columns = [col for col in resource_data.columns 
                        if any(term in col.lower() for term in ['skill', 'certification'])]
        
        if not skill_columns:
            return skills_info
        
        # Process each skill record
        for _, row in resource_data.iterrows():
            for skill_col in skill_columns:
                skill_name = str(row[skill_col]).strip()
                
                if skill_name and skill_name.lower() not in ['nan', 'none', '']:
                    skill_info = {
                        "name": skill_name,
                        "rating": self._extract_rating(row),
                        "category": str(row.get("category", "")).strip(),
                        "evaluation_date": self._extract_date(row)
                    }
                    
                    skills_info["skills"].append(skill_info)
        
        # Calculate aggregated metrics
        if skills_info["skills"]:
            ratings = [s["rating"] for s in skills_info["skills"] if s["rating"] > 0]
            
            skills_info["skill_count"] = len(skills_info["skills"])
            skills_info["avg_rating"] = np.mean(ratings) if ratings else 0.0
            skills_info["max_rating"] = max(ratings) if ratings else 0.0
            
            # Extract categories
            categories = [s["category"] for s in skills_info["skills"] if s["category"]]
            skills_info["skill_categories"] = list(set(categories))
            
            # Identify expertise areas (skills with high ratings)
            expertise_threshold = 4.0  # Assuming 5-point scale
            expertise_skills = [s["name"] for s in skills_info["skills"] 
                              if s["rating"] >= expertise_threshold]
            skills_info["expertise_areas"] = expertise_skills
        
        return skills_info
    
    def _extract_rating(self, row: pd.Series) -> float:
        """Extract rating value from row."""
        rating_columns = [col for col in row.index 
                         if any(term in col.lower() for term in ['rating', 'proficiency', 'level'])]
        
        for col in rating_columns:
            try:
                value = row[col]
                
                # Handle different rating formats
                if isinstance(value, (int, float)):
                    return float(value)
                
                elif isinstance(value, str):
                    # Extract numeric rating from string
                    rating_map = {
                        'basic': 1.0, 'beginner': 1.0, 'novice': 1.0,
                        'intermediate': 2.0, 'competent': 2.5,
                        'advanced': 3.0, 'proficient': 3.5,
                        'expert': 4.0, 'master': 5.0, 'certified': 4.0
                    }
                    
                    value_lower = value.lower().strip()
                    
                    # Try direct mapping
                    if value_lower in rating_map:
                        return rating_map[value_lower]
                    
                    # Try extracting number
                    import re
                    numbers = re.findall(r'\d+\.?\d*', value)
                    if numbers:
                        return float(numbers[0])
            
            except:
                continue
        
        return 0.0  # Default rating
    
    def _extract_date(self, row: pd.Series) -> Optional[datetime]:
        """Extract evaluation date from row."""
        date_columns = [col for col in row.index 
                       if any(term in col.lower() for term in ['date', 'time', 'evaluation'])]
        
        for col in date_columns:
            try:
                value = row[col]
                if pd.notna(value):
                    return pd.to_datetime(value)
            except:
                continue
        
        return None
    
    def _calculate_experience_metrics(self, resource_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate experience-related metrics."""
        experience_info = {
            "years_experience": 0.0,
            "skill_freshness": 0.0,
            "learning_velocity": 0.0,
            "versatility_score": 0.0
        }
        
        # Calculate years of experience (based on evaluation dates)
        dates = []
        for _, row in resource_data.iterrows():
            date = self._extract_date(row)
            if date:
                dates.append(date)
        
        if dates:
            date_range = max(dates) - min(dates)
            experience_info["years_experience"] = date_range.days / 365.25
            
            # Skill freshness (how recent are the evaluations)
            latest_date = max(dates)
            days_since_latest = (datetime.now() - latest_date).days
            experience_info["skill_freshness"] = max(0, 1 - (days_since_latest / 365))
        
        # Versatility (number of different skill categories)
        categories = set()
        for _, row in resource_data.iterrows():
            category = str(row.get("category", "")).strip()
            if category and category.lower() != 'nan':
                categories.add(category)
        
        experience_info["versatility_score"] = len(categories) / 10.0  # Normalize to 0-1
        
        return experience_info
    
    def _estimate_availability(self, resource_data: pd.DataFrame) -> float:
        """Estimate resource availability (placeholder implementation)."""
        # In a real system, this would consider:
        # - Current project assignments
        # - Planned time off
        # - Capacity planning data
        # - Historical utilization
        
        # For now, return a random availability between 0.3 and 1.0
        np.random.seed(hash(str(resource_data.iloc[0].get("resource_name", ""))) % 2**32)
        return np.random.uniform(0.3, 1.0)
    
    def recommend_resources(self, request_text: str, n: int = 5) -> List[Dict[str, Any]]:
        """
        Recommend top resources for a given request.
        
        Args:
            request_text: Description of the resource request
            n: Number of resources to recommend
            
        Returns:
            List of resource recommendations with scores and rationale
        """
        logger.info(f"Generating {n} resource recommendations for: '{request_text[:100]}...'")
        
        if not self.resource_profiles:
            logger.error("No resource profiles available")
            return []
        
        # First, match request to skills/services
        skill_matches = self.matcher.match_skills_and_services(request_text, top_k=20)
        
        # Extract relevant skills
        relevant_skills = set()
        for entity_type, matches in skill_matches.items():
            for match in matches[:10]:  # Top 10 from each type
                relevant_skills.add(match["name"].lower())
        
        if not relevant_skills:
            logger.warning("No relevant skills found for request")
            return []
        
        # Score all resources
        resource_scores = []
        
        for resource_id, profile in self.resource_profiles.items():
            score_info = self._score_resource_for_request(
                profile, relevant_skills, request_text
            )
            
            if score_info["total_score"] > 0.1:  # Minimum threshold
                resource_scores.append({
                    "resource_id": resource_id,
                    "name": profile["name"],
                    "email": profile["email"],
                    "practice": profile["practice"],
                    "location": profile["location"],
                    "score": score_info["total_score"],
                    "score_breakdown": score_info["breakdown"],
                    "rationale": self._generate_recommendation_rationale(score_info),
                    "matching_skills": score_info["matching_skills"],
                    "profile": profile
                })
        
        # Sort by score and return top n unique resources
        resource_scores.sort(key=lambda x: x["score"], reverse=True)
        
        # Deduplicate by resource name and email to avoid duplicates
        seen_resources = set()
        unique_recommendations = []
        
        for rec in resource_scores:
            resource_key = (rec["name"].lower(), rec["email"].lower())
            if resource_key not in seen_resources:
                seen_resources.add(resource_key)
                unique_recommendations.append(rec)
                
                if len(unique_recommendations) >= n:
                    break
        
        logger.info(f"Generated {len(resource_scores)} candidate recommendations, returning {len(unique_recommendations)} unique")
        
        return unique_recommendations
    
    def _score_resource_for_request(self, profile: Dict, relevant_skills: set, 
                                  request_text: str) -> Dict[str, Any]:
        """Score a resource for a specific request."""
        breakdown = {}
        
        # 1. Skill match score
        skill_match_score, matching_skills = self._calculate_skill_match_score(
            profile, relevant_skills
        )
        breakdown["skill_match"] = skill_match_score
        
        # 2. Experience score
        experience_score = self._calculate_experience_score(profile)
        breakdown["experience"] = experience_score
        
        # 3. Recency score (skill freshness)
        recency_score = profile.get("skill_freshness", 0.5)
        breakdown["recency"] = recency_score
        
        # 4. Utilization score (availability)
        utilization_score = profile.get("availability", 0.5)
        breakdown["utilization"] = utilization_score
        
        # Calculate weighted total score
        total_score = (
            config.skill_match_weight * skill_match_score +
            config.experience_weight * experience_score +
            config.recency_weight * recency_score +
            config.utilization_weight * utilization_score
        )
        
        return {
            "total_score": total_score,
            "breakdown": breakdown,
            "matching_skills": matching_skills
        }
    
    def _calculate_skill_match_score(self, profile: Dict, 
                                   relevant_skills: set) -> Tuple[float, List[str]]:
        """Calculate how well resource skills match the request."""
        resource_skills = profile.get("skills", [])
        
        if not resource_skills:
            return 0.0, []
        
        matching_skills = []
        skill_scores = []
        
        for skill_info in resource_skills:
            skill_name = skill_info["name"].lower()
            skill_rating = skill_info["rating"]
            
            # Check for exact matches
            if skill_name in relevant_skills:
                match_score = 1.0
                matching_skills.append(skill_info["name"])
            else:
                # Check for partial matches
                match_score = 0.0
                for relevant_skill in relevant_skills:
                    # Word-level overlap
                    skill_words = set(skill_name.split())
                    relevant_words = set(relevant_skill.split())
                    
                    if skill_words & relevant_words:
                        overlap = len(skill_words & relevant_words)
                        total_words = len(skill_words | relevant_words)
                        partial_score = overlap / total_words
                        
                        if partial_score > match_score:
                            match_score = partial_score
                            
                        if partial_score > 0.5:  # Significant overlap
                            matching_skills.append(skill_info["name"])
            
            # Weight by skill rating
            weighted_score = match_score * (skill_rating / 5.0)  # Normalize rating to 0-1
            skill_scores.append(weighted_score)
        
        # Calculate overall skill match score
        if skill_scores:
            # Use average of top 5 skills to avoid penalizing versatile resources
            top_scores = sorted(skill_scores, reverse=True)[:5]
            overall_score = np.mean(top_scores)
        else:
            overall_score = 0.0
        
        return overall_score, matching_skills
    
    def _calculate_experience_score(self, profile: Dict) -> float:
        """Calculate experience score based on various factors."""
        factors = []
        
        # Years of experience
        years_exp = profile.get("years_experience", 0)
        exp_score = min(years_exp / 10.0, 1.0)  # Normalize to 0-1, cap at 10 years
        factors.append(exp_score)
        
        # Average skill rating
        avg_rating = profile.get("avg_rating", 0)
        rating_score = avg_rating / 5.0  # Normalize to 0-1
        factors.append(rating_score)
        
        # Number of expertise areas
        expertise_count = len(profile.get("expertise_areas", []))
        expertise_score = min(expertise_count / 5.0, 1.0)  # Normalize, cap at 5
        factors.append(expertise_score)
        
        # Versatility
        versatility = profile.get("versatility_score", 0)
        factors.append(versatility)
        
        return np.mean(factors) if factors else 0.0
    
    def _generate_recommendation_rationale(self, score_info: Dict) -> List[str]:
        """Generate human-readable rationale for recommendation."""
        rationale = []
        breakdown = score_info["breakdown"]
        
        # Skill match rationale
        skill_score = breakdown["skill_match"]
        if skill_score > 0.8:
            rationale.append("Excellent skill match for requirements")
        elif skill_score > 0.6:
            rationale.append("Good skill alignment with requirements")
        elif skill_score > 0.3:
            rationale.append("Partial skill match with transferable experience")
        else:
            rationale.append("Limited direct skill match but potential for growth")
        
        # Experience rationale
        exp_score = breakdown["experience"]
        if exp_score > 0.8:
            rationale.append("Highly experienced professional")
        elif exp_score > 0.6:
            rationale.append("Solid experience level")
        elif exp_score > 0.3:
            rationale.append("Moderate experience with growth potential")
        
        # Recency rationale
        recency_score = breakdown["recency"]
        if recency_score > 0.8:
            rationale.append("Recently updated skills")
        elif recency_score < 0.3:
            rationale.append("Skills may need refreshing")
        
        # Availability rationale
        util_score = breakdown["utilization"]
        if util_score > 0.8:
            rationale.append("High availability")
        elif util_score > 0.5:
            rationale.append("Moderate availability")
        else:
            rationale.append("Limited availability")
        
        # Matching skills
        matching_skills = score_info["matching_skills"]
        if matching_skills:
            if len(matching_skills) == 1:
                rationale.append(f"Key skill: {matching_skills[0]}")
            else:
                rationale.append(f"Key skills: {', '.join(matching_skills[:3])}")
        
        return rationale
    
    def get_resource_insights(self, resource_id: str) -> Dict[str, Any]:
        """Get detailed insights for a specific resource."""
        if resource_id not in self.resource_profiles:
            return {"error": "Resource not found"}
        
        profile = self.resource_profiles[resource_id]
        
        insights = {
            "profile": profile,
            "strengths": self._identify_strengths(profile),
            "development_areas": self._identify_development_areas(profile),
            "similar_resources": self._find_similar_resources(resource_id, k=5),
            "recommended_opportunities": self._recommend_opportunities_for_resource(profile)
        }
        
        return insights
    
    def _identify_strengths(self, profile: Dict) -> List[str]:
        """Identify key strengths of a resource."""
        strengths = []
        
        # High-rated skills
        expertise_areas = profile.get("expertise_areas", [])
        if expertise_areas:
            strengths.append(f"Expert in: {', '.join(expertise_areas[:3])}")
        
        # High experience
        if profile.get("years_experience", 0) > 5:
            strengths.append(f"Experienced professional ({profile['years_experience']:.1f} years)")
        
        # High versatility
        if profile.get("versatility_score", 0) > 0.7:
            strengths.append("Highly versatile across multiple domains")
        
        # Recent skills
        if profile.get("skill_freshness", 0) > 0.8:
            strengths.append("Recently updated and current skills")
        
        return strengths
    
    def _identify_development_areas(self, profile: Dict) -> List[str]:
        """Identify potential development areas for a resource."""
        development_areas = []
        
        # Low skill freshness
        if profile.get("skill_freshness", 1.0) < 0.5:
            development_areas.append("Consider skill refresher training")
        
        # Limited versatility
        if profile.get("versatility_score", 1.0) < 0.3:
            development_areas.append("Expand into additional technology domains")
        
        # Average ratings could be improved
        avg_rating = profile.get("avg_rating", 5.0)
        if avg_rating < 3.5:
            development_areas.append("Focus on advancing current skills to expert level")
        
        return development_areas
    
    def _find_similar_resources(self, resource_id: str, k: int = 5) -> List[Dict]:
        """Find resources similar to the given resource."""
        if resource_id not in self.resource_profiles:
            return []
        
        target_profile = self.resource_profiles[resource_id]
        target_skills = set(s["name"].lower() for s in target_profile.get("skills", []))
        
        similarities = []
        
        for other_id, other_profile in self.resource_profiles.items():
            if other_id == resource_id:
                continue
            
            other_skills = set(s["name"].lower() for s in other_profile.get("skills", []))
            
            # Calculate Jaccard similarity of skills
            if target_skills and other_skills:
                intersection = len(target_skills & other_skills)
                union = len(target_skills | other_skills)
                similarity = intersection / union if union > 0 else 0.0
                
                if similarity > 0.1:  # Minimum threshold
                    similarities.append({
                        "resource_id": other_id,
                        "name": other_profile["name"],
                        "practice": other_profile["practice"],
                        "similarity": similarity,
                        "common_skills": list(target_skills & other_skills)
                    })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        return similarities[:k]
    
    def _recommend_opportunities_for_resource(self, profile: Dict) -> List[Dict]:
        """Recommend opportunities that would be good for this resource."""
        # This would typically use historical assignment data
        # For now, return opportunities based on skill categories
        
        recommendations = []
        
        skill_categories = profile.get("skill_categories", [])
        expertise_areas = profile.get("expertise_areas", [])
        
        # Recommend opportunities in expertise areas
        for skill in expertise_areas[:3]:
            recommendations.append({
                "type": "expertise_match",
                "description": f"Projects requiring {skill} expertise",
                "confidence": 0.9,
                "reasoning": "Matches core expertise area"
            })
        
        # Recommend opportunities for skill development
        if profile.get("versatility_score", 0) < 0.5:
            recommendations.append({
                "type": "development",
                "description": "Cross-training opportunities in related technologies",
                "confidence": 0.7,
                "reasoning": "Would increase versatility"
            })
        
        return recommendations


def recommend_resources(request_text: str, n: int = 5, 
                       resources_df: pd.DataFrame = None) -> List[Dict[str, Any]]:
    """
    Main function to get resource recommendations.
    
    Args:
        request_text: Text description of the resource need
        n: Number of recommendations to return
        resources_df: DataFrame with resource data (loaded if None)
        
    Returns:
        List of resource recommendations with scores and rationale
    """
    logger.info("Getting resource recommendations")
    
    # Load data if not provided
    if resources_df is None:
        try:
            from .io_loader import load_processed_data
            data = load_processed_data()
            
            # Find resource data
            for name, df in data.items():
                if 'resource' in name.lower():
                    resources_df = df
                    break
            
            if resources_df is None:
                logger.error("No resource data found")
                return []
                
        except Exception as e:
            logger.error(f"Failed to load resource data: {e}")
            return []
    
    # Load taxonomy
    try:
        from .taxonomy import main as build_taxonomy
        _, taxonomy_query = build_taxonomy()
    except Exception as e:
        logger.error(f"Failed to load taxonomy: {e}")
        return []
    
    # Initialize matcher and recommender
    matcher = SmartMatcher(taxonomy_query)
    matcher.initialize()
    
    recommender = ResourceRecommender(taxonomy_query, matcher)
    recommender.initialize(resources_df)
    
    # Get recommendations
    recommendations = recommender.recommend_resources(request_text, n)
    
    return recommendations


def main():
    """Main function to test recommendation pipeline."""
    logger.info("Starting recommendation pipeline test")
    
    # Test with sample requests
    test_requests = [
        "Need experienced Python developer for cloud migration project",
        "Looking for VMware specialist for infrastructure upgrade",
        "Require cybersecurity expert for security assessment",
        "Need project manager for agile development initiative"
    ]
    
    test_results = {}
    
    for request in test_requests:
        logger.info(f"Testing request: {request}")
        
        try:
            recommendations = recommend_resources(request, n=3)
            
            test_results[request] = {
                "recommendations_count": len(recommendations),
                "recommendations": recommendations[:3],  # Top 3 for logging
                "avg_score": np.mean([r["score"] for r in recommendations]) if recommendations else 0.0
            }
            
            # Log top recommendation
            if recommendations:
                top_rec = recommendations[0]
                logger.info(f"  Top recommendation: {top_rec['name']} (score: {top_rec['score']:.3f})")
                logger.info(f"  Rationale: {'; '.join(top_rec['rationale'][:2])}")
        
        except Exception as e:
            logger.error(f"Recommendation failed for '{request}': {e}")
            test_results[request] = {"error": str(e)}
    
    # Save test results
    from .utils import save_metrics
    save_metrics(test_results, "recommendation_test_results.json")
    
    logger.info("Recommendation pipeline test completed")
    
    return test_results


if __name__ == "__main__":
    main()
