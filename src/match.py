"""
Smart skill matching and ranking pipeline.

This module provides intelligent matching of opportunities to skills,
skillsets, and services with transparent scoring and rationale.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

from .utils import config, logger, Timer
from .features import TextVectorizer
from .taxonomy import TaxonomyQuery, TaxonomyBuilder
from .focus_area_integration import FocusAreaIntegrator


class SmartMatcher:
    """Smart matching system for skills, skillsets, and services with Focus Area support."""
    
    def __init__(self, taxonomy_query: TaxonomyQuery):
        self.taxonomy = taxonomy_query
        self.text_vectorizer = None
        self.tfidf_vectorizer = None
        self.entity_vectors = {}
        self.frequency_stats = {}
        self.focus_area_integrator = FocusAreaIntegrator()
        
    def initialize(self, opportunity_df: pd.DataFrame = None):
        """Initialize matcher with text vectorizers and frequency stats."""
        logger.info("Initializing smart matcher")
        
        # Initialize text vectorizers
        self._initialize_vectorizers()
        
        # Calculate frequency stats from opportunities if available
        if opportunity_df is not None:
            self._calculate_frequency_stats(opportunity_df)
        
        # Precompute entity vectors
        self._precompute_entity_vectors()
        
        logger.info("Smart matcher initialized successfully")
    
    def _initialize_vectorizers(self):
        """Initialize text vectorizers."""
        # Collect all entity text for vectorizer training
        all_texts = []
        
        for entity_type, entities in self.taxonomy.entities.items():
            for entity_data in entities.values():
                text_parts = [
                    entity_data["name"],
                    entity_data.get("description", ""),
                    entity_data.get("category", "")
                ]
                combined_text = ' '.join(filter(None, text_parts))
                if combined_text.strip():
                    all_texts.append(combined_text)
        
        if all_texts:
            # Initialize primary vectorizer
            self.text_vectorizer = TextVectorizer(mode="auto")
            self.text_vectorizer.fit(all_texts)
            
            # Initialize TF-IDF for BM25-style scoring
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=1,
                max_df=0.9
            )
            self.tfidf_vectorizer.fit(all_texts)
            
            logger.info("Text vectorizers initialized")
        else:
            logger.warning("No text data found for vectorizer initialization")
    
    def _calculate_frequency_stats(self, opportunity_df: pd.DataFrame):
        """Calculate usage frequency statistics from opportunities."""
        logger.info("Calculating frequency statistics from opportunities")
        
        # Find text columns in opportunities
        text_columns = [col for col in opportunity_df.select_dtypes(include=['object']).columns
                       if opportunity_df[col].astype(str).str.len().mean() > 5]
        
        # Extract entity mentions from opportunity text
        entity_mentions = defaultdict(int)
        
        for _, row in opportunity_df.iterrows():
            # Combine text from all text columns
            combined_text = ' '.join(str(row[col]) for col in text_columns 
                                   if pd.notna(row[col])).lower()
            
            # Count mentions of each entity
            for entity_type, entities in self.taxonomy.entities.items():
                for entity_id, entity_data in entities.items():
                    entity_name = entity_data["name"].lower()
                    
                    # Count exact matches and partial matches
                    if entity_name in combined_text:
                        entity_mentions[entity_id] += 2  # Exact match gets higher weight
                    
                    # Check for word-level matches
                    entity_words = set(re.findall(r'\b\w+\b', entity_name))
                    text_words = set(re.findall(r'\b\w+\b', combined_text))
                    
                    overlap = len(entity_words & text_words)
                    if overlap > 0:
                        entity_mentions[entity_id] += overlap / len(entity_words)
        
        # Normalize frequencies
        total_mentions = sum(entity_mentions.values())
        if total_mentions > 0:
            self.frequency_stats = {
                entity_id: count / total_mentions 
                for entity_id, count in entity_mentions.items()
            }
        
        logger.info(f"Calculated frequency stats for {len(self.frequency_stats)} entities")
    
    def _precompute_entity_vectors(self):
        """Precompute vectors for all entities."""
        if not self.text_vectorizer:
            return
        
        logger.info("Precomputing entity vectors")
        
        for entity_type, entities in self.taxonomy.entities.items():
            entity_texts = []
            entity_ids = []
            
            for entity_id, entity_data in entities.items():
                text = f"{entity_data['name']} {entity_data.get('description', '')} {entity_data.get('category', '')}"
                entity_texts.append(text)
                entity_ids.append(entity_id)
            
            if entity_texts:
                vectors = self.text_vectorizer.transform(entity_texts)
                
                for i, entity_id in enumerate(entity_ids):
                    self.entity_vectors[entity_id] = vectors[i]
        
        logger.info(f"Precomputed vectors for {len(self.entity_vectors)} entities")
    
    def match_skills_and_services(self, text: str, top_k: int = 10, 
                                  include_focus_areas: bool = True) -> Dict[str, List[Dict]]:
        """
        Match text to skills, skillsets, services, and Focus Areas.
        
        Args:
            text: Input text (opportunity description, query, etc.)
            top_k: Number of top matches to return for each entity type
            
        Returns:
            Dict with keys 'skills', 'skillsets', 'services', 'focus_areas', each containing
            list of matches with name, score, and rationale
        """
        logger.info(f"Matching text: '{text[:100]}...'")
        
        if not self.text_vectorizer:
            logger.error("Matcher not initialized")
            return {"skills": [], "skillsets": [], "services": [], "focus_areas": []}
        
        # Generate candidates for each entity type
        results = {}
        
        for entity_type in ["skills", "skillsets", "services"]:
            candidates = self._generate_candidates(text, entity_type)
            scored_candidates = self._score_candidates(text, candidates, entity_type)
            ranked_candidates = self._rank_candidates(scored_candidates)
            
            results[entity_type] = ranked_candidates[:top_k]
        
        # Add Focus Area predictions if requested
        if include_focus_areas:
            focus_area_predictions = self.focus_area_integrator.classify_text_to_focus_area(text, top_k=3)
            results["focus_areas"] = [
                {
                    "name": fa,
                    "score": conf,
                    "rationale": f"Matched based on keywords and domain alignment (confidence: {conf:.2%})",
                    "requirements": self.focus_area_integrator.get_focus_area_requirements(fa)
                }
                for fa, conf in focus_area_predictions
            ]
        
        return results
    
    def _generate_candidates(self, text: str, entity_type: str) -> List[str]:
        """Generate candidate entities for matching."""
        candidates = set()
        
        # Method 1: All entities of this type (for small entity sets)
        entities = self.taxonomy.entities[entity_type]
        if len(entities) <= 1000:  # Include all if manageable size
            candidates.update(entities.keys())
        
        # Method 2: N-gram overlap
        text_ngrams = self._extract_ngrams(text, n=2)
        
        for entity_id, entity_data in entities.items():
            entity_text = f"{entity_data['name']} {entity_data.get('description', '')}"
            entity_ngrams = self._extract_ngrams(entity_text, n=2)
            
            # Check for n-gram overlap
            overlap = len(text_ngrams & entity_ngrams)
            if overlap > 0:
                candidates.add(entity_id)
        
        # Method 3: Keyword matching
        text_keywords = self._extract_keywords(text)
        
        for entity_id, entity_data in entities.items():
            entity_keywords = self._extract_keywords(
                f"{entity_data['name']} {entity_data.get('category', '')}"
            )
            
            # Check for keyword overlap
            if text_keywords & entity_keywords:
                candidates.add(entity_id)
        
        return list(candidates)
    
    def _extract_ngrams(self, text: str, n: int = 2) -> set:
        """Extract n-grams from text."""
        words = re.findall(r'\b\w+\b', text.lower())
        ngrams = set()
        
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            ngrams.add(ngram)
        
        return ngrams
    
    def _extract_keywords(self, text: str) -> set:
        """Extract meaningful keywords from text."""
        # Common technology and business keywords
        tech_keywords = {
            'cloud', 'aws', 'azure', 'gcp', 'kubernetes', 'docker', 'vmware',
            'python', 'java', 'javascript', 'sql', 'nosql', 'database',
            'security', 'network', 'storage', 'backup', 'infrastructure',
            'development', 'deployment', 'management', 'administration',
            'hpe', 'simplivity', 'proliant', 'alletra', 'greenlake'
        }
        
        text_words = set(re.findall(r'\b\w+\b', text.lower()))
        return text_words & tech_keywords
    
    def _score_candidates(self, text: str, candidates: List[str], 
                         entity_type: str) -> List[Dict]:
        """Score candidate entities against input text."""
        if not candidates:
            return []
        
        scored_candidates = []
        
        # Get text vector
        try:
            text_vector = self.text_vectorizer.transform([text])[0]
        except Exception as e:
            logger.warning(f"Failed to vectorize text: {e}")
            return []
        
        for candidate_id in candidates:
            if candidate_id not in self.entity_vectors:
                continue
            
            entity_data = self.taxonomy.entities[entity_type][candidate_id]
            entity_vector = self.entity_vectors[candidate_id]
            
            # Calculate multiple similarity scores
            scores = self._calculate_similarity_scores(
                text, text_vector, entity_data, entity_vector, candidate_id
            )
            
            # Calculate weighted final score with technical relevance
            final_score = (
                config.cosine_weight * scores["cosine"] +
                config.tfidf_weight * scores["tfidf"] +
                0.3 * scores["technical_relevance"] +  # High weight for technical relevance
                scores["soft_skills_penalty"] +  # Apply penalty for irrelevant soft skills
                config.frequency_weight * scores["frequency"] +
                config.recency_weight * scores["recency"]
            )
            
            scored_candidates.append({
                "entity_id": candidate_id,
                "name": entity_data["name"],
                "description": entity_data.get("description", ""),
                "category": entity_data.get("category", ""),
                "score": final_score,
                "score_breakdown": scores,
                "rationale": self._generate_rationale(scores, entity_data)
            })
        
        return scored_candidates
    
    def _calculate_similarity_scores(self, text: str, text_vector: np.ndarray,
                                   entity_data: Dict, entity_vector: np.ndarray,
                                   entity_id: str) -> Dict[str, float]:
        """Calculate multiple similarity scores."""
        scores = {}
        
        # 1. Cosine similarity (semantic)
        try:
            cosine_sim = cosine_similarity([text_vector], [entity_vector])[0][0]
            scores["cosine"] = max(0, cosine_sim)
        except:
            scores["cosine"] = 0.0
        
        # 2. TF-IDF similarity (lexical)
        try:
            entity_text = f"{entity_data['name']} {entity_data.get('description', '')}"
            tfidf_vectors = self.tfidf_vectorizer.transform([text, entity_text])
            tfidf_sim = cosine_similarity(tfidf_vectors[0:1], tfidf_vectors[1:2])[0][0]
            scores["tfidf"] = max(0, tfidf_sim)
        except:
            scores["tfidf"] = 0.0
        
        # 3. Technical relevance boost
        text_lower = text.lower()
        entity_name_lower = entity_data["name"].lower()
        
        technical_terms = {
            'python', 'java', 'javascript', 'react', 'node', 'aws', 'azure', 'gcp',
            'kubernetes', 'docker', 'devops', 'ci/cd', 'terraform', 'ansible',
            'machine learning', 'ai', 'data science', 'analytics', 'sql', 'nosql',
            'mongodb', 'postgresql', 'mysql', 'redis', 'kafka', 'spark', 'hadoop',
            'microservices', 'api', 'rest', 'graphql', 'blockchain', 'cybersecurity',
            'penetration testing', 'vulnerability', 'cloud', 'migration', 'modernization',
            'programming', 'development', 'software', 'application', 'database'
        }
        
        technical_boost = 0.0
        text_words = text_lower.split()
        
        for word in text_words:
            if word in technical_terms:
                if word in entity_name_lower:
                    technical_boost += 2.0  # Strong boost for exact technical matches
                # Check for partial matches in technical terms
                for tech_term in technical_terms:
                    if word in tech_term and tech_term in entity_name_lower:
                        technical_boost += 1.0
        
        scores["technical_relevance"] = min(technical_boost, 3.0)  # Cap at 3.0
        
        # 4. Soft skills penalty for technical requests
        soft_skills_penalty = 0.0
        soft_skill_indicators = ['decision making', 'leadership', 'communication', 
                               'teamwork', 'collaboration', 'management', 'strategy',
                               'planning', 'organization', 'coordination']
        
        has_tech_request = any(word in technical_terms for word in text_words)
        is_soft_skill = any(soft_term in entity_name_lower for soft_term in soft_skill_indicators)
        
        if has_tech_request and is_soft_skill and scores["cosine"] < 0.3:
            soft_skills_penalty = -1.5  # Penalty for irrelevant soft skills
        
        scores["soft_skills_penalty"] = soft_skills_penalty
        
        # 5. Frequency score (popularity)
        scores["frequency"] = self.frequency_stats.get(entity_id, 0.1)  # Default low frequency
        
        # 6. Recency score (placeholder - would use actual recency data)
        scores["recency"] = 0.5  # Neutral score
        
        return scores
    
    def _generate_rationale(self, scores: Dict[str, float], entity_data: Dict) -> List[str]:
        """Generate human-readable rationale for the match."""
        rationale = []
        
        if scores["cosine"] > 0.7:
            rationale.append("Strong semantic similarity")
        elif scores["cosine"] > 0.5:
            rationale.append("Moderate semantic similarity")
        
        if scores["tfidf"] > 0.3:
            rationale.append("High lexical overlap")
        
        if scores["frequency"] > 0.1:
            rationale.append("Frequently used in similar contexts")
        
        # Category-based rationale
        category = entity_data.get("category", "")
        if category:
            rationale.append(f"Category: {category}")
        
        if not rationale:
            rationale.append("Basic relevance detected")
        
        return rationale
    
    def _rank_candidates(self, scored_candidates: List[Dict]) -> List[Dict]:
        """Rank candidates by score with tie-breaking."""
        if not scored_candidates:
            return []
        
        # Sort by score (primary) and then by name (tie-breaker)
        ranked = sorted(scored_candidates, 
                       key=lambda x: (x["score"], x["name"]), 
                       reverse=True)
        
        return ranked
    
    def find_similar_entities(self, entity_name: str, entity_type: str, 
                            k: int = 10) -> List[Dict]:
        """Find entities similar to a given entity."""
        entity_id = self.taxonomy._find_entity_id_by_name(entity_name, entity_type)
        
        if not entity_id or entity_id not in self.entity_vectors:
            return []
        
        target_vector = self.entity_vectors[entity_id]
        similarities = []
        
        # Compare with all entities of the same type
        for other_id, other_vector in self.entity_vectors.items():
            if other_id == entity_id:
                continue
            
            # Check if same entity type
            other_entity_type = None
            for ent_type, entities in self.taxonomy.entities.items():
                if other_id in entities:
                    other_entity_type = ent_type
                    break
            
            if other_entity_type != entity_type:
                continue
            
            # Calculate similarity
            try:
                similarity = cosine_similarity([target_vector], [other_vector])[0][0]
                
                if similarity > 0.1:  # Minimum threshold
                    other_entity_data = self.taxonomy.entities[entity_type][other_id]
                    similarities.append({
                        "entity_id": other_id,
                        "name": other_entity_data["name"],
                        "description": other_entity_data.get("description", ""),
                        "category": other_entity_data.get("category", ""),
                        "similarity": similarity
                    })
            except:
                continue
        
        # Sort by similarity
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        return similarities[:k]
    
    def get_entity_recommendations(self, text: str, 
                                 include_explanations: bool = True) -> Dict[str, Any]:
        """Get comprehensive entity recommendations with explanations."""
        logger.info("Getting entity recommendations with explanations")
        
        # Get basic matches
        matches = self.match_skills_and_services(text, top_k=10)
        
        recommendations = {
            "query": text,
            "matches": matches,
            "summary": {
                "total_skills": len(matches["skills"]),
                "total_skillsets": len(matches["skillsets"]), 
                "total_services": len(matches["services"]),
                "best_match_type": self._determine_best_match_type(matches),
                "confidence": self._calculate_overall_confidence(matches)
            }
        }
        
        if include_explanations:
            recommendations["explanations"] = self._generate_match_explanations(text, matches)
        
        return recommendations
    
    def _determine_best_match_type(self, matches: Dict[str, List]) -> str:
        """Determine which entity type has the best matches."""
        best_scores = {}
        
        for entity_type, entity_matches in matches.items():
            if entity_matches:
                best_scores[entity_type] = entity_matches[0]["score"]
            else:
                best_scores[entity_type] = 0.0
        
        return max(best_scores, key=best_scores.get) if best_scores else "none"
    
    def _calculate_overall_confidence(self, matches: Dict[str, List]) -> float:
        """Calculate overall confidence in the matches."""
        all_scores = []
        
        for entity_matches in matches.values():
            for match in entity_matches[:3]:  # Top 3 from each type
                all_scores.append(match["score"])
        
        return np.mean(all_scores) if all_scores else 0.0
    
    def _generate_match_explanations(self, text: str, matches: Dict[str, List]) -> Dict[str, Any]:
        """Generate explanations for why matches were selected."""
        explanations = {
            "text_analysis": self._analyze_input_text(text),
            "match_reasoning": {},
            "suggestions": []
        }
        
        # Explain top match from each category
        for entity_type, entity_matches in matches.items():
            if entity_matches:
                top_match = entity_matches[0]
                explanations["match_reasoning"][entity_type] = {
                    "top_match": top_match["name"],
                    "score": top_match["score"],
                    "reasons": top_match["rationale"],
                    "score_breakdown": top_match["score_breakdown"]
                }
        
        # Generate suggestions
        explanations["suggestions"] = self._generate_improvement_suggestions(text, matches)
        
        return explanations
    
    def _analyze_input_text(self, text: str) -> Dict[str, Any]:
        """Analyze the input text to understand its characteristics."""
        return {
            "length": len(text),
            "word_count": len(text.split()),
            "keywords": list(self._extract_keywords(text)),
            "complexity": "high" if len(text.split()) > 20 else "medium" if len(text.split()) > 5 else "low",
            "has_technical_terms": bool(self._extract_keywords(text))
        }
    
    def _generate_improvement_suggestions(self, text: str, matches: Dict[str, List]) -> List[str]:
        """Generate suggestions for improving matches."""
        suggestions = []
        
        # Check if matches are low quality
        all_scores = []
        for entity_matches in matches.values():
            if entity_matches:
                all_scores.append(entity_matches[0]["score"])
        
        avg_score = np.mean(all_scores) if all_scores else 0.0
        
        if avg_score < 0.3:
            suggestions.append("Consider providing more specific technical details")
            suggestions.append("Include relevant technology names or product names")
        
        if len(text.split()) < 5:
            suggestions.append("Provide more descriptive text for better matching")
        
        # Check for missing entity types
        for entity_type, entity_matches in matches.items():
            if not entity_matches:
                suggestions.append(f"No {entity_type} matches found - consider broader terminology")
        
        return suggestions


def match_skills_and_services(text: str, taxonomy_query: TaxonomyQuery = None, 
                            top_k: int = 10) -> Dict[str, List[Dict]]:
    """
    Main function to match text to skills and services.
    
    Args:
        text: Input text to match
        taxonomy_query: Pre-initialized taxonomy query object
        top_k: Number of top matches per entity type
        
    Returns:
        Dictionary with matched skills, skillsets, and services
    """
    if taxonomy_query is None:
        # Try to load existing taxonomy
        try:
            from .taxonomy import main as build_taxonomy
            _, taxonomy_query = build_taxonomy()
        except Exception as e:
            logger.error(f"Failed to load taxonomy: {e}")
            return {"skills": [], "skillsets": [], "services": []}
    
    # Initialize matcher
    matcher = SmartMatcher(taxonomy_query)
    matcher.initialize()
    
    # Perform matching
    return matcher.match_skills_and_services(text, top_k)


def main():
    """Main function to test matching pipeline."""
    logger.info("Starting matching pipeline test")
    
    # Load taxonomy
    try:
        from .taxonomy import main as build_taxonomy
        _, taxonomy_query = build_taxonomy()
    except Exception as e:
        logger.error(f"Failed to load taxonomy: {e}")
        return
    
    # Initialize matcher
    matcher = SmartMatcher(taxonomy_query)
    matcher.initialize()
    
    # Test with sample queries
    test_queries = [
        "Need Python developer for cloud infrastructure project",
        "VMware vSphere administration and management",
        "HPE SimpliVity installation and configuration",
        "Cybersecurity assessment and vulnerability testing",
        "Project management for agile development"
    ]
    
    test_results = {}
    
    for query in test_queries:
        logger.info(f"Testing query: {query}")
        
        try:
            results = matcher.match_skills_and_services(query, top_k=5)
            recommendations = matcher.get_entity_recommendations(query)
            
            test_results[query] = {
                "matches": results,
                "recommendations": recommendations,
                "summary": {
                    "skills_found": len(results["skills"]),
                    "skillsets_found": len(results["skillsets"]),
                    "services_found": len(results["services"]),
                    "best_match_type": recommendations["summary"]["best_match_type"],
                    "confidence": recommendations["summary"]["confidence"]
                }
            }
            
            # Log top match from each category
            for entity_type, matches in results.items():
                if matches:
                    top_match = matches[0]
                    logger.info(f"  Top {entity_type}: {top_match['name']} (score: {top_match['score']:.3f})")
        
        except Exception as e:
            logger.error(f"Matching failed for query '{query}': {e}")
            test_results[query] = {"error": str(e)}
    
    # Save test results
    save_metrics(test_results, "matching_test_results.json")
    
    logger.info("Matching pipeline test completed")
    
    return test_results


if __name__ == "__main__":
    main()
