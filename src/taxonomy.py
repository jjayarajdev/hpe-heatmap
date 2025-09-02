"""
Bidirectional taxonomy builder and query system.

This module builds authoritative two-way mappings between skills, skillsets,
and services with intelligent suggestions and confidence scores.
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional, Set
import logging
import json
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

from .utils import config, logger, Timer, save_metrics
from .features import TextVectorizer


class TaxonomyBuilder:
    """Build bidirectional taxonomy with intelligent suggestions."""
    
    def __init__(self):
        self.graph = nx.Graph()  # Undirected graph for bidirectional relationships
        self.entities = {
            "services": {},
            "skillsets": {},
            "skills": {}
        }
        self.mappings = {
            "service_skillset": [],
            "skillset_skill": [],
            "service_skill": []  # Derived mappings
        }
        self.suggested_edges = []
        self.confidence_scores = {}
        self.text_vectorizer = None
        
    def build_taxonomy(self, services_df: pd.DataFrame, 
                      skillsets_df: pd.DataFrame,
                      skills_df: pd.DataFrame,
                      service_skillset_df: pd.DataFrame = None,
                      skillset_skill_df: pd.DataFrame = None) -> Dict[str, Any]:
        """Build complete taxonomy from input DataFrames."""
        logger.info("Building bidirectional taxonomy")
        
        with Timer(logger, "taxonomy building"):
            # 1. Process entities
            self._process_services(services_df)
            self._process_skillsets(skillsets_df)
            self._process_skills(skills_df)
            
            # 2. Process explicit mappings
            if service_skillset_df is not None:
                self._process_service_skillset_mappings(service_skillset_df)
            
            if skillset_skill_df is not None:
                self._process_skillset_skill_mappings(skillset_skill_df)
            
            # 3. Build graph
            self._build_graph()
            
            # 4. Initialize text vectorizer for similarity calculations
            self._initialize_text_vectorizer()
            
            # 5. Generate suggested mappings
            self._generate_suggested_mappings()
            
            # 6. Calculate confidence scores
            self._calculate_confidence_scores()
        
        taxonomy_stats = {
            "entities": {
                "services": len(self.entities["services"]),
                "skillsets": len(self.entities["skillsets"]),
                "skills": len(self.entities["skills"])
            },
            "mappings": {
                "service_skillset": len(self.mappings["service_skillset"]),
                "skillset_skill": len(self.mappings["skillset_skill"]),
                "service_skill": len(self.mappings["service_skill"])
            },
            "graph": {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
                "density": nx.density(self.graph)
            },
            "suggestions": len(self.suggested_edges)
        }
        
        logger.info(f"Taxonomy built: {taxonomy_stats}")
        return taxonomy_stats
    
    def _process_services(self, df: pd.DataFrame):
        """Process services from DataFrame."""
        logger.info(f"Processing {len(df)} services")
        
        for idx, row in df.iterrows():
            # Try to find service name column
            service_name = None
            service_id = None
            
            # Common column name patterns for services
            name_columns = [col for col in df.columns 
                           if any(term in col.lower() for term in ['service', 'name'])]
            
            if name_columns:
                service_name = str(row[name_columns[0]]).strip()
            
            # Try to find ID column
            id_columns = [col for col in df.columns 
                         if any(term in col.lower() for term in ['id', 'number'])]
            
            if id_columns:
                service_id = str(row[id_columns[0]]).strip()
            else:
                service_id = f"SVC_{idx:06d}"
            
            if service_name and service_name.lower() not in ['nan', 'none', '']:
                self.entities["services"][service_id] = {
                    "id": service_id,
                    "name": service_name,
                    "category": str(row.get("category", row.get("focus_area", ""))).strip(),
                    "description": str(row.get("description", "")).strip(),
                    "aliases": self._extract_aliases(service_name),
                    "metadata": {col: str(row[col]) for col in df.columns 
                               if col not in name_columns + id_columns}
                }
    
    def _process_skillsets(self, df: pd.DataFrame):
        """Process skillsets from DataFrame.""" 
        logger.info(f"Processing {len(df)} skillsets")
        
        for idx, row in df.iterrows():
            # Find skillset name
            skillset_name = None
            skillset_id = None
            
            name_columns = [col for col in df.columns 
                           if any(term in col.lower() for term in ['skillset', 'name'])]
            
            if name_columns:
                skillset_name = str(row[name_columns[0]]).strip()
            
            # Generate ID
            skillset_id = f"SKS_{idx:06d}"
            
            if skillset_name and skillset_name.lower() not in ['nan', 'none', '']:
                self.entities["skillsets"][skillset_id] = {
                    "id": skillset_id,
                    "name": skillset_name,
                    "category": str(row.get("category", row.get("technical_domain", ""))).strip(),
                    "description": str(row.get("description", "")).strip(),
                    "parent_id": None,  # Will be filled from hierarchy
                    "hierarchy": str(row.get("hierarchy", "")).strip(),
                    "metadata": {col: str(row[col]) for col in df.columns 
                               if col not in name_columns}
                }
    
    def _process_skills(self, df: pd.DataFrame):
        """Process skills from DataFrame."""
        logger.info(f"Processing {len(df)} skills")
        
        for idx, row in df.iterrows():
            # Find skill name
            skill_name = None
            skill_id = None
            
            name_columns = [col for col in df.columns 
                           if any(term in col.lower() for term in ['skill', 'name', 'certification'])]
            
            if name_columns:
                skill_name = str(row[name_columns[0]]).strip()
            
            # Generate ID
            skill_id = f"SKL_{idx:06d}"
            
            if skill_name and skill_name.lower() not in ['nan', 'none', '']:
                self.entities["skills"][skill_id] = {
                    "id": skill_id,
                    "name": skill_name,
                    "description": str(row.get("description", "")).strip(),
                    "category": str(row.get("category", row.get("technical_domain", ""))).strip(),
                    "hierarchy": str(row.get("hierarchy", "")).strip(),
                    "skillset_id": None,  # Will be filled from mappings
                    "synonyms": self._extract_synonyms(skill_name),
                    "metadata": {col: str(row[col]) for col in df.columns 
                               if col not in name_columns}
                }
    
    def _extract_aliases(self, name: str) -> List[str]:
        """Extract potential aliases from a name."""
        aliases = []
        
        # Common abbreviations
        abbrev_map = {
            'management': 'mgmt',
            'development': 'dev',
            'administration': 'admin',
            'technology': 'tech',
            'engineering': 'eng',
            'operations': 'ops',
            'infrastructure': 'infra',
            'application': 'app',
            'security': 'sec'
        }
        
        name_lower = name.lower()
        for full, abbrev in abbrev_map.items():
            if full in name_lower:
                aliases.append(name.replace(full, abbrev))
                aliases.append(name.replace(full.title(), abbrev.upper()))
        
        return list(set(aliases))
    
    def _extract_synonyms(self, skill_name: str) -> List[str]:
        """Extract potential synonyms for a skill."""
        synonyms = []
        
        # Technology synonyms
        tech_synonyms = {
            'vmware': ['vsphere', 'esx', 'vcenter'],
            'microsoft': ['ms', 'msft'],
            'amazon': ['aws', 'amazon web services'],
            'kubernetes': ['k8s'],
            'docker': ['containers'],
            'python': ['py'],
            'javascript': ['js'],
            'sql': ['structured query language']
        }
        
        skill_lower = skill_name.lower()
        for tech, syn_list in tech_synonyms.items():
            if tech in skill_lower:
                for syn in syn_list:
                    synonyms.append(skill_name.replace(tech, syn))
        
        return list(set(synonyms))
    
    def _process_service_skillset_mappings(self, df: pd.DataFrame):
        """Process service-skillset mappings."""
        logger.info(f"Processing {len(df)} service-skillset mappings")
        
        for _, row in df.iterrows():
            # Find service and skillset identifiers
            service_ref = self._find_entity_reference(row, "service")
            skillset_ref = self._find_entity_reference(row, "skillset")
            
            if service_ref and skillset_ref:
                mapping = {
                    "service_id": service_ref,
                    "skillset_id": skillset_ref,
                    "mandatory": self._parse_boolean(row.get("mandatory", True)),
                    "weight": float(row.get("weight", 1.0)),
                    "confidence": 1.0  # Explicit mappings have high confidence
                }
                
                self.mappings["service_skillset"].append(mapping)
    
    def _process_skillset_skill_mappings(self, df: pd.DataFrame):
        """Process skillset-skill mappings."""
        logger.info(f"Processing {len(df)} skillset-skill mappings")
        
        for _, row in df.iterrows():
            skillset_ref = self._find_entity_reference(row, "skillset")
            skill_ref = self._find_entity_reference(row, "skill")
            
            if skillset_ref and skill_ref:
                mapping = {
                    "skillset_id": skillset_ref,
                    "skill_id": skill_ref,
                    "weight": float(row.get("weight", 1.0)),
                    "hierarchy_level": int(row.get("hierarchy_level", 1)),
                    "confidence": 1.0
                }
                
                self.mappings["skillset_skill"].append(mapping)
    
    def _find_entity_reference(self, row: pd.Series, entity_type: str) -> Optional[str]:
        """Find reference to an entity in a row."""
        # Look for direct ID reference
        id_columns = [col for col in row.index 
                     if entity_type in col.lower() and 'id' in col.lower()]
        
        if id_columns and pd.notna(row[id_columns[0]]):
            return str(row[id_columns[0]]).strip()
        
        # Look for name reference and match to entity
        name_columns = [col for col in row.index 
                       if entity_type in col.lower() and 'name' in col.lower()]
        
        if name_columns and pd.notna(row[name_columns[0]]):
            name = str(row[name_columns[0]]).strip().lower()
            
            # Find matching entity by name
            entities = self.entities[f"{entity_type}s"]
            for entity_id, entity_data in entities.items():
                if entity_data["name"].lower() == name:
                    return entity_id
                
                # Check aliases
                for alias in entity_data.get("aliases", []):
                    if alias.lower() == name:
                        return entity_id
        
        return None
    
    def _parse_boolean(self, value: Any) -> bool:
        """Parse boolean value from various formats."""
        if isinstance(value, bool):
            return value
        
        str_val = str(value).lower().strip()
        return str_val in ['true', '1', 'yes', 'mandatory', 'required']
    
    def _build_graph(self):
        """Build NetworkX graph from entities and mappings."""
        logger.info("Building taxonomy graph")
        
        # Add nodes for all entities
        for entity_type, entities in self.entities.items():
            for entity_id, entity_data in entities.items():
                # Avoid duplicate 'name' attribute
                node_attrs = {
                    "type": entity_type[:-1],  # Remove 's' from plural
                    "entity_name": entity_data["name"],
                    "category": entity_data.get("category", ""),
                    "description": entity_data.get("description", "")
                }
                
                # Add other metadata without conflicts
                for key, value in entity_data.items():
                    if key not in ["name"] and key not in node_attrs:
                        node_attrs[key] = value
                
                self.graph.add_node(entity_id, **node_attrs)
        
        # Add edges from mappings
        for mapping in self.mappings["service_skillset"]:
            self.graph.add_edge(
                mapping["service_id"],
                mapping["skillset_id"],
                type="service_skillset",
                weight=mapping["weight"],
                mandatory=mapping["mandatory"],
                confidence=mapping["confidence"]
            )
        
        for mapping in self.mappings["skillset_skill"]:
            self.graph.add_edge(
                mapping["skillset_id"],
                mapping["skill_id"],
                type="skillset_skill",
                weight=mapping["weight"],
                hierarchy_level=mapping["hierarchy_level"],
                confidence=mapping["confidence"]
            )
        
        # Create derived service-skill mappings
        self._create_derived_mappings()
        
        logger.info(f"Graph built with {self.graph.number_of_nodes()} nodes "
                   f"and {self.graph.number_of_edges()} edges")
    
    def _create_derived_mappings(self):
        """Create derived service-skill mappings through skillsets."""
        logger.info("Creating derived service-skill mappings")
        
        derived_count = 0
        
        for service_id in [n for n in self.graph.nodes() 
                          if self.graph.nodes[n]["type"] == "service"]:
            
            # Find connected skillsets
            skillsets = [n for n in self.graph.neighbors(service_id)
                        if self.graph.nodes[n]["type"] == "skillset"]
            
            for skillset_id in skillsets:
                # Find skills connected to this skillset
                skills = [n for n in self.graph.neighbors(skillset_id)
                         if self.graph.nodes[n]["type"] == "skill"]
                
                for skill_id in skills:
                    # Create derived service-skill mapping
                    if not self.graph.has_edge(service_id, skill_id):
                        # Calculate derived weight
                        service_skillset_weight = self.graph[service_id][skillset_id]["weight"]
                        skillset_skill_weight = self.graph[skillset_id][skill_id]["weight"]
                        derived_weight = service_skillset_weight * skillset_skill_weight
                        
                        self.graph.add_edge(
                            service_id,
                            skill_id,
                            type="derived_service_skill",
                            weight=derived_weight,
                            confidence=0.8,  # Lower confidence for derived mappings
                            path=[service_id, skillset_id, skill_id]
                        )
                        
                        self.mappings["service_skill"].append({
                            "service_id": service_id,
                            "skill_id": skill_id,
                            "weight": derived_weight,
                            "confidence": 0.8,
                            "type": "derived"
                        })
                        
                        derived_count += 1
        
        logger.info(f"Created {derived_count} derived service-skill mappings")
    
    def _initialize_text_vectorizer(self):
        """Initialize text vectorizer for similarity calculations."""
        logger.info("Initializing text vectorizer for similarity calculations")
        
        # Collect all text for vectorizer training
        all_texts = []
        
        for entity_type, entities in self.entities.items():
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
            self.text_vectorizer = TextVectorizer(mode="auto")
            self.text_vectorizer.fit(all_texts)
            logger.info("Text vectorizer initialized successfully")
        else:
            logger.warning("No text data available for vectorizer initialization")
    
    def _generate_suggested_mappings(self):
        """Generate intelligent mapping suggestions."""
        logger.info("Generating mapping suggestions")
        
        if not self.text_vectorizer:
            logger.warning("Text vectorizer not available, skipping suggestions")
            return
        
        suggestions = []
        
        # Suggest service-skillset mappings
        suggestions.extend(self._suggest_service_skillset_mappings())
        
        # Suggest skillset-skill mappings  
        suggestions.extend(self._suggest_skillset_skill_mappings())
        
        # Filter and rank suggestions
        self.suggested_edges = self._filter_and_rank_suggestions(suggestions)
        
        logger.info(f"Generated {len(self.suggested_edges)} mapping suggestions")
    
    def _suggest_service_skillset_mappings(self) -> List[Dict]:
        """Suggest service-skillset mappings based on similarity."""
        suggestions = []
        
        service_texts = []
        service_ids = []
        
        for service_id, service_data in self.entities["services"].items():
            text = f"{service_data['name']} {service_data.get('description', '')} {service_data.get('category', '')}"
            service_texts.append(text)
            service_ids.append(service_id)
        
        skillset_texts = []
        skillset_ids = []
        
        for skillset_id, skillset_data in self.entities["skillsets"].items():
            text = f"{skillset_data['name']} {skillset_data.get('description', '')} {skillset_data.get('category', '')}"
            skillset_texts.append(text)
            skillset_ids.append(skillset_id)
        
        if not service_texts or not skillset_texts:
            return suggestions
        
        # Calculate similarities
        try:
            service_vectors = self.text_vectorizer.transform(service_texts)
            skillset_vectors = self.text_vectorizer.transform(skillset_texts)
            
            similarities = cosine_similarity(service_vectors, skillset_vectors)
            
            # Find high-similarity pairs that aren't already mapped
            existing_pairs = {(m["service_id"], m["skillset_id"]) 
                            for m in self.mappings["service_skillset"]}
            
            for i, service_id in enumerate(service_ids):
                for j, skillset_id in enumerate(skillset_ids):
                    if (service_id, skillset_id) not in existing_pairs:
                        similarity = similarities[i, j]
                        
                        if similarity > config.edge_threshold:
                            suggestions.append({
                                "type": "service_skillset",
                                "source": service_id,
                                "target": skillset_id,
                                "similarity": similarity,
                                "evidence": ["cosine_similarity"],
                                "confidence": similarity
                            })
        
        except Exception as e:
            logger.warning(f"Error generating service-skillset suggestions: {e}")
        
        return suggestions
    
    def _suggest_skillset_skill_mappings(self) -> List[Dict]:
        """Suggest skillset-skill mappings based on similarity."""
        suggestions = []
        
        # Similar approach as service-skillset but for skillset-skill
        skillset_texts = []
        skillset_ids = []
        
        for skillset_id, skillset_data in self.entities["skillsets"].items():
            text = f"{skillset_data['name']} {skillset_data.get('category', '')}"
            skillset_texts.append(text)
            skillset_ids.append(skillset_id)
        
        skill_texts = []
        skill_ids = []
        
        for skill_id, skill_data in self.entities["skills"].items():
            text = f"{skill_data['name']} {skill_data.get('description', '')} {skill_data.get('category', '')}"
            skill_texts.append(text)
            skill_ids.append(skill_id)
        
        if not skillset_texts or not skill_texts:
            return suggestions
        
        try:
            skillset_vectors = self.text_vectorizer.transform(skillset_texts)
            skill_vectors = self.text_vectorizer.transform(skill_texts)
            
            similarities = cosine_similarity(skillset_vectors, skill_vectors)
            
            existing_pairs = {(m["skillset_id"], m["skill_id"]) 
                            for m in self.mappings["skillset_skill"]}
            
            for i, skillset_id in enumerate(skillset_ids):
                for j, skill_id in enumerate(skill_ids):
                    if (skillset_id, skill_id) not in existing_pairs:
                        similarity = similarities[i, j]
                        
                        if similarity > config.edge_threshold:
                            suggestions.append({
                                "type": "skillset_skill",
                                "source": skillset_id,
                                "target": skill_id,
                                "similarity": similarity,
                                "evidence": ["cosine_similarity"],
                                "confidence": similarity
                            })
        
        except Exception as e:
            logger.warning(f"Error generating skillset-skill suggestions: {e}")
        
        return suggestions
    
    def _filter_and_rank_suggestions(self, suggestions: List[Dict]) -> List[Dict]:
        """Filter and rank suggestions by quality."""
        if not suggestions:
            return []
        
        # Sort by confidence/similarity
        suggestions.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Take top suggestions per node
        suggestions_per_source = defaultdict(list)
        
        for suggestion in suggestions:
            source = suggestion["source"]
            suggestions_per_source[source].append(suggestion)
        
        # Limit suggestions per source node
        filtered_suggestions = []
        for source, source_suggestions in suggestions_per_source.items():
            top_suggestions = source_suggestions[:config.suggestions_per_node]
            filtered_suggestions.extend(top_suggestions)
        
        return filtered_suggestions
    
    def _calculate_confidence_scores(self):
        """Calculate confidence scores for all mappings."""
        logger.info("Calculating confidence scores")
        
        # For explicit mappings, confidence is already set
        # For suggested mappings, confidence is based on multiple signals
        
        for suggestion in self.suggested_edges:
            confidence_factors = []
            
            # Similarity score
            confidence_factors.append(suggestion["similarity"])
            
            # Co-occurrence frequency (if we had usage data)
            # This would require actual usage/assignment data
            confidence_factors.append(0.5)  # Placeholder
            
            # String similarity
            source_name = self._get_entity_name(suggestion["source"])
            target_name = self._get_entity_name(suggestion["target"])
            string_sim = self._calculate_string_similarity(source_name, target_name)
            confidence_factors.append(string_sim)
            
            # Calculate weighted confidence
            weights = [config.taxonomy_alpha, config.taxonomy_beta, config.taxonomy_gamma]
            weighted_confidence = sum(w * f for w, f in zip(weights, confidence_factors))
            
            suggestion["confidence"] = min(weighted_confidence, 1.0)
            self.confidence_scores[f"{suggestion['source']}-{suggestion['target']}"] = weighted_confidence
    
    def _get_entity_name(self, entity_id: str) -> str:
        """Get entity name from ID."""
        for entity_type, entities in self.entities.items():
            if entity_id in entities:
                return entities[entity_id]["name"]
        return ""
    
    def _calculate_string_similarity(self, name1: str, name2: str) -> float:
        """Calculate string similarity between two names."""
        if not name1 or not name2:
            return 0.0
        
        # Simple Jaccard similarity on words
        words1 = set(re.findall(r'\w+', name1.lower()))
        words2 = set(re.findall(r'\w+', name2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def save_taxonomy(self, base_path: str = "artifacts"):
        """Save taxonomy to JSON and Parquet formats."""
        base_path = Path(base_path)
        
        # Prepare taxonomy data
        taxonomy_data = {
            "services": list(self.entities["services"].values()),
            "skillsets": list(self.entities["skillsets"].values()),
            "skills": list(self.entities["skills"].values()),
            "edges": [],
            "suggested_edges": self.suggested_edges,
            "metadata": {
                "created_at": pd.Timestamp.now().isoformat(),
                "config": {
                    "alpha": config.taxonomy_alpha,
                    "beta": config.taxonomy_beta,
                    "gamma": config.taxonomy_gamma,
                    "edge_threshold": config.edge_threshold
                }
            }
        }
        
        # Add explicit edges
        for edge_data in self.graph.edges(data=True):
            source, target, attrs = edge_data
            taxonomy_data["edges"].append({
                "source": source,
                "target": target,
                "source_type": self.graph.nodes[source]["type"],
                "target_type": self.graph.nodes[target]["type"],
                "weight": attrs.get("weight", 1.0),
                "confidence": attrs.get("confidence", 1.0),
                "type": attrs.get("type", "explicit"),
                "evidence": attrs.get("evidence", ["explicit"])
            })
        
        # Save as JSON
        json_path = base_path / "taxonomy.json"
        with open(json_path, 'w') as f:
            json.dump(taxonomy_data, f, indent=2, default=str)
        
        logger.info(f"Taxonomy saved to {json_path}")
        
        # Save as Parquet for efficient querying
        self._save_taxonomy_parquet(taxonomy_data, base_path)
        
        return json_path
    
    def _save_taxonomy_parquet(self, taxonomy_data: Dict, base_path: Path):
        """Save taxonomy as Parquet files for efficient querying."""
        # Entities
        for entity_type in ["services", "skillsets", "skills"]:
            if taxonomy_data[entity_type]:
                entity_df = pd.json_normalize(taxonomy_data[entity_type])
                parquet_path = base_path / f"{entity_type}.parquet"
                entity_df.to_parquet(parquet_path, index=False)
        
        # Edges
        if taxonomy_data["edges"]:
            edges_df = pd.DataFrame(taxonomy_data["edges"])
            edges_path = base_path / "edges.parquet"
            edges_df.to_parquet(edges_path, index=False)
        
        # Suggested edges
        if taxonomy_data["suggested_edges"]:
            suggestions_df = pd.DataFrame(taxonomy_data["suggested_edges"])
            suggestions_path = base_path / "suggested_edges.parquet"
            suggestions_df.to_parquet(suggestions_path, index=False)
        
        logger.info("Taxonomy saved as Parquet files")


class TaxonomyQuery:
    """Query interface for the bidirectional taxonomy."""
    
    def __init__(self, taxonomy_builder: TaxonomyBuilder):
        self.builder = taxonomy_builder
        self.graph = taxonomy_builder.graph
        self.entities = taxonomy_builder.entities
    
    def skills_for_service(self, service_name: str, k: int = 20) -> List[Dict]:
        """Get skills for a given service."""
        service_id = self._find_entity_id_by_name(service_name, "services")
        
        if not service_id:
            return []
        
        # Find connected skills (direct and through skillsets)
        connected_skills = []
        
        for neighbor in self.graph.neighbors(service_id):
            neighbor_type = self.graph.nodes[neighbor]["type"]
            edge_data = self.graph[service_id][neighbor]
            
            if neighbor_type == "skill":
                # Direct service-skill connection
                skill_data = self.entities["skills"][neighbor]
                connected_skills.append({
                    "skill_id": neighbor,
                    "skill_name": skill_data["name"],
                    "description": skill_data.get("description", ""),
                    "category": skill_data.get("category", ""),
                    "weight": edge_data.get("weight", 1.0),
                    "confidence": edge_data.get("confidence", 1.0),
                    "connection_type": edge_data.get("type", "direct")
                })
            
            elif neighbor_type == "skillset":
                # Through skillset
                skillset_weight = edge_data.get("weight", 1.0)
                
                for skill_neighbor in self.graph.neighbors(neighbor):
                    if self.graph.nodes[skill_neighbor]["type"] == "skill":
                        skill_edge_data = self.graph[neighbor][skill_neighbor]
                        skill_data = self.entities["skills"][skill_neighbor]
                        
                        # Combined weight through skillset
                        combined_weight = skillset_weight * skill_edge_data.get("weight", 1.0)
                        
                        connected_skills.append({
                            "skill_id": skill_neighbor,
                            "skill_name": skill_data["name"],
                            "description": skill_data.get("description", ""),
                            "category": skill_data.get("category", ""),
                            "weight": combined_weight,
                            "confidence": min(edge_data.get("confidence", 1.0), 
                                            skill_edge_data.get("confidence", 1.0)),
                            "connection_type": "through_skillset",
                            "skillset_name": self.entities["skillsets"][neighbor]["name"]
                        })
        
        # Remove duplicates and sort by weight
        unique_skills = {}
        for skill in connected_skills:
            skill_id = skill["skill_id"]
            if skill_id not in unique_skills or skill["weight"] > unique_skills[skill_id]["weight"]:
                unique_skills[skill_id] = skill
        
        sorted_skills = sorted(unique_skills.values(), key=lambda x: x["weight"], reverse=True)
        
        return sorted_skills[:k]
    
    def services_for_skill(self, skill_name: str, k: int = 20) -> List[Dict]:
        """Get services for a given skill."""
        skill_id = self._find_entity_id_by_name(skill_name, "skills")
        
        if not skill_id:
            return []
        
        connected_services = []
        
        for neighbor in self.graph.neighbors(skill_id):
            neighbor_type = self.graph.nodes[neighbor]["type"]
            edge_data = self.graph[skill_id][neighbor]
            
            if neighbor_type == "service":
                # Direct skill-service connection
                service_data = self.entities["services"][neighbor]
                connected_services.append({
                    "service_id": neighbor,
                    "service_name": service_data["name"],
                    "category": service_data.get("category", ""),
                    "description": service_data.get("description", ""),
                    "weight": edge_data.get("weight", 1.0),
                    "confidence": edge_data.get("confidence", 1.0),
                    "connection_type": edge_data.get("type", "direct")
                })
            
            elif neighbor_type == "skillset":
                # Through skillset
                skillset_weight = edge_data.get("weight", 1.0)
                
                for service_neighbor in self.graph.neighbors(neighbor):
                    if self.graph.nodes[service_neighbor]["type"] == "service":
                        service_edge_data = self.graph[neighbor][service_neighbor]
                        service_data = self.entities["services"][service_neighbor]
                        
                        combined_weight = skillset_weight * service_edge_data.get("weight", 1.0)
                        
                        connected_services.append({
                            "service_id": service_neighbor,
                            "service_name": service_data["name"],
                            "category": service_data.get("category", ""),
                            "description": service_data.get("description", ""),
                            "weight": combined_weight,
                            "confidence": min(edge_data.get("confidence", 1.0),
                                            service_edge_data.get("confidence", 1.0)),
                            "connection_type": "through_skillset",
                            "skillset_name": self.entities["skillsets"][neighbor]["name"]
                        })
        
        # Remove duplicates and sort
        unique_services = {}
        for service in connected_services:
            service_id = service["service_id"]
            if service_id not in unique_services or service["weight"] > unique_services[service_id]["weight"]:
                unique_services[service_id] = service
        
        sorted_services = sorted(unique_services.values(), key=lambda x: x["weight"], reverse=True)
        
        return sorted_services[:k]
    
    def _find_entity_id_by_name(self, name: str, entity_type: str) -> Optional[str]:
        """Find entity ID by name with fuzzy matching."""
        name_lower = name.lower().strip()
        
        # Exact match first
        for entity_id, entity_data in self.entities[entity_type].items():
            if entity_data["name"].lower() == name_lower:
                return entity_id
        
        # Fuzzy match
        for entity_id, entity_data in self.entities[entity_type].items():
            entity_name_lower = entity_data["name"].lower()
            
            # Substring match
            if name_lower in entity_name_lower or entity_name_lower in name_lower:
                return entity_id
            
            # Check aliases/synonyms
            for alias in entity_data.get("aliases", []):
                if alias.lower() == name_lower:
                    return entity_id
            
            for synonym in entity_data.get("synonyms", []):
                if synonym.lower() == name_lower:
                    return entity_id
        
        return None
    
    def get_taxonomy_stats(self) -> Dict[str, Any]:
        """Get comprehensive taxonomy statistics."""
        return {
            "entities": {
                "services": len(self.entities["services"]),
                "skillsets": len(self.entities["skillsets"]),
                "skills": len(self.entities["skills"])
            },
            "mappings": {
                "explicit_service_skillset": len([m for m in self.builder.mappings["service_skillset"]]),
                "explicit_skillset_skill": len([m for m in self.builder.mappings["skillset_skill"]]),
                "derived_service_skill": len([m for m in self.builder.mappings["service_skill"]]),
                "suggested_edges": len(self.builder.suggested_edges)
            },
            "graph_metrics": {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
                "density": nx.density(self.graph),
                "connected_components": nx.number_connected_components(self.graph)
            }
        }


def build_taxonomy_from_data(data: Dict[str, pd.DataFrame]) -> Tuple[TaxonomyBuilder, TaxonomyQuery]:
    """Build taxonomy from processed data."""
    logger.info("Building taxonomy from processed data")
    
    builder = TaxonomyBuilder()
    
    # Find relevant DataFrames
    services_df = None
    skillsets_df = None
    skills_df = None
    service_skillset_df = None
    skillset_skill_df = None
    
    # Combine all skill datasets
    skill_dfs = []
    skillset_dfs = []
    
    # Match DataFrames to entity types
    for name, df in data.items():
        name_lower = name.lower()
        
        if 'service' in name_lower and 'skillset' in name_lower:
            service_skillset_df = df
        elif name_lower.startswith('skill_'):
            skill_dfs.append(df)
        elif name_lower.startswith('skillset_'):
            skillset_dfs.append(df)
        elif 'service' in name_lower and 'skillset' not in name_lower:
            services_df = df
    
    # Combine skill datasets
    if skill_dfs:
        skills_df = pd.concat(skill_dfs, ignore_index=True)
        logger.info(f"Combined {len(skill_dfs)} skill datasets into {skills_df.shape}")
    
    # Combine skillset datasets  
    if skillset_dfs:
        skillsets_df = pd.concat(skillset_dfs, ignore_index=True)
        logger.info(f"Combined {len(skillset_dfs)} skillset datasets into {skillsets_df.shape}")
    
    # Create services DataFrame from service_skillset data if no direct services found
    if services_df is None and service_skillset_df is not None:
        # Extract unique services from the mapping
        service_columns = [col for col in service_skillset_df.columns 
                          if 'service' in col.lower() and 'name' in col.lower()]
        if service_columns:
            services_data = service_skillset_df[service_columns + [col for col in service_skillset_df.columns 
                                                                  if 'focus' in col.lower() or 'category' in col.lower()]].drop_duplicates()
            services_df = services_data.rename(columns={service_columns[0]: 'service_name'})
            logger.info(f"Created services DataFrame from mappings: {services_df.shape}")
    
    # Build taxonomy
    has_data = (
        (services_df is not None and not services_df.empty) or 
        (skillsets_df is not None and not skillsets_df.empty) or 
        (skills_df is not None and not skills_df.empty)
    )
    
    if has_data:
        stats = builder.build_taxonomy(
            services_df if services_df is not None else pd.DataFrame(),
            skillsets_df if skillsets_df is not None else pd.DataFrame(),
            skills_df if skills_df is not None else pd.DataFrame(),
            service_skillset_df,
            skillset_skill_df
        )
        
        # Save taxonomy
        taxonomy_path = builder.save_taxonomy()
        
        # Save stats
        save_metrics(stats, "taxonomy_stats.json")
        
        # Create query interface
        query_interface = TaxonomyQuery(builder)
        
        logger.info("Taxonomy building completed successfully")
        
        return builder, query_interface
    
    else:
        logger.error("No suitable data found for taxonomy building")
        raise ValueError("No suitable data found for taxonomy building")


def main():
    """Main function to run taxonomy building pipeline."""
    logger.info("Starting taxonomy building pipeline")
    
    # Load processed data
    from .io_loader import load_processed_data
    data = load_processed_data()
    
    if not data:
        logger.error("No processed data found. Run previous pipeline steps first.")
        return
    
    # Build taxonomy
    builder, query_interface = build_taxonomy_from_data(data)
    
    # Test queries
    logger.info("Testing taxonomy queries")
    
    # Get some sample entities for testing
    sample_services = list(builder.entities["services"].values())[:3]
    sample_skills = list(builder.entities["skills"].values())[:3]
    
    test_results = {}
    
    for service in sample_services:
        skills = query_interface.skills_for_service(service["name"], k=5)
        test_results[f"skills_for_{service['name']}"] = len(skills)
    
    for skill in sample_skills:
        services = query_interface.services_for_skill(skill["name"], k=5)
        test_results[f"services_for_{skill['name']}"] = len(services)
    
    # Save test results
    save_metrics(test_results, "taxonomy_queries_test.json")
    
    # Get final stats
    final_stats = query_interface.get_taxonomy_stats()
    save_metrics(final_stats, "taxonomy_final_stats.json")
    
    logger.info("Taxonomy pipeline completed successfully")
    
    return builder, query_interface


if __name__ == "__main__":
    main()
