"""
Data loading and schema harmonization module.

This module handles loading Excel files, detecting schemas, and creating
standardized parquet files with unified column names and data types.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging
import warnings
from dataclasses import dataclass

from .utils import config, logger, Timer, validate_dataframe, save_metrics

warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class EntitySchema:
    """Schema definition for standardized entities."""
    name: str
    required_columns: List[str]
    optional_columns: List[str]
    id_column: str
    
    def all_columns(self) -> List[str]:
        return self.required_columns + self.optional_columns


# Define canonical schemas for each entity type
SCHEMAS = {
    "opportunity": EntitySchema(
        name="opportunity",
        required_columns=["opportunity_id", "title", "status"],
        optional_columns=["description", "client", "industry", "geography", 
                         "created_at", "tags", "service_labels", "practice_name",
                         "delivery_method", "delivery_location", "assigned_resource"],
        id_column="opportunity_id"
    ),
    
    "resource": EntitySchema(
        name="resource",
        required_columns=["resource_id", "resource_name"],
        optional_columns=["email", "manager", "practice", "location", "domain", 
                         "subdomain", "evaluation_date"],
        id_column="resource_id"
    ),
    
    "skill": EntitySchema(
        name="skill",
        required_columns=["skill_id", "skill_name"],
        optional_columns=["description", "category", "hierarchy", "skillset_id",
                         "technical_domain", "synonyms"],
        id_column="skill_id"
    ),
    
    "skillset": EntitySchema(
        name="skillset",
        required_columns=["skillset_id", "skillset_name"],
        optional_columns=["description", "parent_id", "category", "focus_area"],
        id_column="skillset_id"
    ),
    
    "service": EntitySchema(
        name="service",
        required_columns=["service_id", "service_name"],
        optional_columns=["category", "description", "aliases", "focus_area", "product_line"],
        id_column="service_id"
    ),
    
    "resource_skill": EntitySchema(
        name="resource_skill",
        required_columns=["resource_id", "skill_id"],
        optional_columns=["rating", "proficiency", "certification", "evaluation_date"],
        id_column=None
    ),
    
    "service_skillset": EntitySchema(
        name="service_skillset",
        required_columns=["service_id", "skillset_id"],
        optional_columns=["mandatory", "weight", "confidence"],
        id_column=None
    ),
    
    "skillset_skill": EntitySchema(
        name="skillset_skill",
        required_columns=["skillset_id", "skill_id"],
        optional_columns=["weight", "hierarchy_level"],
        id_column=None
    )
}


class ExcelLoader:
    """Load and analyze Excel files with schema detection."""
    
    def __init__(self, data_path: str = None):
        self.data_path = Path(data_path or config.data_raw_path)
        self.excel_files = list(self.data_path.glob("*.xlsx"))
        self.sheet_data = {}
        self.column_mappings = {}
        
        logger.info(f"Found {len(self.excel_files)} Excel files in {self.data_path}")
    
    def load_all_excel_files(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Load all Excel files and return dict of {filename: {sheet: dataframe}}."""
        all_data = {}
        
        with Timer(logger, "loading all Excel files"):
            for file_path in self.excel_files:
                try:
                    file_data = self._load_excel_file(file_path)
                    all_data[file_path.stem] = file_data
                    logger.info(f"Loaded {file_path.name} with {len(file_data)} sheets")
                except Exception as e:
                    logger.error(f"Failed to load {file_path.name}: {e}")
        
        self.sheet_data = all_data
        return all_data
    
    def _load_excel_file(self, file_path: Path) -> Dict[str, pd.DataFrame]:
        """Load a single Excel file and return dict of {sheet: dataframe}."""
        try:
            excel_file = pd.ExcelFile(file_path)
            sheets = {}
            
            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    
                    # Basic cleaning
                    df = self._basic_clean_dataframe(df)
                    
                    if not df.empty:
                        sheets[sheet_name] = df
                        logger.debug(f"  Sheet '{sheet_name}': {df.shape}")
                    else:
                        logger.warning(f"  Sheet '{sheet_name}' is empty")
                        
                except Exception as e:
                    logger.warning(f"  Failed to load sheet '{sheet_name}': {e}")
            
            return sheets
            
        except Exception as e:
            logger.error(f"Failed to load Excel file {file_path}: {e}")
            return {}
    
    def _basic_clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply basic cleaning to DataFrame."""
        if df.empty:
            return df
        
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Clean column names
        df.columns = df.columns.astype(str)
        df.columns = [col.strip().replace('\n', ' ').replace('\r', ' ') 
                     for col in df.columns]
        
        # Remove duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]
        
        return df
    
    def analyze_schemas(self) -> Dict[str, Any]:
        """Analyze loaded data to understand schemas and relationships."""
        if not self.sheet_data:
            self.load_all_excel_files()
        
        analysis = {
            "files": {},
            "potential_entities": {},
            "column_analysis": {},
            "join_analysis": {}
        }
        
        # Analyze each file and sheet
        for filename, sheets in self.sheet_data.items():
            file_analysis = {
                "sheets": {},
                "total_rows": 0,
                "total_columns": 0
            }
            
            for sheet_name, df in sheets.items():
                sheet_analysis = self._analyze_sheet(df, f"{filename}.{sheet_name}")
                file_analysis["sheets"][sheet_name] = sheet_analysis
                file_analysis["total_rows"] += df.shape[0]
                file_analysis["total_columns"] += df.shape[1]
            
            analysis["files"][filename] = file_analysis
        
        # Detect potential entities
        analysis["potential_entities"] = self._detect_entities()
        
        # Analyze column patterns
        analysis["column_analysis"] = self._analyze_column_patterns()
        
        # Analyze potential joins
        analysis["join_analysis"] = self._analyze_join_potential()
        
        return analysis
    
    def _analyze_sheet(self, df: pd.DataFrame, sheet_id: str) -> Dict[str, Any]:
        """Analyze a single sheet."""
        analysis = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "null_percentages": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            "unique_counts": {},
            "sample_values": {},
            "potential_ids": [],
            "potential_text_fields": [],
            "potential_categorical": []
        }
        
        # Analyze each column
        for col in df.columns:
            try:
                series = df[col].dropna()
                if len(series) == 0:
                    continue
                
                unique_count = series.nunique()
                analysis["unique_counts"][col] = unique_count
                
                # Sample values (first 5 unique non-null values)
                sample_values = series.unique()[:5].tolist()
                analysis["sample_values"][col] = [str(val) for val in sample_values]
                
                # Detect potential ID columns
                if (unique_count == len(series) and 
                    (col.lower().endswith('_id') or col.lower().endswith('id') or 
                     col.lower().startswith('id') or 'number' in col.lower())):
                    analysis["potential_ids"].append(col)
                
                # Detect text fields
                if (df[col].dtype == 'object' and 
                    series.str.len().mean() > 20):
                    analysis["potential_text_fields"].append(col)
                
                # Detect categorical fields
                if (unique_count < len(series) * 0.1 and unique_count < 50):
                    analysis["potential_categorical"].append(col)
                    
            except Exception as e:
                logger.warning(f"Error analyzing column {col}: {e}")
        
        return analysis
    
    def _detect_entities(self) -> Dict[str, List[str]]:
        """Detect which sheets likely contain which entities."""
        entity_mapping = {}
        
        for entity_name, schema in SCHEMAS.items():
            entity_mapping[entity_name] = []
            
            for filename, sheets in self.sheet_data.items():
                for sheet_name, df in sheets.items():
                    score = self._calculate_entity_match_score(df, schema)
                    
                    if score > 0.3:  # Threshold for potential match
                        sheet_id = f"{filename}.{sheet_name}"
                        entity_mapping[entity_name].append({
                            "sheet": sheet_id,
                            "score": score,
                            "shape": df.shape
                        })
        
        # Sort by score
        for entity_name in entity_mapping:
            entity_mapping[entity_name].sort(key=lambda x: x["score"], reverse=True)
        
        return entity_mapping
    
    def _calculate_entity_match_score(self, df: pd.DataFrame, schema: EntitySchema) -> float:
        """Calculate how well a DataFrame matches an entity schema."""
        if df.empty:
            return 0.0
        
        columns_lower = [col.lower() for col in df.columns]
        
        # Score based on matching required columns (fuzzy matching)
        required_matches = 0
        for req_col in schema.required_columns:
            req_col_lower = req_col.lower()
            
            # Exact match
            if req_col_lower in columns_lower:
                required_matches += 1
                continue
            
            # Fuzzy match - check if any column contains the required column name
            fuzzy_matches = [col for col in columns_lower 
                           if req_col_lower in col or col in req_col_lower]
            if fuzzy_matches:
                required_matches += 0.7
        
        # Score based on matching optional columns
        optional_matches = 0
        for opt_col in schema.optional_columns:
            opt_col_lower = opt_col.lower()
            
            if opt_col_lower in columns_lower:
                optional_matches += 1
            else:
                fuzzy_matches = [col for col in columns_lower 
                               if opt_col_lower in col or col in opt_col_lower]
                if fuzzy_matches:
                    optional_matches += 0.5
        
        # Calculate final score
        required_score = required_matches / len(schema.required_columns) if schema.required_columns else 0
        optional_score = optional_matches / len(schema.optional_columns) if schema.optional_columns else 0
        
        # Weight required columns more heavily
        final_score = 0.7 * required_score + 0.3 * optional_score
        
        return final_score
    
    def _analyze_column_patterns(self) -> Dict[str, Any]:
        """Analyze common column patterns across all sheets."""
        all_columns = []
        column_frequencies = {}
        
        for filename, sheets in self.sheet_data.items():
            for sheet_name, df in sheets.items():
                for col in df.columns:
                    col_normalized = col.lower().strip()
                    all_columns.append(col_normalized)
                    column_frequencies[col_normalized] = column_frequencies.get(col_normalized, 0) + 1
        
        # Find common patterns
        common_columns = {k: v for k, v in column_frequencies.items() if v > 1}
        
        return {
            "total_unique_columns": len(set(all_columns)),
            "common_columns": common_columns,
            "most_frequent_columns": sorted(column_frequencies.items(), 
                                          key=lambda x: x[1], reverse=True)[:20]
        }
    
    def _analyze_join_potential(self) -> Dict[str, Any]:
        """Analyze potential joins between sheets."""
        join_analysis = {
            "potential_joins": [],
            "join_success_rates": {}
        }
        
        sheets_list = []
        for filename, sheets in self.sheet_data.items():
            for sheet_name, df in sheets.items():
                sheets_list.append({
                    "id": f"{filename}.{sheet_name}",
                    "df": df,
                    "filename": filename,
                    "sheet_name": sheet_name
                })
        
        # Check all pairs of sheets
        for i, sheet1 in enumerate(sheets_list):
            for j, sheet2 in enumerate(sheets_list):
                if i >= j:  # Avoid duplicates and self-joins
                    continue
                
                join_info = self._analyze_sheet_join(sheet1, sheet2)
                if join_info["best_join_rate"] > 0.1:  # Only include promising joins
                    join_analysis["potential_joins"].append(join_info)
        
        # Sort by join success rate
        join_analysis["potential_joins"].sort(key=lambda x: x["best_join_rate"], reverse=True)
        
        return join_analysis
    
    def _analyze_sheet_join(self, sheet1: Dict, sheet2: Dict) -> Dict[str, Any]:
        """Analyze potential join between two sheets."""
        df1, df2 = sheet1["df"], sheet2["df"]
        
        join_info = {
            "sheet1": sheet1["id"],
            "sheet2": sheet2["id"],
            "potential_keys": [],
            "best_join_rate": 0.0
        }
        
        # Check all column pairs for potential joins
        for col1 in df1.columns:
            for col2 in df2.columns:
                try:
                    # Skip if columns have very different names (unless they're ID columns)
                    col1_lower, col2_lower = col1.lower(), col2.lower()
                    
                    if (col1_lower != col2_lower and 
                        not any(id_term in col1_lower for id_term in ['id', 'number', 'name']) and
                        not any(id_term in col2_lower for id_term in ['id', 'number', 'name'])):
                        continue
                    
                    # Calculate join success rate
                    series1 = df1[col1].dropna().astype(str)
                    series2 = df2[col2].dropna().astype(str)
                    
                    if len(series1) == 0 or len(series2) == 0:
                        continue
                    
                    common_values = set(series1) & set(series2)
                    join_rate = len(common_values) / max(len(set(series1)), len(set(series2)))
                    
                    if join_rate > 0.1:  # Minimum threshold
                        join_info["potential_keys"].append({
                            "col1": col1,
                            "col2": col2,
                            "join_rate": join_rate,
                            "common_values": len(common_values)
                        })
                        
                        join_info["best_join_rate"] = max(join_info["best_join_rate"], join_rate)
                
                except Exception as e:
                    continue  # Skip problematic column pairs
        
        # Sort potential keys by join rate
        join_info["potential_keys"].sort(key=lambda x: x["join_rate"], reverse=True)
        
        return join_info
    
    def create_column_mappings(self) -> Dict[str, Dict[str, str]]:
        """Create mappings from actual column names to standardized names."""
        mappings = {}
        
        entity_detection = self._detect_entities()
        
        for entity_name, matches in entity_detection.items():
            if not matches:
                continue
            
            schema = SCHEMAS[entity_name]
            
            # Use the best matching sheet for this entity
            best_match = matches[0]
            sheet_id = best_match["sheet"]
            
            # Find the DataFrame
            filename, sheet_name = sheet_id.split('.', 1)
            df = self.sheet_data[filename][sheet_name]
            
            # Create column mapping
            mapping = self._create_column_mapping(df.columns, schema)
            mappings[sheet_id] = mapping
            
            logger.info(f"Created mapping for {entity_name} from {sheet_id}")
            logger.debug(f"  Mapping: {mapping}")
        
        self.column_mappings = mappings
        return mappings
    
    def _create_column_mapping(self, actual_columns: List[str], 
                              schema: EntitySchema) -> Dict[str, str]:
        """Create mapping from actual columns to schema columns."""
        mapping = {}
        actual_lower = [col.lower() for col in actual_columns]
        
        all_schema_columns = schema.all_columns()
        
        for schema_col in all_schema_columns:
            schema_col_lower = schema_col.lower()
            
            # Try exact match first
            if schema_col_lower in actual_lower:
                idx = actual_lower.index(schema_col_lower)
                mapping[actual_columns[idx]] = schema_col
                continue
            
            # Try fuzzy matching
            best_match = None
            best_score = 0
            
            for i, actual_col in enumerate(actual_columns):
                actual_col_lower = actual_col.lower()
                
                # Check if schema column is contained in actual column
                if schema_col_lower in actual_col_lower:
                    score = len(schema_col_lower) / len(actual_col_lower)
                    if score > best_score:
                        best_score = score
                        best_match = actual_col
                
                # Check if actual column is contained in schema column
                elif actual_col_lower in schema_col_lower:
                    score = len(actual_col_lower) / len(schema_col_lower)
                    if score > best_score:
                        best_score = score
                        best_match = actual_col
            
            if best_match and best_score > 0.3:  # Minimum similarity threshold
                mapping[best_match] = schema_col
        
        return mapping


def harmonize_dataframes(loader: ExcelLoader) -> Dict[str, pd.DataFrame]:
    """Harmonize all DataFrames to standard schemas and save as parquet."""
    
    if not loader.column_mappings:
        loader.create_column_mappings()
    
    harmonized_data = {}
    
    with Timer(logger, "harmonizing dataframes"):
        
        # Process ALL sheets, not just mapped ones
        for filename, sheets in loader.sheet_data.items():
            for sheet_name, df in sheets.items():
                sheet_id = f"{filename}.{sheet_name}"
                df_copy = df.copy()
                
                # Apply column mapping if available
                column_mapping = loader.column_mappings.get(sheet_id, {})
                if column_mapping:
                    df_harmonized = df_copy.rename(columns=column_mapping)
                else:
                    df_harmonized = df_copy
                
                # Determine entity type based on best match
                entity_type = None
                best_score = 0
                
                for ent_name, schema in SCHEMAS.items():
                    score = loader._calculate_entity_match_score(df_harmonized, schema)
                    if score > best_score:
                        best_score = score
                        entity_type = ent_name
                
                # Process if we have a reasonable match OR if it contains important data
                if (entity_type and best_score > 0.2) or df_harmonized.shape[0] > 100:
                    if not entity_type:
                        # Assign generic entity type based on content
                        entity_type = _infer_entity_type_from_content(df_harmonized, sheet_name)
                    
                    # Apply entity-specific processing
                    df_processed = _process_entity_dataframe(df_harmonized, entity_type)
                    
                    # Generate ID column if needed
                    schema = SCHEMAS.get(entity_type)
                    if schema and schema.id_column and schema.id_column not in df_processed.columns:
                        df_processed = _generate_id_column(df_processed, schema.id_column, entity_type)
                    
                    clean_name = f"{entity_type}_{sheet_id.replace('.', '_').replace(' ', '_').replace('(', '').replace(')', '')}"
                    harmonized_data[clean_name] = df_processed
                    
                    logger.info(f"Harmonized {sheet_id} as {entity_type}: {df_processed.shape}")
    
    # Save harmonized data
    output_path = Path(config.data_processed_path)
    output_path.mkdir(exist_ok=True)
    
    for name, df in harmonized_data.items():
        parquet_path = output_path / f"{name}.parquet"
        df.to_parquet(parquet_path, index=False)
        logger.info(f"Saved {parquet_path}")
    
    return harmonized_data


def _process_entity_dataframe(df: pd.DataFrame, entity_type: str) -> pd.DataFrame:
    """Apply entity-specific processing to DataFrame."""
    df = df.copy()
    
    if entity_type == "opportunity":
        # Process opportunity-specific fields
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        
        # Clean text fields
        text_fields = ['title', 'description', 'tags']
        for field in text_fields:
            if field in df.columns:
                df[field] = df[field].astype(str).fillna('')
                df[field] = df[field].str.strip()
    
    elif entity_type == "resource":
        # Process resource-specific fields
        if 'evaluation_date' in df.columns:
            df['evaluation_date'] = pd.to_datetime(df['evaluation_date'], errors='coerce')
    
    elif entity_type == "resource_skill":
        # Process resource-skill relationships
        if 'rating' in df.columns:
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        
        if 'evaluation_date' in df.columns:
            df['evaluation_date'] = pd.to_datetime(df['evaluation_date'], errors='coerce')
    
    # Common processing for all entities
    # Remove completely empty rows
    df = df.dropna(how='all')
    
    # Clean string columns
    string_columns = df.select_dtypes(include=['object']).columns
    for col in string_columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace('nan', np.nan)
    
    return df


def _infer_entity_type_from_content(df: pd.DataFrame, sheet_name: str) -> str:
    """Infer entity type from DataFrame content and sheet name."""
    sheet_name_lower = sheet_name.lower()
    
    # Check sheet name for hints
    if any(term in sheet_name_lower for term in ['skill', 'certification']):
        return 'skill'
    elif 'service' in sheet_name_lower:
        return 'service'
    elif 'request' in sheet_name_lower or 'opportunity' in sheet_name_lower or 'rawdata' in sheet_name_lower:
        return 'opportunity'
    elif 'resource' in sheet_name_lower or 'details' in sheet_name_lower:
        return 'resource'
    
    # Check column names for hints
    columns_lower = [col.lower() for col in df.columns]
    
    if any('skill' in col for col in columns_lower):
        return 'skill'
    elif any('service' in col for col in columns_lower):
        return 'service'
    elif any(term in col for col in columns_lower for term in ['request', 'opportunity', 'rr']):
        return 'opportunity'
    elif any(term in col for col in columns_lower for term in ['resource', 'employee', 'person']):
        return 'resource'
    
    # Default fallback
    return 'skill'  # Most common entity type in our data


def _generate_id_column(df: pd.DataFrame, id_column: str, entity_type: str) -> pd.DataFrame:
    """Generate ID column for entity."""
    df = df.copy()
    
    # Create sequential IDs with entity prefix
    prefix = entity_type[:3].upper()
    df[id_column] = [f"{prefix}_{i:06d}" for i in range(len(df))]
    
    logger.info(f"Generated {id_column} for {len(df)} {entity_type} records")
    
    return df


def load_processed_data() -> Dict[str, pd.DataFrame]:
    """Load all processed parquet files."""
    processed_path = Path(config.data_processed_path)
    
    if not processed_path.exists():
        logger.error(f"Processed data directory not found: {processed_path}")
        return {}
    
    data = {}
    parquet_files = list(processed_path.glob("*.parquet"))
    
    for file_path in parquet_files:
        try:
            df = pd.read_parquet(file_path)
            data[file_path.stem] = df
            logger.info(f"Loaded {file_path.name}: {df.shape}")
        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {e}")
    
    return data


def main():
    """Main function to run data loading and harmonization."""
    logger.info("Starting data loading and harmonization pipeline")
    
    # Load Excel files
    loader = ExcelLoader()
    
    # Analyze schemas
    with Timer(logger, "schema analysis"):
        analysis = loader.analyze_schemas()
    
    # Save analysis results
    save_metrics(analysis, "schema_analysis.json")
    logger.info("Schema analysis saved to artifacts/metrics/schema_analysis.json")
    
    # Create harmonized datasets
    harmonized_data = harmonize_dataframes(loader)
    
    # Generate data quality report
    quality_metrics = {}
    for name, df in harmonized_data.items():
        quality_metrics[name] = validate_dataframe(df, name)
    
    save_metrics(quality_metrics, "data_quality.json")
    logger.info("Data quality metrics saved to artifacts/metrics/data_quality.json")
    
    logger.info(f"Pipeline complete. Processed {len(harmonized_data)} datasets.")
    
    return harmonized_data


if __name__ == "__main__":
    main()
