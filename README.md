# HPE Deal Intelligence Platform

Transform opportunities into strategic advantage through complete opportunity-to-skills visibility with high-performance SQLite database backend.

## Quick Start

```bash
# Launch the application launcher
python3 launch_apps.py

# Or run specific apps directly
streamlit run apps/opportunity_chain_db.py        # Database version (FASTEST)
streamlit run apps/opportunity_chain_complete.py  # Excel version
```

## Project Structure

```
HPE/Heatmap/
├── apps/                      # Streamlit applications
│   ├── opportunity_chain_db.py           # SQLite database version (34.6x faster)
│   ├── opportunity_chain_complete.py     # Complete chain visualization
│   ├── opportunity_skills_platform.py    # Opportunity-to-skills mapping
│   ├── skills_chain_platform.py          # Clean skills chain
│   ├── deal_skills_chain_platform_db.py  # Legacy database version
│   └── deal_intelligence_platform.py     # Legacy dashboard
│
├── data/                      # Source data and database
│   ├── heatmap.db                        # SQLite database (17.06 MB)
│   ├── Opportunioty and PL sample.xlsx   # 22,002 opportunities
│   ├── Services_to_skillsets Mapping.xlsx    # Service mappings
│   ├── Skillsets_to_Skills_mapping.xlsx      # Skill definitions
│   └── DETAILS (28).xlsx                     # Employee skills with ratings
│
├── scripts/                   # Analysis and utility scripts
│   ├── create_heatmap_db.py             # Database creation script
│   ├── validate_complete_chain.py       # Chain validation
│   ├── analyze_broken_chains.py         # Gap analysis
│   └── ...
│
├── docs/                      # Documentation
├── launch_apps.py            # Main application launcher
└── requirements.txt          # Python dependencies
```

## Key Features

### 1. Opportunity-to-Skills Chain Platform (Database Version)
- **34.6x faster** than Excel version using SQLite backend
- Complete 6-step chain: Opportunities → Product Lines → Services → Skillsets → Skills → Resources
- Tabbed interface for better organization
- Real-time resource drill-down with proficiency ratings (2-5)
- Direct PL code mapping (6 matched PLs)

### 2. Opportunity Chain (Excel Version)
- Same functionality as database version
- Direct Excel file processing
- Interactive Sankey visualizations
- Skills gap analysis and matching

### 3. Clean Skills Chain Platform
- Direct Product Line to Skills mapping
- No deals/focus area dependencies
- Resource availability analysis
- Clean architecture

## Database Performance

### SQLite Implementation (heatmap.db)
- **Database Size**: 17.06 MB
- **Tables**: 4 (opportunities, services_skillsets, skillsets_skills, employee_skills)
- **Performance**: 10-100x faster than Excel
- **Indexes**: Optimized for key columns

### Table Statistics:
- **opportunities**: 22,002 records
- **services_skillsets**: 226 mappings
- **skillsets_skills**: 3,451 mappings
- **employee_skills**: 6,654 skill records

## Key Findings

- **22,002** opportunities tracked
- **6** Product Lines with direct matches
- **179** unique skillsets mapped
- **505** unique resources with skills
- **Proficiency Ratings**: Numeric scale (2-5) from Proficieny_Rating column

## Recent Updates

### Database Migration
- **SQLite Database**: Created `heatmap.db` with all Excel data
- **Performance**: 34.6x faster load times (0.09s vs 3.1s)
- **Direct PL Mapping**: Only direct code matches used (6 PLs)
- **Fixed Ratings**: Using Proficieny_Rating column for numeric values (2-5)

### Tabbed Interface
- **Tab 1**: Opportunity Selection and Overview
- **Tab 2**: Product Line Analysis
- **Tab 3**: Service Requirements
- **Tab 4**: Skill Analysis
- **Tab 5**: Resource Matching with drill-down

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Create SQLite database from Excel files
python3 scripts/create_heatmap_db.py

# Launch application selector
python3 launch_apps.py
```

## Usage

### For Fast Database Version
1. Run `python3 launch_apps.py`
2. Select option 1 for Database version (FASTEST)
3. Navigate through tabs to explore the complete chain

### For Excel-Based Analysis
1. Run `python3 launch_apps.py`
2. Select option 2 for Excel version
3. Same functionality, loads directly from Excel files

### For Clean Skills Chain
1. Run `python3 launch_apps.py`
2. Select option 3 for Skills Chain Platform
3. Direct PL to Skills mapping without opportunity dependencies

## Product Line Mapping

Only direct code matches are used:
| Opportunity PL Code | Service PL Code | Product Line Name |
|---------------------|-----------------|-------------------|
| 60 | 60 (IJ) | Cloud-Native Pltfms |
| 1Z | 1Z (PN) | Network |
| 5V | 5V (II) | Hybrid Workplace |
| 4J | 4J (SX) | Education Services |
| G4 | G4 (PK) | Private Platforms |
| PD | PD (C8) | HPE POD Modular DC |

## Excel Reports

Find comprehensive Excel reports in `reports/` directory:
- **PL_MAPPING_TABLE_DIRECT_ONLY.md**: Direct PL mapping documentation
- Additional reports from previous analyses

## Database Setup

To recreate the SQLite database:
```bash
cd scripts
python3 create_heatmap_db.py
```

This will:
1. Load all Excel files from `data/` directory
2. Create `heatmap.db` with 4 tables
3. Add performance indexes
4. Display database statistics

## Support

For issues or questions:
- Database creation: `scripts/create_heatmap_db.py`
- Application logic: `apps/opportunity_chain_db.py`
- Documentation: `docs/` directory

---

*Platform Version: 3.0*
*Data Current As Of: January 2025*
*Total Opportunities: 22,002*