# HPE Opportunity Intelligence Platform

## Overview
The HPE Opportunity Intelligence Platform provides a complete chain analysis from opportunities to required resources, helping identify skill gaps and resource availability for HPE opportunities.

## Quick Start

### 1. Setup Database (One-time)
```bash
python3 scripts/create_heatmap_db.py
```
This creates `data/heatmap.db` from Excel source files (17MB, 32,133 records).

### 2. Launch Application
```bash
python3 launch_apps.py
```
Choose from:
- **Option 1**: Database version (34.6x faster, recommended)
- **Option 2**: Excel version (direct file processing)

## Architecture

### Data Chain
```
Opportunities → Product Lines → Services → Skillsets → Skills → Resources
```

### Database Tables
- **opportunities**: 22,002 HPE opportunities
- **services_skillsets**: 226 service-to-skillset mappings
- **skillsets_skills**: 3,451 skillset-to-skill mappings
- **employee_skills**: 6,654 employee skill records

### Valid Product Line Mappings
Only 6 Product Lines have complete chains:
- **60** → Cloud-Native Platforms
- **1Z** → Network
- **5V** → Hybrid Workplace
- **4J** → Education Services
- **G4** → Private Platforms
- **PD** → HPE POD Modular DC

## Application Features

### 4-Tab Interface
1. **Overview Tab**: Opportunity selection and details
2. **Chain Analysis Tab**: Complete opportunity-to-resources chain
3. **Resources Tab**: Resource matching and proficiency ratings
4. **Search & Filter Tab**: Advanced filtering capabilities

### Key Capabilities
- Analyze 22,002 opportunities
- Identify required services and skillsets
- Match resources based on skills
- Proficiency ratings (2-Basic to 5-Expert)
- Skills gap analysis
- Export results to Excel

## Performance
- **Database Version**: <0.1s query time
- **Excel Version**: 3-5s load time
- **Improvement**: 34.6x faster with database

## Project Structure
```
HPE/Heatmap/
├── apps/
│   ├── opportunity_chain_db.py      # Database version
│   └── opportunity_chain_complete.py # Excel version
├── scripts/
│   └── create_heatmap_db.py        # Database creator
├── data/
│   ├── heatmap.db                  # SQLite database
│   └── *.xlsx                      # Source Excel files
├── docs/
│   ├── README.md                   # This file
│   ├── IMPLEMENTATION_GUIDE.md     # Usage guide
│   ├── DATABASE_ER_DIAGRAM_ENHANCED.md # Database docs
│   └── ...                         # Other documentation
└── launch_apps.py                  # Application launcher
```

## Requirements
- Python 3.8+
- Streamlit
- pandas
- sqlite3
- openpyxl

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage Examples

### Find Resources for an Opportunity
1. Select opportunity in Overview tab
2. Navigate to Chain Analysis tab
3. View required services and skillsets
4. Check Resources tab for matches

### Identify Skills Gaps
1. Go to Search & Filter tab
2. Filter by Product Line
3. View skillsets with low resource coverage
4. Export gap analysis

## Notes
- Only opportunities with valid Product Line codes will show complete chains
- Database version recommended for performance
- Proficiency ratings: 2 (Basic) to 5 (Expert)
- Use minimum proficiency 3 for resource matching

## Support
For issues or questions, refer to the IMPLEMENTATION_GUIDE.md or contact the platform team.

---
*Version 3.0 | January 2025*