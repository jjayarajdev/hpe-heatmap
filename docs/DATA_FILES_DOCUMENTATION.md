# Data Files Documentation

## Overview
The HPE Opportunity Intelligence Platform uses both **Excel source files** and a **SQLite database**. The database provides 34.6x faster performance while Excel files serve as the data source.

---

## Primary Data Files

### 1. **SQLite Database (Recommended)**
**File:** `data/heatmap.db`
- **Size:** 17.06 MB
- **Purpose:** Optimized database for fast queries
- **Tables:** 4 tables with 32,133 total records
- **Performance:** <0.1s query time
- **Used by:** opportunity_chain_db.py

### 2. **Opportunities Data**
**File:** `data/HPE deals FY25.xlsx`
- **Purpose:** Contains 22,002 HPE opportunities
- **Key Columns:**
  - HPE Opportunity Id
  - Opportunity Name
  - Product Line (PL code)
  - Stage
  - Schedule Amount (converted)
- **Used by:** opportunity_chain_complete.py (Excel version)

### 3. **Services to Skillsets Mapping**
**File:** `data/Services_to_skillsets Mapping.xlsx`
- **Size:** 68 KB
- **Sheet:** Master v5
- **Purpose:** Maps Product Lines → Services → Skillsets
- **Key Columns:**
  - FY25 PL (e.g., "60 (IJ)")
  - New Service Name
  - Skill Set
  - Services Description
- **Records:** 226 mappings

### 4. **Skillsets to Skills Mapping**
**File:** `data/Skillsets_to_Skills_mapping.xlsx`
- **Size:** 120 KB
- **Purpose:** Maps Skillsets to individual Skills
- **Structure:** Multiple sheets by category
- **Key Columns:**
  - FY'25 Skillset Name
  - FY'24 Skillset Name (fallback when FY'25 is "No change")
  - Skills Name
- **Records:** 3,451 skill mappings

### 5. **Employee Skills Data**
**File:** `data/DETAILS (28).xlsx`
- **Size:** 764 KB (largest file)
- **Sheet:** Export
- **Purpose:** Employee skills and proficiency ratings
- **Records:** 6,654 employee skill records
- **Key Columns:**
  - Resource_Name
  - Skill_Certification_Name
  - Skill_Set_Name
  - Proficieny_Rating (2-5 scale)

---

## Data Flow

```
Opportunities (HPE deals FY25.xlsx)
    ↓ [Product Line column]
Product Lines → Services (Services_to_skillsets Mapping.xlsx)
    ↓ [Skill Set column]
Skillsets → Skills (Skillsets_to_Skills_mapping.xlsx)
    ↓ [Skills Name column]
Skills → Resources (DETAILS (28).xlsx)
    [Resource_Name with Proficieny_Rating]
```

---

## Application Usage

### opportunity_chain_db.py (Database Version)
- Uses `heatmap.db` SQLite database
- All data pre-loaded and indexed
- 34.6x faster than Excel version
- Recommended for production use

### opportunity_chain_complete.py (Excel Version)
- Reads all 4 Excel files directly
- More flexible for quick data updates
- Slower performance (3-5s load time)
- Good for data validation

---

## Important Notes

### Valid Product Line Mappings
Only 6 Product Lines have complete chains to resources:
- **60** → Cloud-Native Platforms (60 (IJ))
- **1Z** → Network (1Z (PN))
- **5V** → Hybrid Workplace (5V (II))
- **4J** → Education Services (4J (SX))
- **G4** → Private Platforms (G4 (PK))
- **PD** → HPE POD Modular DC (PD (C8))

### Database vs Excel
1. **Database (Recommended):**
   - 34.6x faster performance
   - Pre-indexed for optimal queries
   - 17MB single file
   - Created with `scripts/create_heatmap_db.py`

2. **Excel Files:**
   - Direct source data access
   - Easier manual updates
   - Multiple file reads required
   - Slower but more flexible

---

## Updating Data

### To Update Excel Files:
1. Replace files in `data/` directory
2. Keep same filenames and column structure
3. Run `python3 scripts/create_heatmap_db.py` to rebuild database
4. Restart applications

### To Query Database:
```bash
sqlite3 data/heatmap.db
```

### Column Requirements
When updating files, ensure these columns exist:

**Opportunities file:**
- HPE Opportunity Id
- Opportunity Name
- Product Line
- Schedule Amount (converted)

**Services mapping:**
- FY25 PL (format: "60 (IJ)")
- New Service Name
- Skill Set

**Skills mapping:**
- FY'25 Skillset Name (or FY'24 as fallback)
- Skills Name

**Employee data:**
- Resource_Name
- Skill_Set_Name
- Proficieny_Rating (note the typo in original data)

---

## File Relationships Summary

| File/Database | Size | Records | Primary Use |
|--------------|------|---------|-------------|
| heatmap.db | 17.06 MB | 32,133 total | Fast queries (recommended) |
| HPE deals FY25.xlsx | ~2 MB | 22,002 opportunities | Opportunity data |
| Services_to_skillsets Mapping.xlsx | 68 KB | 226 mappings | PL→Services→Skillsets |
| Skillsets_to_Skills_mapping.xlsx | 120 KB | 3,451 mappings | Skillsets→Skills |
| DETAILS (28).xlsx | 764 KB | 6,654 records | Employee skills |

---

*Last Updated: January 2025*
*Database Size: 17.06 MB*
*Total Excel Files: ~3 MB across 4 files*