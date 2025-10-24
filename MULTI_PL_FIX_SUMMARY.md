# Multi-PL Opportunity Fix Summary

## Problem
The application was only showing **one Product Line (PL)** per opportunity, even though many opportunities have multiple PLs in the database.

### Root Cause
The `build_complete_chain()` method was using a dictionary with `opportunity_id` as the key. When an opportunity had multiple PLs (stored as separate rows in the database), each subsequent row would **overwrite** the previous one, keeping only the last PL encountered.

```python
# OLD CODE (BUGGY)
self.opportunity_to_pl[opp_id] = {
    'pl_code': pl_code,  # Single value - gets overwritten!
    ...
}
```

## Solution
Changed data structures to store **all PLs per opportunity** as lists and aggregate services/skillsets/skills from all PLs.

### Code Changes

#### 1. Data Structure (opportunity_chain_db.py:285-318)
```python
# NEW CODE (FIXED)
if opp_id not in self.opportunity_to_pl:
    self.opportunity_to_pl[opp_id] = {
        'pl_codes': [],   # Changed to list
        'pl_fulls': [],   # Changed to list
        ...
    }

# Append each PL instead of overwriting
self.opportunity_to_pl[opp_id]['pl_codes'].append(pl_code)
self.opportunity_to_pl[opp_id]['pl_fulls'].append(pl_full)
```

#### 2. Chain Aggregation (opportunity_chain_db.py:397-457)
- Changed `get_complete_chain()` to iterate through ALL PLs
- Aggregates services from all PLs (union of all services)
- Increased limits: 20 services, 30 skillsets, 75 skills, 30 resources
- Results in MORE comprehensive resource matching

```python
# Aggregate services from ALL Product Lines
for pl_code in pl_codes:
    pl_services = self.pl_to_services.get(pl_code, set())
    chain['services'].update(pl_services)
```

#### 3. UI Updates (opportunity_chain_db.py:872-890)
- Changed PL display from single code to a **popover** showing count
- Click the popover to see all PLs with full descriptions
- Example: "4 PLs" (click to expand and see: 1Z - Network, 4J - Education Services, etc.)

#### 4. Sankey Visualization (opportunity_chain_db.py:606-653)
- Updated to show **multiple PL nodes** flowing from one opportunity
- Services connect to all applicable PLs
- Value is distributed across PLs proportionally

#### 5. Coverage Metrics (opportunity_chain_db.py:814-823)
- Fixed to check if ANY of the PLs are in the mapping (not just one)
- Changed from `data['pl_code']` to `any(pl_code in self.pl_mapping for pl_code in data['pl_codes'])`

## Impact

### Statistics
- **Total Opportunities:** 22,000
- **Multi-PL Opportunities:** 683 (3.1%)
- **Single-PL Opportunities:** 21,317 (96.9%)

### Example Multi-PL Opportunities (Now Fixed!)

1. **OPE-XX1Y7XY322** - GEHC EMRAD Project
   - 4 PLs: 1Z, 4J, 60, 96
   - Previously showed only 1 PL ❌
   - Now shows all 4 PLs ✅

2. **OPE-XX16XXXX41** - Biotrigo Storage Cluster
   - 9 PLs: 1Z, 5V, 60, 96, G4, H3, KH, PD, SQ
   - Previously showed only 1 PL ❌
   - Now shows all 9 PLs ✅

## Benefits

1. ✅ **Complete Visibility** - See all PLs for every opportunity
2. ✅ **Better Resource Matching** - Aggregates skills from ALL PLs, finding more relevant resources
3. ✅ **Accurate Analysis** - TCV USD properly accumulates across all PL rows
4. ✅ **Clear UI** - Popover interface keeps display clean while showing full details on demand

## Testing

Run the application and select any multi-PL opportunity:
```bash
streamlit run apps/opportunity_chain_db.py
```

Navigate to "Chain Analysis" tab, select an opportunity, and click on the "Product Lines" popover in step 2 to see all PLs.

## Verification

The application now successfully loads and processes multi-PL opportunities:

```
Test Opportunity: OPE-XX1Y7XY322
Product Lines: ['1Z', '4J', '60', '96'] ✅ All 4 PLs shown!
Services: 15 (aggregated from all PLs)
Skillsets: 30 (aggregated from all PLs)
Skills: 75 (aggregated from all PLs)
Resources: 30 (matched across all PLs)
```

---

**Fixed Date:** 2025-10-24
**Modified File:** apps/opportunity_chain_db.py
**Lines Changed:** 285-318, 397-457, 872-890, 606-653, 814-823
