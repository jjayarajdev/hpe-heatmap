# Cascading Filters Feature

## Overview
Added interactive cascading filters to the Chain Analysis tab, allowing users to drill down from opportunities to specific resources by selecting services, skillsets, and skills. Each selection automatically filters the next step in the chain.

## How It Works

### The Cascade Flow
```
Opportunity → Product Lines → [SELECT SERVICES] → [SELECT SKILLSETS] → [SELECT SKILLS] → Resources
```

1. **Start with Full Chain**: All services, skillsets, skills, and resources are shown
2. **Filter Services**: Select specific services → Only skillsets from those services are shown
3. **Filter Skillsets**: Select specific skillsets → Only skills from those skillsets are shown
4. **Filter Skills**: Select specific skills → Only resources with those skills are shown

### Example Workflow
```
Opportunity: OPE-XX1Y7XY322 (4 PLs: 1Z, 4J, 60, 96)

UNFILTERED:
  ├─ 15 Services
  ├─ 30 Skillsets
  ├─ 75 Skills
  └─ 30 Resources

SELECT 2 SERVICES:
  ├─ 2 Services ✓ (filtered)
  ├─ 21 Skillsets (cascade filtered from selected services)
  ├─ 75 Skills
  └─ 30 Resources

SELECT 2 SKILLSETS (from the 21):
  ├─ 2 Services ✓ (filtered)
  ├─ 2 Skillsets ✓ (filtered)
  ├─ 8 Skills (cascade filtered from selected skillsets)
  └─ Resources (filtered to those with the 8 skills)
```

## UI Changes

### Chain Analysis Tab Layout

#### 6-Step Header (Read-Only Summary)
Shows counts at each step based on current filters:
- Step 1: Opportunity ID
- Step 2: Product Lines (with popover showing all PLs)
- Step 3: Services count
- Step 4: Skillsets count
- Step 5: Skills count
- Step 6: Resources matched

#### Drill-Down Filter Section
Three multi-select dropdowns arranged horizontally:

| 📋 Select Services | 🎯 Select Skillsets | ⚙️ Select Skills |
|-------------------|---------------------|------------------|
| (Optional filter) | (Optional filter)   | (Optional filter)|

**Features:**
- Multi-select capability (choose multiple items)
- Searchable dropdowns (type to filter)
- Options automatically update based on previous selections
- Reset button appears when filters are active

### Visual Feedback
- Info banner explaining interactive filtering
- Real-time count updates in step headers
- Reset button (🔄) to clear all filters

## Implementation Details

### New Methods

#### `get_filtered_chain(opportunity_id, selected_services, selected_skillsets, selected_skills)`
Returns a chain filtered based on user selections with cascading logic:
- Takes the base chain from `get_complete_chain()`
- Applies service filter → recalculates skillsets
- Applies skillset filter → recalculates skills
- Applies skill filter → recalculates resources

**Location:** apps/opportunity_chain_db.py:833-884

### Session State Management
Tracks user selections across reruns:
- `current_opp_id`: Currently selected opportunity
- `selected_services`: List of selected service names
- `selected_skillsets`: List of selected skillset names
- `selected_skills`: List of selected skill names

Auto-resets when user changes opportunity.

### Cascading Logic
```python
# Services → Skillsets
if selected_services:
    chain['services'] = [s for s in chain['services'] if s in selected_services]
    chain['skillsets'] = set()
    for service in chain['services']:
        chain['skillsets'].update(self.service_to_skillsets.get(service, set()))

# Skillsets → Skills
if selected_skillsets:
    chain['skillsets'] = [s for s in chain['skillsets'] if s in selected_skillsets]
    chain['skills'] = set()
    for skillset in chain['skillsets']:
        chain['skills'].update(self.skillset_to_skills.get(skillset, set()))

# Skills → Resources
if selected_skills:
    chain['skills'] = [s for s in chain['skills'] if s in selected_skills]
    # Recalculate resource scores based on selected skills only
    ...
```

## Benefits

### 1. **Focused Analysis**
Drill down to specific areas of interest instead of seeing everything at once.

**Example:** "I only care about Cloud-Native services. Show me the skillsets, skills, and resources for those."

### 2. **Gap Identification**
Quickly see which specific skillsets have resource shortages.

**Example:** Select a service → See it needs 10 skillsets → Find only 2 have available resources.

### 3. **Resource Matching**
Find the exact people for specific skill requirements.

**Example:** Select 3 critical skills → See exactly which 5 resources have those skills.

### 4. **Requirement Scoping**
Understand the full dependency chain for specific services.

**Example:** "If I deliver these 2 services, what skillsets and resources do I need?"

## Usage Tips

### Best Practices
1. **Start Broad**: View the full chain first to understand scope
2. **Drill Down**: Select services you're interested in
3. **Refine Further**: Narrow to specific skillsets if needed
4. **Find Resources**: See exactly who can deliver

### Common Workflows

**Workflow 1: Service-Specific Staffing**
```
1. Select opportunity
2. Choose 2-3 target services
3. View required skillsets (auto-filtered)
4. See available resources (auto-filtered)
```

**Workflow 2: Skill Gap Analysis**
```
1. Select opportunity
2. Browse all skillsets
3. Select skillsets with low resource counts
4. Identify which skills are missing
```

**Workflow 3: Resource Capability Check**
```
1. Select opportunity
2. Pick critical skills (e.g., "Azure Kubernetes")
3. See all resources with those skills
4. Check their proficiency ratings
```

## Testing

Verified with opportunity OPE-XX1Y7XY322:
- ✅ Unfiltered: 15 services → 30 skillsets → 75 skills → 30 resources
- ✅ 2 services selected: 21 skillsets (cascade) → 75 skills → 30 resources
- ✅ 2 services + 2 skillsets: 8 skills (cascade) → resources (filtered)
- ✅ Session state persists across interactions
- ✅ Reset button clears all filters correctly

## Code Location

**Modified File:** `apps/opportunity_chain_db.py`

**Key Sections:**
- Lines 833-884: `get_filtered_chain()` method
- Lines 886-1071: Updated `render_chain_analysis_tab()` with cascading filters
- Lines 916-921: Session state initialization

**Added Dependencies:**
- Streamlit session state (st.session_state)
- Multi-select widgets (st.multiselect)

---

**Feature Added:** 2025-10-24
**Status:** ✅ Complete and Tested
**Compatibility:** Works with multi-PL opportunities
