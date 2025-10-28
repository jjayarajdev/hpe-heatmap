# Resource Count Deduplication Fix

## Problem Discovered

User noticed incorrect resource counts for skills in the Search & Filter tab:
- "Cloud Management Platform - Morpheus Automation" showed **44 resources**
- Database query revealed only **22 unique people** have this skill

**Root Cause:** Database contained duplicate rows for some employees with the same skill, and the code was counting each row instead of unique people.

## Database Investigation

```sql
SELECT
    Skill_Certification_Name,
    COUNT(DISTINCT Resource_Name) as unique_resources,
    COUNT(*) as total_rows
FROM employee_skills
WHERE Skill_Certification_Name LIKE '%Morpheus%'
GROUP BY Skill_Certification_Name;
```

**Results:**
| Skill | Unique Resources | Total Rows | Issue |
|-------|------------------|------------|-------|
| Cloud Management Platform - Morpheus Automation | 22 | 44 | ❌ 2x duplicates |
| Morpheus Cloud Management Platform | 20 | 20 | ✅ No duplicates |
| Ability to articulate... | 2 | 2 | ✅ No duplicates |

**Analysis:** Some skills have 2 database rows per person (possibly from different data sources or updates), causing double-counting.

## The Bug

### Original Code (apps/opportunity_chain_db.py:245-279)
```python
def process_employee_data(self):
    self.employee_profiles = {}

    for _, row in self.employees_df.iterrows():
        emp_name = row.get('Resource_Name')
        if emp_name not in self.employee_profiles:
            self.employee_profiles[emp_name] = {
                'skills': [],  # ← List allows duplicates!
                ...
            }

        if pd.notna(row.get('Skill_Certification_Name')):
            # PROBLEM: Appends every row, including duplicates
            self.employee_profiles[emp_name]['skills'].append({
                'skill': row['Skill_Certification_Name'],
                'rating': proficiency_rating,
                ...
            })
```

**What happened:**
1. Database has 2 rows for "Person A" with skill "Morpheus Automation"
2. Loop iterates both rows
3. Appends skill TWICE to Person A's skills list
4. Later, when building `skill_to_resources`, Person A gets added TWICE
5. Result: 44 entries instead of 22 unique people

## The Fix

### New Code (apps/opportunity_chain_db.py:245-294)
```python
def process_employee_data(self):
    self.employee_profiles = {}

    for _, row in self.employees_df.iterrows():
        emp_name = row.get('Resource_Name')
        if emp_name not in self.employee_profiles:
            self.employee_profiles[emp_name] = {
                'skills_dict': {},  # ← Dict prevents duplicates!
                ...
            }

        if pd.notna(row.get('Skill_Certification_Name')):
            skill_name = row['Skill_Certification_Name']

            # Deduplicate: Only add if not already exists
            if skill_name not in self.employee_profiles[emp_name]['skills_dict']:
                self.employee_profiles[emp_name]['skills_dict'][skill_name] = {
                    'skill': skill_name,
                    'rating': proficiency_rating,
                    ...
                }
            else:
                # Keep the higher rating if duplicate exists
                existing_rating = self.employee_profiles[emp_name]['skills_dict'][skill_name]['rating']
                if proficiency_rating > existing_rating:
                    self.employee_profiles[emp_name]['skills_dict'][skill_name]['rating'] = proficiency_rating

    # Convert dict back to list for backward compatibility
    for emp_name, profile in self.employee_profiles.items():
        profile['skills'] = list(profile['skills_dict'].values())
        del profile['skills_dict']
```

**How it works:**
1. Use a **dictionary** (`skills_dict`) keyed by skill name
2. **First occurrence:** Add skill to dict
3. **Duplicate occurrence:** Check if new rating is higher, update if so
4. **Result:** Each person has each skill exactly once, with their best rating
5. **Convert** dict to list at the end for backward compatibility

## Impact Analysis

### Before Fix
```
Search Skills: "Morpheus"

• Cloud Management Platform - Morpheus Automation (44 resources) ❌
• Morpheus Cloud Management Platform (20 resources) ✅
• Ability to articulate... (2 resources) ✅
```

### After Fix
```
Search Skills: "Morpheus"

• Cloud Management Platform - Morpheus Automation (22 resources) ✅
• Morpheus Cloud Management Platform (20 resources) ✅
• Ability to articulate... (2 resources) ✅
```

### Test Results
```
Skill: Cloud Management Platform - Morpheus Automation
  Unique Resources: 22
  Total Entries: 22
  ✅ No duplicates
  ✅ CORRECT! Now showing accurate count

Expected from DB: 22 unique resources
Actual count: 22
✅ Match!
```

## Benefits

### 1. **Accurate Resource Counts**
- Shows true number of people with each skill
- No more inflated numbers from database duplicates

### 2. **Better Decision Making**
- Analysts see correct skill availability
- Resource planning based on accurate data
- Gap analysis shows real shortages, not artifacts

### 3. **Keeps Best Rating**
- If duplicate rows have different ratings, keeps the higher one
- Example: Person has rating 3 in one row, rating 4 in another → Keeps 4
- Shows most optimistic (but realistic) resource capability

### 4. **Backward Compatible**
- Still returns `profile['skills']` as a list
- All existing code continues to work
- No changes needed to consuming code

## Edge Cases Handled

### Case 1: Different Ratings for Same Skill
```
Database:
  Person A, Morpheus Automation, Rating: 3
  Person A, Morpheus Automation, Rating: 4

Result: Keep rating 4 (higher proficiency)
```

### Case 2: Same Rating Multiple Times
```
Database:
  Person B, Cloud Platform, Rating: 3
  Person B, Cloud Platform, Rating: 3

Result: Keep first entry, rating 3
```

### Case 3: Skills with No Duplicates
```
Database:
  Person C, Network Skills, Rating: 2

Result: Works as before, no change
```

## Performance Impact

**Minimal:**
- Dictionary lookup is O(1)
- One-time conversion from dict to list at the end
- Overall processing time unchanged

**Memory:**
- Temporary dict per employee (deleted after conversion)
- Net memory usage same as before

## Database Integrity Note

**Why are there duplicates?**

Possible reasons:
1. **Multiple data sources** merged without deduplication
2. **Skill updates** creating new rows instead of updating existing
3. **Different skill certifications** with similar names
4. **Data migration** artifacts

**Recommendation:** Consider cleaning the source database, but this fix makes the app resilient to such data quality issues.

## Testing

### Manual Testing
1. Search for "Morpheus" in Skills tab
2. Verify counts match database `COUNT(DISTINCT Resource_Name)`
3. Check other skills with known duplicates

### Automated Testing
```python
# Verify no duplicate resources per skill
for skill, resources in platform.skill_to_resources.items():
    unique_count = len(set(r['name'] for r in resources))
    total_count = len(resources)
    assert unique_count == total_count, f"Duplicates found in {skill}"
```

✅ All tests pass

## Code Location

**Modified File:** `apps/opportunity_chain_db.py`

**Changed Section:** Lines 245-294 (`process_employee_data` method)

**Changes:**
1. Use `skills_dict` instead of `skills` list during processing
2. Add deduplication logic with rating comparison
3. Convert dict to list at the end for compatibility

---

**Bug Fixed:** 2025-10-24
**Severity:** Medium (affected resource planning accuracy)
**Impact:** All skill resource counts now accurate
**Status:** ✅ Resolved and Tested
