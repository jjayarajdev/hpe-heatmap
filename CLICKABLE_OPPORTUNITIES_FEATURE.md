# Clickable Opportunities Feature

## Overview
Added clickable opportunity links in the Overview tab that automatically navigate to the Chain Analysis tab with the selected opportunity pre-loaded.

## User Experience

### Before
1. View opportunities in Overview tab (static table)
2. Manually switch to Chain Analysis tab
3. Search for the opportunity ID in the dropdown
4. Select the opportunity

**Problem:** 4 steps, time-consuming, prone to errors

### After
1. **Click** any opportunity in Overview tab
2. **Done!** - Automatically opens Chain Analysis with that opportunity selected

**Benefit:** 1 click instead of 4 steps! ⚡

## Visual Design

### Overview Tab - Clickable Opportunity Cards

```
📊 Platform Overview

📈 Opportunity Distribution
💡 Click any opportunity to view its complete chain

┌────────────────────────────────────────────────────────────────┐
│ 🔗 OPE-XX1Y14144                │ 4J - Education...    │ $21.45M │
│ Serverausschreibung 2025...                                    │
├────────────────────────────────────────────────────────────────┤
│ 🔗 OPE-XX141X2Y16               │ 96 - Industry...     │ $18.32M │
│ Data Center Modernization...                                   │
├────────────────────────────────────────────────────────────────┤
│ 🔗 OPE-XX1424604X               │ 1Z - Network         │ $15.67M │
│ Network Infrastructure...                                      │
└────────────────────────────────────────────────────────────────┘
          ↑
    Click here to jump to Chain Analysis!
```

**Features:**
- 🔗 Clickable opportunity ID button
- Opportunity name preview
- Product Line code and description
- TCV USD value
- Hover tooltip with full opportunity name

## How It Works

### 1. Click Flow

```
User Action:
┌─────────────────────────────────────┐
│ Overview Tab                        │
│                                     │
│ [🔗 OPE-XX1Y14144] ← CLICK HERE    │
│ Serverausschreibung 2025...         │
└─────────────────────────────────────┘
                ↓
        (Button clicked)
                ↓
┌─────────────────────────────────────┐
│ Session State Updated:              │
│ • selected_opportunity_id =         │
│   "OPE-XX1Y14144"                   │
│ • active_tab = 1 (Chain Analysis)   │
└─────────────────────────────────────┘
                ↓
        (App reruns)
                ↓
┌─────────────────────────────────────┐
│ Chain Analysis Tab                  │
│                                     │
│ Opportunity: [OPE-XX1Y14144...]     │
│             ↑                       │
│    Pre-selected automatically!      │
│                                     │
│ [Chain visualization appears]       │
└─────────────────────────────────────┘
```

### 2. Technical Implementation

#### Session State Variables
```python
st.session_state.active_tab = 1  # Which tab to show
st.session_state.selected_opportunity_id = "OPE-XXX"  # Which opportunity
```

#### Overview Tab - Clickable Buttons
```python
for idx, row in top_opps.iterrows():
    opp_id = row['HPE Opportunity Id']

    if st.button(
        f"🔗 {opp_id}",
        key=f"opp_overview_{opp_id}",
        help=f"Click to analyze: {opp_name[:50]}"
    ):
        # Store selected opportunity and switch tab
        st.session_state.selected_opportunity_id = opp_id
        st.session_state.active_tab = 1
        st.rerun()
```

#### Chain Analysis Tab - Pre-selection
```python
# Determine default index based on pre-selected opportunity
default_index = 0
if 'selected_opportunity_id' in st.session_state:
    # Find the opportunity in the dropdown options
    for idx, (label, opp_id) in enumerate(opp_options.items()):
        if opp_id == st.session_state.selected_opportunity_id:
            default_index = idx
            break
    # Clear after using it (one-time action)
    del st.session_state.selected_opportunity_id

# Selectbox with pre-selected opportunity
selected = st.selectbox(
    "Choose an opportunity to explore",
    options=list(opp_options.keys()),
    index=default_index  # ← Pre-selected!
)
```

#### Tab Navigation - Radio-Based Control
```python
# Use radio buttons instead of st.tabs for programmatic control
selected_tab = st.radio(
    "Navigation",
    options=range(len(tab_names)),
    format_func=lambda x: tab_names[x],
    index=st.session_state.active_tab,  # ← Respects session state
    horizontal=True
)

if selected_tab == 0:
    self.render_overview_tab()
elif selected_tab == 1:
    self.render_chain_analysis_tab()
# ...
```

**Why Radio Instead of st.tabs?**
- `st.tabs()` doesn't support programmatic tab switching
- Radio buttons styled horizontally look similar to tabs
- Full control over which tab is active via session state

## User Workflow Example

### Scenario: "I want to analyze the largest opportunity"

**Old Way (4 steps):**
1. Look at Overview → See "OPE-XX1Y14144" is $21.45M
2. Remember or copy the opportunity ID
3. Click Chain Analysis tab
4. Search dropdown for "OPE-XX1Y14144" and select

**New Way (1 click):**
1. Click 🔗 OPE-XX1Y14144 button → **Done!**

**Time saved:** ~30 seconds per lookup

**Annual impact:** If analysts look up 10 opportunities per day:
- 10 opps × 30 sec × 250 work days = **1,250 minutes saved/year** = **20.8 hours**

## Additional Benefits

### 1. **Reduced Errors**
- No more mistyping opportunity IDs
- No more selecting wrong opportunity from dropdown

### 2. **Faster Insights**
- Immediate drill-down from summary to detail
- Natural exploration workflow

### 3. **Better UX**
- Intuitive click-to-navigate pattern
- Clear visual affordance (🔗 icon + button styling)
- Hover tooltip shows full opportunity name

### 4. **Consistent Navigation**
- Same pattern can be extended to other sections
- Example: Click PL in overview → See PL-filtered chain

## UI Components

### Opportunity Card Layout
```
┌──────────────────────────────────────────────────┐
│ [🔗 OPE-XXX (button)]  │ PL Code - Desc │ Value │
│ Opportunity Name...    │                │       │
└──────────────────────────────────────────────────┘
  ↑                         ↑                ↑
Clickable ID           Product Line      TCV USD
(jumps to analysis)
```

### Styling
- **Button:** Full-width, styled like a link with 🔗 icon
- **Layout:** 3 columns (60% ID, 25% PL, 15% Value)
- **Hover:** Tooltip shows full opportunity name
- **Caption:** Truncated opportunity name below button
- **Divider:** Visual separation between opportunities

## Testing

### Test Scenarios

✅ **Scenario 1: Click opportunity from Overview**
1. Start on Overview tab
2. Click "🔗 OPE-XX1Y14144"
3. Verify switches to Chain Analysis tab
4. Verify opportunity is pre-selected in dropdown
5. Verify chain displays correctly

✅ **Scenario 2: Click multiple opportunities**
1. Click opportunity A → See chain for A
2. Return to Overview
3. Click opportunity B → See chain for B
4. Verify no interference between selections

✅ **Scenario 3: Opportunity not in top 50**
1. Click opportunity ranked #5 in Overview
2. If not in top 50 of Chain Analysis dropdown
3. Verify gracefully defaults to first option
4. User can still manually select

## Code Location

**Modified File:** `apps/opportunity_chain_db.py`

**Key Changes:**
1. Lines 719-746: Tab navigation with session state control
2. Lines 755-794: Clickable opportunity cards in Overview
3. Lines 946-962: Pre-selection logic in Chain Analysis

**Added Session State:**
- `st.session_state.active_tab` - Controls which tab is shown
- `st.session_state.selected_opportunity_id` - Pre-selects opportunity

## Future Enhancements

### Possible Extensions
1. **Breadcrumbs:** Show navigation path (Overview → Chain Analysis → OPP-XXX)
2. **Back Button:** Quick return to Overview with scroll position preserved
3. **Deep Linking:** URL parameters for direct opportunity access
4. **Keyboard Shortcuts:** Arrow keys to navigate opportunities
5. **Recent History:** Track last 5 viewed opportunities

### Similar Patterns
- Click PL in overview → See all opportunities for that PL
- Click resource → See all opportunities they match
- Click service → See all opportunities requiring it

---

**Feature Added:** 2025-10-24
**Status:** ✅ Complete and Tested
**Impact:** Saves ~20 hours/year per analyst
**User Benefit:** 1-click navigation to opportunity details
