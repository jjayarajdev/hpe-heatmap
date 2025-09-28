# Product Line Mapping Table - Direct Matches Only

## Mapping Policy: Direct Code Matches Only

This document shows ONLY the direct mappings where the opportunity PL code appears in the service PL code.
- **Source**: `HPE deals FY25.xlsx` → Column: `Product Line`
- **Target**: `Services_to_skillsets Mapping.xlsx` → Columns: `FY25 PL` and `FY25 PL Name`
- **Rule**: Only include mappings where the opportunity code matches the service code

---

## Direct Match Mapping Table

| Opportunity PL Code | Opportunity PL Description | → | Service FY25 PL | Service FY25 PL Name | Match Type |
|---------------------|---------------------------|---|-----------------|---------------------|------------|
| **60** | Cloud-Native Pltfms | → | **60 (IJ)** | Cloud-Native Pltfms | ✅ Direct |
| **1Z** | Network | → | **1Z (PN)** | Network | ✅ Direct |
| **5V** | Hybrid Workplace | → | **5V (II)** | Hybrid Workplace | ✅ Direct |
| **4J** | Education Services | → | **4J (SX)** | Education | ✅ Direct |
| **G4** | Private Platforms | → | **G4 (PK)** | Private Platforms | ✅ Direct |
| **PD** | HPE POD Modular DC | → | **PD (C8)** | HPE POD Modular DC | ✅ Direct |

### Additional Direct Matches (these codes exist in services but NOT in current opportunity data):
| Code | → | Service FY25 PL | Service FY25 PL Name | Match Type | Note |
|------|---|-----------------|---------------------|------------|------|
| **EA** | → | **EA (IL)** | DC&Sustainable Cons | ✅ Direct | Not in opp data |
| **XJ** | → | **XJ (XK)** | App Modernization | ✅ Direct | Not in opp data |
| **XF** | → | **XF (K0)** | Security Data & AI | ✅ Direct | Not in opp data |
| **GD** | → | **GD (JO)** | Security IT & Hybrid | ✅ Direct | Not in opp data |
| **GU** | → | **GU (JS)** | Security IE | ✅ Direct | Not in opp data |
| **SP** | → | **SP (XM)** | Data Solns & Pltfms | ✅ Direct | Not in opp data |
| **6C** | → | **6C (IK)** | Data Storage | ✅ Direct | Not in opp data |
| **JK** | → | **JK (JL)** | A&PS NonStop | ✅ Direct | Not in opp data |
| **G5** | → | **G5 (PM)** | AI Solns & Pltfms | ✅ Direct | Not in opp data |

---

## Excluded Non-Direct Mappings

These mappings are EXCLUDED because the codes don't match directly:

| Opportunity PL Code | Opportunity PL Description | Service PL (from table) | Reason for Exclusion |
|---------------------|---------------------------|------------------------|---------------------|
| **96** | Industry Standard Servers Support | G4 (PK) | ❌ 96 ≠ G4 |
| **BJ** | GL_HPC Services | G5 (PM) | ❌ BJ ≠ G5 (also G5 not in opp data) |
| **H3** | HPC Services | N/A | ❌ No mapping |
| **KH** | Advisory Services | XF (K0) | ❌ KH ≠ XF |
| **SQ** | Management of Change | GD (JO) | ❌ SQ ≠ GD |
| **PN** | GL_Network | GU (JS) | ❌ PN ≠ GU |
| **SX** | GL_EDU | SP (XM) | ❌ SX ≠ SP |
| **II** | GL_Hybrid Workplace | 6C (IK) | ❌ II ≠ 6C |
| **UM** | GL_A&PS 3PP | 1Z (PN) | ❌ UM ≠ 1Z |
| **NO** | Cloud Technology Partners | JK (JL) | ❌ NO ≠ JK |
| **V7** | GL_MOC | PD (C8) | ❌ V7 ≠ PD |

---

## Revenue Impact of Direct Matches vs Excluded

### Opportunities with Direct Matches:
| Opportunity PL | Revenue | % of Total | Has Direct Match |
|---------------|---------|------------|-----------------|
| **60** | ~$5M | ~1% | ✅ Yes |
| **1Z** | ~$3M | <1% | ✅ Yes |
| **5V** | ~$2M | <1% | ✅ Yes |
| **4J** | ~$10M | ~2% | ✅ Yes |
| **G4** | $6.68M | 1.3% | ✅ Yes |
| **PD** | ~$4M | <1% | ✅ Yes |

### Opportunities WITHOUT Direct Matches (Excluded):
| Opportunity PL | Revenue | % of Total | Has Direct Match |
|---------------|---------|------------|-----------------|
| **96** | $369.45M | 73.5% | ❌ No |
| **31** | $20.59M | 4.1% | ❌ No (not in table) |
| **86** | $17.10M | 3.4% | ❌ No (not in table) |
| **3P** | $25.65M | 5.1% | ❌ No (not in table) |
| **BJ** | ~$1M | <1% | ❌ No |
| **KH** | ~$2M | <1% | ❌ No |

**Important Note**: By using only direct matches, we exclude approximately 85-90% of the opportunity revenue from the analysis, including the largest PL (96) which represents 73.5% of total revenue.

---

## Implementation in Code

```python
def create_pl_mapping(self):
    """Create mapping between opportunity PLs and service PLs"""
    # ONLY use direct matches where opportunity PL code appears in service PL code
    self.pl_mapping = {
        # Direct code matches that exist in opportunity data
        '60': ['60 (IJ)'],    # Cloud-Native Pltfms
        '1Z': ['1Z (PN)'],    # Network
        '5V': ['5V (II)'],    # Hybrid Workplace
        '4J': ['4J (SX)'],    # Education Services
        'G4': ['G4 (PK)'],    # Private Platforms
        'PD': ['PD (C8)'],    # HPE POD Modular DC

        # Note: The following codes exist in services but NOT in current opportunity data
        # They are included here in case they appear in future opportunity data
        # 'EA': ['EA (IL)'],    # DC&Sustainable Cons - not in opp data
        # 'XJ': ['XJ (XK)'],    # App Modernization - not in opp data
        # 'XF': ['XF (K0)'],    # Security Data & AI - not in opp data
        # 'GD': ['GD (JO)'],    # Security IT & Hybrid - not in opp data
        # 'GU': ['GU (JS)'],    # Security IE - not in opp data
        # 'SP': ['SP (XM)'],    # Data Solns & Pltfms - not in opp data
        # '6C': ['6C (IK)'],    # Data Storage - not in opp data
        # 'JK': ['JK (JL)'],    # A&PS NonStop - not in opp data
        # 'G5': ['G5 (PM)'],    # AI Solns & Pltfms - not in opp data
    }
```

---

## Summary

- **Direct Matches**: 6 primary PLs with matching codes (60, 1Z, 5V, 4J, G4, PD)
- **Excluded**: 11+ PLs with non-matching codes
- **Revenue Coverage**: Only ~10-15% of opportunity revenue has direct matches
- **Largest Exclusion**: PL 96 (73.5% of revenue) has no direct match
- **Important Note**: G5 exists in service data but NOT in opportunity data

---

*Last Updated: 2025-09-26*
*Policy: Direct code matches only*