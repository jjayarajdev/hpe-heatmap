#!/usr/bin/env python3
"""
Create and populate heatmap.db SQLite database from Excel files
Maintains exact structure as read from Excel files
"""

import sqlite3
import pandas as pd
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def create_database():
    """Create the heatmap.db database with all data from Excel files"""

    # Database path
    db_path = Path('data/heatmap.db')

    # Remove existing database if it exists
    if db_path.exists():
        print(f"Removing existing database: {db_path}")
        os.remove(db_path)

    # Create new database connection
    conn = sqlite3.connect(db_path)
    print(f"Created database: {db_path}")

    try:
        # 1. Load Opportunity data
        print("\n1. Loading Opportunity data...")
        opp_df = pd.read_excel('data/Opportunioty and PL sample.xlsx')
        opp_df.to_sql('opportunities', conn, index=False, if_exists='replace')
        print(f"   ✓ Loaded {len(opp_df)} opportunity records")

        # 2. Load Services to Skillsets mapping
        print("\n2. Loading Services to Skillsets mapping...")
        services_df = pd.read_excel('data/Services_to_skillsets Mapping.xlsx', sheet_name='Master v5')
        services_df.to_sql('services_skillsets', conn, index=False, if_exists='replace')
        print(f"   ✓ Loaded {len(services_df)} service-skillset mappings")

        # 3. Load Skillsets to Skills mapping (multiple sheets)
        print("\n3. Loading Skillsets to Skills mapping...")
        excel_file = pd.ExcelFile('data/Skillsets_to_Skills_mapping.xlsx')

        # Store each sheet separately with sheet name as a column
        all_skills_data = []
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            df['sheet_name'] = sheet_name  # Add sheet name as a column
            all_skills_data.append(df)
            print(f"   - Sheet '{sheet_name}': {len(df)} rows")

        # Combine all sheets
        skills_df = pd.concat(all_skills_data, ignore_index=True)
        skills_df.to_sql('skillsets_skills', conn, index=False, if_exists='replace')
        print(f"   ✓ Total loaded: {len(skills_df)} skillset-skill mappings")

        # 4. Load Employee Skills data
        print("\n4. Loading Employee Skills data...")
        emp_df = pd.read_excel('data/DETAILS (28).xlsx', sheet_name='Export')
        emp_df.to_sql('employee_skills', conn, index=False, if_exists='replace')
        print(f"   ✓ Loaded {len(emp_df)} employee skill records")

        # Create indexes for better performance
        print("\n5. Creating database indexes...")
        cursor = conn.cursor()

        # Indexes for opportunities
        cursor.execute("CREATE INDEX idx_opp_id ON opportunities('HPE Opportunity Id')")
        cursor.execute("CREATE INDEX idx_opp_pl ON opportunities('Product Line')")

        # Indexes for services_skillsets
        cursor.execute("CREATE INDEX idx_services_name ON services_skillsets('New Service Name')")
        cursor.execute("CREATE INDEX idx_services_pl ON services_skillsets('FY25 PL')")
        cursor.execute("CREATE INDEX idx_services_skillset ON services_skillsets('Skill Set')")

        # Indexes for skillsets_skills
        cursor.execute("CREATE INDEX idx_skills_fy24 ON skillsets_skills(\"FY'24 Skillset Name\")")
        cursor.execute("CREATE INDEX idx_skills_fy25 ON skillsets_skills(\"FY'25 Skillset Name\")")

        # Indexes for employee_skills
        cursor.execute("CREATE INDEX idx_emp_name ON employee_skills('Resource_Name')")
        cursor.execute("CREATE INDEX idx_emp_skill ON employee_skills('Skill_Certification_Name')")
        cursor.execute("CREATE INDEX idx_emp_skillset ON employee_skills('Skill_Set_Name')")

        print("   ✓ Created performance indexes")

        # Commit changes
        conn.commit()

        # Print database statistics
        print("\n" + "="*60)
        print("DATABASE SUMMARY")
        print("="*60)

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()

        for table_name in tables:
            table = table_name[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"   Table '{table}': {count:,} records")

        print("\n✅ Database created successfully!")
        print(f"   Location: {db_path.absolute()}")
        print(f"   Size: {db_path.stat().st_size / (1024*1024):.2f} MB")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    print("Creating Heatmap SQLite Database")
    print("="*60)
    create_database()