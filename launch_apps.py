#!/usr/bin/env python3
"""
HPE Opportunity Intelligence Platform - Application Launcher
Choose between database and Excel versions of the opportunity chain application
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# ANSI color codes
GREEN = '\033[92m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
RED = '\033[91m'
CYAN = '\033[96m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_header():
    """Print application header"""
    print(f"""
{CYAN}{BOLD}
╔══════════════════════════════════════════════════════════════════╗
║          HPE OPPORTUNITY INTELLIGENCE PLATFORM                    ║
║                  Application Launcher                              ║
╚══════════════════════════════════════════════════════════════════╝
{RESET}
    """)

def print_app_menu():
    """Display available applications"""
    apps = {
        "1": {
            "name": "🗄️ Opportunity Chain (DATABASE - FASTEST)",
            "file": "apps/opportunity_chain_db.py",
            "description": "Database-powered version with superior performance",
            "features": [
                "SQLite database backend for instant loading (34.6x faster)",
                "Complete opportunity to skills chain analysis",
                "4-tab interface: Overview, Chain Analysis, Resources, Search",
                "Optimized for large-scale data operations"
            ]
        },
        "2": {
            "name": "📊 Opportunity Chain (Excel Version)",
            "file": "apps/opportunity_chain_complete.py",
            "description": "Direct Excel processing with complete chain functionality",
            "features": [
                "Complete chain: Opportunity → PL → Services → Skillsets → Skills → Resources",
                "Same features as database version",
                "Interactive resource drill-down",
                "Skills gap analysis and matching"
            ]
        }
    }

    print(f"{GREEN}{BOLD}Available Applications:{RESET}\n")

    for key, app in apps.items():
        print(f"{BLUE}{BOLD}[{key}] {app['name']}{RESET}")
        print(f"    {app['description']}")
        print(f"{YELLOW}    Features:{RESET}")
        for feature in app['features']:
            print(f"      • {feature}")
        print()

    return apps

def launch_streamlit_app(app_path):
    """Launch a Streamlit application"""
    print(f"\n{GREEN}Launching application...{RESET}")
    print(f"{CYAN}Access the application at: {BOLD}http://localhost:8501{RESET}")
    print(f"{YELLOW}Press Ctrl+C to stop the application{RESET}\n")

    try:
        # Launch streamlit
        process = subprocess.Popen(
            ['streamlit', 'run', app_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        # Print output
        for line in process.stdout:
            if 'Local URL' in line or 'Network URL' in line:
                print(f"{CYAN}{line.strip()}{RESET}")

        process.wait()

    except KeyboardInterrupt:
        print(f"\n{YELLOW}Stopping application...{RESET}")
        process.terminate()
        time.sleep(1)
        if process.poll() is None:
            process.kill()
        print(f"{GREEN}Application stopped successfully{RESET}")
    except Exception as e:
        print(f"{RED}Error launching application: {e}{RESET}")

def main():
    """Main launcher function"""
    print_header()

    # Check if data directory exists
    data_dir = Path("data")
    if not data_dir.exists():
        print(f"{RED}Warning: 'data' directory not found!{RESET}")
        print(f"{YELLOW}Please ensure all Excel data files are in the 'data' folder{RESET}\n")

    # Check if database exists
    db_path = Path("data/heatmap.db")
    if not db_path.exists():
        print(f"{YELLOW}Note: Database not found. Run 'python scripts/create_heatmap_db.py' to create it.{RESET}\n")

    # Display menu
    apps = print_app_menu()

    # Get user choice
    print(f"{GREEN}Enter your choice (1-2) or 'q' to quit: {RESET}", end='')
    choice = input().strip()

    if choice.lower() == 'q':
        print(f"{CYAN}Goodbye!{RESET}")
        sys.exit(0)

    if choice not in apps:
        print(f"{RED}Invalid choice! Please select 1 or 2{RESET}")
        sys.exit(1)

    selected_app = apps[choice]
    app_path = selected_app["file"]

    # Check if file exists
    if not Path(app_path).exists():
        print(f"{RED}Error: Application file not found: {app_path}{RESET}")
        sys.exit(1)

    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{GREEN}Selected: {selected_app['name']}{RESET}")
    print(f"{CYAN}{'='*60}{RESET}")

    # Launch the app
    launch_streamlit_app(app_path)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Application launcher interrupted{RESET}")
    except Exception as e:
        print(f"{RED}Unexpected error: {e}{RESET}")
        sys.exit(1)