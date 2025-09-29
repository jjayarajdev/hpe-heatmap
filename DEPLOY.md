# Streamlit Cloud Deployment Guide

## Prerequisites
- GitHub repository: https://github.com/jjayarajdev/hpe-heatmap (branch: dev-29Sept)
- Streamlit Cloud account (free at https://share.streamlit.io)

## Deployment Steps

### 1. Sign up for Streamlit Cloud
1. Go to https://share.streamlit.io
2. Sign in with your GitHub account
3. Authorize Streamlit to access your repositories

### 2. Deploy the Application
1. Click "New app" in Streamlit Cloud
2. Fill in the deployment details:
   - **Repository**: jjayarajdev/hpe-heatmap
   - **Branch**: dev-29Sept
   - **Main file path**: apps/opportunity_chain_db.py
   - **App URL**: Choose a custom URL (e.g., hpe-opportunity-intelligence)

### 3. Configure Secrets (Important!)
Since the database file is large (17MB), you'll need to handle it specially:

#### Option A: Use Excel Version for Deployment
Change the main file path to: `apps/opportunity_chain_complete.py`
This version reads Excel files directly (slower but works on Streamlit Cloud)

#### Option B: Use Smaller Sample Database
1. Create a smaller sample database locally
2. Upload it to GitHub (must be under 10MB)
3. Update the database path in the app

### 4. Environment Variables (if needed)
In Streamlit Cloud settings, add any secrets:
```toml
[database]
path = "data/heatmap.db"
```

## Files Required for Deployment
✅ `.streamlit/config.toml` - Theme and settings
✅ `requirements.txt` - Python dependencies
✅ `apps/opportunity_chain_db.py` - Main application
✅ `data/` - Data files (Excel files for Excel version)

## Post-Deployment

### Access Your App
Your app will be available at:
- https://[your-custom-name].streamlit.app

### Monitor Performance
- Check the Streamlit Cloud dashboard for:
  - Resource usage
  - Error logs
  - User analytics

### Limitations on Streamlit Cloud
- **Memory**: 1GB RAM limit
- **Storage**: Files over 50MB may cause issues
- **CPU**: Shared resources
- **Database**: SQLite works but large files may timeout

## Alternative Deployment Options

### For Production Use:
1. **Streamlit Community Cloud** (current, free)
2. **Heroku** (requires Procfile)
3. **AWS EC2** (full control)
4. **Google Cloud Run** (containerized)
5. **Azure App Service** (enterprise)

## Troubleshooting

### If deployment fails:
1. Check file sizes (database must be <50MB)
2. Verify all imports are in requirements.txt
3. Ensure branch name is correct
4. Check logs in Streamlit Cloud dashboard

### For large database files:
Consider using:
- Cloud storage (S3, GCS)
- Remote database (PostgreSQL, MySQL)
- Data sampling for demo version

## Support
- Streamlit Forums: https://discuss.streamlit.io
- Documentation: https://docs.streamlit.io/streamlit-cloud

---
*Last Updated: September 2025*