# Streamlit Cloud Deployment Guide

## Prerequisites
1. GitHub account
2. Your project pushed to a GitHub repository

## Step-by-Step Deployment

### 1. Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/yourusername/your-repo-name.git
git push -u origin main
```

### 2. Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository
5. Set the main file path to: `app.py`
6. Click "Deploy!"

### 3. Environment Variables (Optional)
If you need the OpenWeatherMap API key:
1. In your Streamlit Cloud app settings
2. Go to "Secrets" section
3. Add:
```toml
OPENWEATHERMAP_API_KEY = "your-api-key-here"
```

### 4. App Configuration
The app is already configured with:
- ✅ Streamlit config file (`.streamlit/config.toml`)
- ✅ Requirements file (`requirements.txt`)
- ✅ Proper file structure
- ✅ Git ignore file (`.gitignore`)

## Troubleshooting
- If deployment fails, check the logs in Streamlit Cloud
- Ensure all data files are in the correct paths
- Verify all dependencies are in `requirements.txt`

## Your app will be available at:
`https://your-app-name-your-username.streamlit.app` 