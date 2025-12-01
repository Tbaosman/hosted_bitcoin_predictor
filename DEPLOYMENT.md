# Deployment Guide for Bitcoin Predictor

This guide explains how to deploy the Bitcoin Predictor application to Render using GitHub Actions for automated model training.

## Overview

The deployment setup consists of:
1. **GitHub Actions**: Automatically trains models and updates data
2. **Render**: Hosts the Flask web application
3. **Automated Workflow**: Models are trained → Committed to repo → Render auto-deploys

## Prerequisites

- GitHub repository with your code
- Render account (free tier works)
- Python 3.11+ support

## Setup Instructions

### Step 1: Configure GitHub Actions

The workflow files are already created:
- `.github/workflows/train-models.yml` - Trains models daily
- `.github/workflows/render-deploy.yml` - Triggers Render deployment

**No additional configuration needed** - the workflows will work out of the box!

### Step 2: Deploy to Render

1. **Go to Render Dashboard**: https://dashboard.render.com

2. **Create a New Web Service**:
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the repository containing this code

3. **Configure the Service**:
   - **Name**: `bitcoin-predictor` (or your preferred name)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app` (already in Procfile)
   - **Plan**: Free tier is sufficient

4. **Environment Variables** (if needed):
   - `PORT`: Automatically set by Render (don't override)
   - Add any API keys or secrets here if you add them later

5. **Click "Create Web Service"**

### Step 3: Verify Deployment

1. **Check Build Logs**: 
   - Render will install dependencies and start the app
   - Look for: "✅ Model loaded successfully" in logs

2. **Access Your App**:
   - Render provides a URL like: `https://bitcoin-predictor.onrender.com`
   - Visit the URL to see your app

3. **Test the API**:
   - Visit: `https://your-app.onrender.com/predict`
   - Should return prediction JSON

## How It Works

### Daily Model Training (GitHub Actions)

1. **Schedule**: Runs daily at 2 AM UTC (configurable in `train-models.yml`)
2. **Process**:
   - Checks out code
   - Installs dependencies
   - Runs `update_date.py` to:
     - Fetch Wikipedia sentiment data
     - Train Bitcoin prediction model
     - Save models to `models/saved_models/`
   - Commits updated models to repository
3. **Result**: Latest models are always in the repo

### Render Auto-Deployment

1. **Trigger**: When code is pushed to `main` branch
2. **Process**:
   - Render detects changes
   - Builds the application
   - Deploys with latest models
3. **Result**: App always has latest trained models

## Manual Operations

### Manually Trigger Model Training

1. Go to GitHub Actions tab
2. Select "Train Models and Update Data" workflow
3. Click "Run workflow" → "Run workflow"

### Manually Trigger Render Deployment

1. Go to Render dashboard
2. Select your service
3. Click "Manual Deploy" → "Deploy latest commit"

### Train Models Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run training script
python update_date.py

# Models will be saved to:
# - models/saved_models/bitcoin_model.pkl
# - models/saved_models/feature_info.json
# - wikipedia_edits.csv
```

## Troubleshooting

### Models Not Loading on Render

**Problem**: App shows "Model not loaded" error

**Solutions**:
1. Check that `models/saved_models/` directory exists in repo
2. Verify model files are committed (not in `.gitignore`)
3. Check Render build logs for errors
4. Ensure GitHub Actions workflow completed successfully

### GitHub Actions Failing

**Problem**: Workflow fails during training

**Solutions**:
1. Check workflow logs for specific error
2. Verify all dependencies in `requirements.txt`
3. Increase timeout if training takes too long
4. Check API rate limits (Wikipedia, Yahoo Finance)

### Render Build Failing

**Problem**: Render deployment fails

**Solutions**:
1. Check build logs for Python version issues
2. Verify `requirements.txt` has all dependencies
3. Ensure `Procfile` exists and is correct
4. Check for memory issues (upgrade plan if needed)

### Models Not Updating

**Problem**: Models are old despite training

**Solutions**:
1. Check GitHub Actions ran successfully
2. Verify models were committed to repo
3. Trigger manual Render deployment
4. Check file timestamps in `models/saved_models/`

## Customization

### Change Training Schedule

Edit `.github/workflows/train-models.yml`:
```yaml
schedule:
  - cron: '0 2 * * *'  # Change to your preferred time (UTC)
```

### Add Environment Variables

In Render dashboard:
1. Go to your service → Environment
2. Add variables like:
   - `API_KEY`: Your API key
   - `DEBUG`: `false` for production

### Use Render Webhook for Auto-Deploy

1. Get webhook URL from Render (Settings → Build & Deploy)
2. Add as GitHub secret: `RENDER_WEBHOOK_URL`
3. Uncomment webhook step in `render-deploy.yml`

## File Structure

```
.
├── .github/
│   └── workflows/
│       ├── train-models.yml      # Daily model training
│       └── render-deploy.yml    # Deployment trigger
├── models/
│   ├── saved_models/            # Trained models (committed to repo)
│   │   ├── bitcoin_model.pkl
│   │   └── feature_info.json
│   ├── model_manager.py
│   ├── price_predictor.py
│   └── sentiment_analyzer.py
├── app.py                       # Flask application
├── update_date.py              # Training script
├── requirements.txt             # Python dependencies
├── Procfile                     # Render start command
└── wikipedia_edits.csv          # Sentiment data (committed)
```

## Monitoring

### Check Model Training Status

- GitHub Actions tab → "Train Models and Update Data"
- View logs to see training progress

### Check App Status

- Render dashboard → Your service
- View logs and metrics

### Monitor Model Performance

- Visit: `https://your-app.onrender.com/api/model_performance`
- Check prediction accuracy and metrics

## Cost Considerations

- **GitHub Actions**: Free tier includes 2,000 minutes/month
- **Render**: Free tier available (with limitations)
- **Training Time**: ~20-30 minutes per run
- **Monthly Cost**: ~$0 (within free tiers)

## Security Notes

1. **Never commit API keys** - Use environment variables
2. **Review commits** - GitHub Actions commits are automated
3. **Monitor usage** - Watch for unexpected API calls
4. **Rate limits** - Be aware of Wikipedia/Yahoo Finance limits

## Support

For issues:
1. Check GitHub Actions logs
2. Check Render build/deploy logs
3. Review application logs in Render
4. Check model files exist in repository

## Next Steps

1. ✅ Deploy to Render
2. ✅ Verify GitHub Actions runs successfully
3. ✅ Test predictions on live app
4. ✅ Monitor for a few days
5. ✅ Adjust schedule if needed

---

**Happy Deploying! 🚀**

