# GitHub Actions Setup for Bitcoin Predictor

## Quick Start Guide

This repository is configured with GitHub Actions to automatically:
1. **Train models daily** at 2 AM UTC
2. **Update Wikipedia sentiment data**
3. **Commit trained models** back to the repository
4. **Trigger Render deployment** when models are updated

## What Was Created

### 1. GitHub Actions Workflows

#### `.github/workflows/train-models.yml`
- **Purpose**: Automatically trains models and updates data
- **Schedule**: Daily at 2 AM UTC (configurable)
- **What it does**:
  - Installs Python and dependencies
  - Runs `update_date.py` to train models
  - Commits updated models to repository
  - Uploads models as artifacts (backup)

#### `.github/workflows/render-deploy.yml`
- **Purpose**: Notifies when deployment should happen
- **Trigger**: When model files or app code changes
- **What it does**:
  - Detects changes to models/app
  - Logs deployment information
  - (Optional) Can trigger Render webhook

### 2. Documentation

- **DEPLOYMENT.md**: Complete deployment guide
- **GITHUB_ACTIONS_SETUP.md**: This file

## How to Use

### Initial Setup

1. **Push these files to GitHub**:
   ```bash
   git add .github/
   git add DEPLOYMENT.md
   git add GITHUB_ACTIONS_SETUP.md
   git add .gitignore
   git commit -m "Add GitHub Actions workflows for automated model training"
   git push
   ```

2. **Verify workflows are active**:
   - Go to your GitHub repo
   - Click "Actions" tab
   - You should see the workflows listed

3. **Test the workflow**:
   - Go to Actions → "Train Models and Update Data"
   - Click "Run workflow" → "Run workflow"
   - Watch it train models and commit results

### Automatic Daily Training

Once set up, the workflow will:
- Run every day at 2 AM UTC
- Train new models with latest data
- Commit models to repository
- Render will auto-deploy when it detects changes

### Manual Training

To manually trigger training:
1. Go to GitHub Actions tab
2. Select "Train Models and Update Data"
3. Click "Run workflow" → "Run workflow"

## Workflow Details

### Training Workflow Steps

1. **Checkout Code**: Gets latest code from repository
2. **Setup Python**: Installs Python 3.11
3. **Install Dependencies**: Installs packages from `requirements.txt`
4. **Train Models**: Runs `update_date.py`
5. **Verify Models**: Checks that model files were created
6. **Commit Changes**: Commits models and data files
7. **Upload Artifacts**: Saves models as backup (7 days retention)

### What Gets Committed

The workflow automatically commits:
- `models/saved_models/bitcoin_model.pkl` - Trained model
- `models/saved_models/feature_info.json` - Model metadata
- `wikipedia_edits.csv` - Updated sentiment data

### Commit Message Format

```
🤖 Auto-update: Retrain models and update data [skip ci]
```

The `[skip ci]` tag prevents infinite loops (workflow won't trigger on its own commits).

## Configuration Options

### Change Training Schedule

Edit `.github/workflows/train-models.yml`:

```yaml
schedule:
  - cron: '0 2 * * *'  # Change time here (UTC)
```

Cron format: `minute hour day month day-of-week`

Examples:
- `'0 2 * * *'` - Daily at 2 AM UTC
- `'0 0 * * 0'` - Weekly on Sunday at midnight
- `'0 */6 * * *'` - Every 6 hours

### Disable Auto-Commit

If you don't want models committed automatically:

1. Comment out the "Commit and push changes" step in `train-models.yml`
2. Models will still be uploaded as artifacts
3. You can manually download and commit them

### Add Render Webhook

To automatically trigger Render deployments:

1. Get webhook URL from Render:
   - Render Dashboard → Your Service → Settings → Build & Deploy
   - Copy "Build Hook URL"

2. Add as GitHub Secret:
   - Repo → Settings → Secrets and variables → Actions
   - New repository secret: `RENDER_WEBHOOK_URL`
   - Paste webhook URL

3. Uncomment webhook step in `render-deploy.yml`:
   ```yaml
   - name: Trigger Render webhook
     if: github.event_name == 'push'
     run: |
       curl -X POST "${{ secrets.RENDER_WEBHOOK_URL }}"
   ```

## Troubleshooting

### Workflow Not Running

**Problem**: Workflow doesn't run on schedule

**Solutions**:
1. Check workflow file is in `.github/workflows/` directory
2. Verify YAML syntax is correct (no errors in Actions tab)
3. Ensure repository has Actions enabled (Settings → Actions)
4. Check if you're on a free plan (scheduled workflows work on free tier)

### Training Fails

**Problem**: Workflow fails during model training

**Solutions**:
1. Check workflow logs for specific error
2. Verify `requirements.txt` has all dependencies
3. Check API rate limits (Wikipedia, Yahoo Finance)
4. Increase timeout if training takes too long:
   ```yaml
   timeout-minutes: 120  # Increase from 60
   ```

### Models Not Committed

**Problem**: Models trained but not committed

**Solutions**:
1. Check if models directory exists: `models/saved_models/`
2. Verify files aren't in `.gitignore`
3. Check workflow logs for git errors
4. Ensure GitHub token has write permissions

### Render Not Deploying

**Problem**: Models updated but Render doesn't deploy

**Solutions**:
1. Check Render is connected to GitHub
2. Verify auto-deploy is enabled in Render
3. Manually trigger deployment in Render dashboard
4. Check if Render webhook is configured (optional)

## Monitoring

### View Workflow Runs

1. Go to GitHub repo → Actions tab
2. Click on workflow name
3. View run history and logs

### Check Model Updates

1. Go to repository
2. Check `models/saved_models/` directory
3. View file history to see update times

### Monitor Training Time

- Typical training: 20-30 minutes
- Check workflow duration in Actions tab
- Adjust timeout if needed

## Best Practices

1. **Monitor First Few Runs**: Check that workflows complete successfully
2. **Review Commits**: Verify models are being committed correctly
3. **Check Logs**: Regularly review workflow logs for errors
4. **Backup Models**: Artifacts are kept for 7 days as backup
5. **Test Locally**: Test `update_date.py` locally before relying on automation

## Cost Considerations

### GitHub Actions

- **Free Tier**: 2,000 minutes/month
- **Training Time**: ~20-30 minutes per run
- **Daily Runs**: ~600-900 minutes/month
- **Cost**: $0 (within free tier)

### Render

- **Free Tier**: Available with limitations
- **Auto-Deploy**: Free
- **Cost**: $0 (within free tier)

## Security

1. **GitHub Token**: Automatically provided, has repo access
2. **Secrets**: Use GitHub Secrets for sensitive data
3. **API Keys**: Don't hardcode, use environment variables
4. **Model Files**: Committed to repo (public repos = public models)

## Next Steps

1. ✅ Push workflows to GitHub
2. ✅ Test workflow manually
3. ✅ Verify models are committed
4. ✅ Set up Render deployment
5. ✅ Monitor first scheduled run
6. ✅ Adjust schedule if needed

## Support

For issues:
- Check workflow logs in GitHub Actions
- Review error messages
- Test `update_date.py` locally
- Check Render deployment logs

---

**Your automated training pipeline is ready! 🚀**

