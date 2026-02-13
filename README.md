# AgriTwin-GH 🌱

Digital Twin System for Greenhouse Agriculture

## 📖 Documentation Website

**Live Documentation:** https://arjun-christopher.github.io/AgriTwin-GH/

> Anyone with the link can view the documentation (repository stays private)

## 🚀 Quick Setup

If the documentation site is showing 404, follow these steps:

### Step 1: Enable GitHub Pages

1. Go to repository **Settings** → **Pages**
2. Under **Source**, select: **Deploy from a branch**
3. Under **Branch**, select: **gh-pages** and **/ (root)**
4. Click **Save**

### Step 2: Enable GitHub Actions Permissions

1. Go to repository **Settings** → **Actions** → **General**
2. Under **Workflow permissions**, select: **Read and write permissions**
3. Check: ✅ **Allow GitHub Actions to create and approve pull requests**
4. Click **Save**

### Step 3: Trigger Deployment

Option A: Make any commit and push
```bash
git commit --allow-empty -m "Trigger docs deployment"
git push origin main
```

Option B: Manually trigger workflow
- Go to **Actions** tab
- Click "Deploy Documentation"  
- Click "Run workflow"

### Step 4: Monitor Deployment

- Go to **Actions** tab
- Watch the "Deploy Documentation" workflow run
- Once complete (green checkmark), your site will be live at:
  - https://arjun-christopher.github.io/AgriTwin-GH/

## ✨ Features

- **Synthetic Data Generation** - Create realistic greenhouse sensor data
- **Digital Twin Simulation** - Real-time greenhouse environment modeling  
- **Disease Risk Assessment** - Monitor and predict disease risks
- **Growth Stage Tracking** - Track crop development stages
- **Control Policies** - MPC-like actions and automated alerts
- **Dashboard Visualizations** - Compare scenarios and monitor performance

## 📚 Documentation

All markdown files in this repository are automatically published to the documentation website. Simply add or edit `.md` files and push - they'll appear on the site automatically!

---

*For detailed deployment instructions, see [DOCS_DEPLOYMENT.md](DOCS_DEPLOYMENT.md)*
