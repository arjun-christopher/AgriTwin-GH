# 📚 MkDocs Material Documentation Site Setup

This repository includes a **MkDocs Material** documentation site that automatically publishes all markdown files as a beautiful, searchable website - **publicly accessible even from this private repository**.

## 🔓 Accessing All Markdown Files

**Important:** All markdown files in the repository are converted to HTML and publicly accessible on the website, even if they don't appear in the navigation menu!

### How to Access Any Markdown File:

Simply convert the file path to a URL:

**Pattern:** `https://arjun-christopher.github.io/AgriTwin-GH/` + `path/to/file` (without .md extension)

**Examples:**

| Repository File | Public URL |
|----------------|------------|
| `docs/index.md` | https://arjun-christopher.github.io/AgriTwin-GH/docs/ |
| `feature_demos/FEATURE_DEMOS_GUIDE.md` | https://arjun-christopher.github.io/AgriTwin-GH/feature_demos/FEATURE_DEMOS_GUIDE/ |
| `README.md` | https://arjun-christopher.github.io/AgriTwin-GH/README/ |
| `docs/DOCS_DEPLOYMENT.md` | https://arjun-christopher.github.io/AgriTwin-GH/docs/DOCS_DEPLOYMENT/ |

### Benefits:

- ✅ **Share specific documentation links** - Send colleagues direct links to any markdown file
- ✅ **Repository stays private** - Only the HTML documentation is public, not your source code
- ✅ **No GitHub account required** - Anyone with the link can read the documentation
- ✅ **Automatic updates** - New markdown files are automatically accessible when you push changes
- ✅ **Full search** - All content is searchable using the built-in search feature

---

## 🌐 Deployment Options

### Option 1: GitHub Pages (Recommended)

The easiest option - your site will be publicly accessible at `https://arjun-christopher.github.io/AgriTwin-GH/`

#### Setup Steps:

1. **Enable GitHub Pages:**
   - Go to your repository Settings → Pages
   - Under "Source", select **Deploy from a branch**
   - Select branch: `gh-pages` and folder: `/ (root)`
   - Click Save

2. **Update Repository URL:**
   - Edit `mkdocs.yml` and replace `YOUR_USERNAME` with your actual GitHub username

3. **Push Changes:**
   ```bash
   git add .
   git commit -m "Add MkDocs documentation site"
   git push origin main
   ```

4. **Automatic Deployment:**
   - The GitHub Action will automatically build and deploy your site
   - Check the "Actions" tab to monitor the deployment
   - Your site will be live at: `https://arjun-christopher.github.io/AgriTwin-GH/`

5. **Share the URL:**
   - Anyone with the link can view your documentation
   - The repository remains private, only the built site is public

---

### Option 2: Netlify (Alternative)

For custom domains and advanced features:

1. **Create Netlify Account:**
   - Sign up at [netlify.com](https://netlify.com)
   - Connect your GitHub account

2. **Deploy Site:**
   - Click "Add new site" → "Import an existing project"
   - Select your repository
   - Build command: `mkdocs build`
   - Publish directory: `site`
   - Click "Deploy"

3. **Set Up Secrets (for auto-deployment from GitHub Actions):**
   - Get your Netlify Auth Token from User Settings → Applications
   - Get your Site ID from Site Settings → Site information
   - Add to GitHub repository secrets:
     - `NETLIFY_AUTH_TOKEN`
     - `NETLIFY_SITE_ID`
   
4. **Uncomment Netlify deployment in `.github/workflows/deploy-docs.yml`**

5. **Your site will be at:** `https://your-site-name.netlify.app`

---

## 🚀 Local Development

Test your documentation site locally before deploying:

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Serve Locally:**
   ```bash
   mkdocs serve
   ```

3. **View at:** `http://127.0.0.1:8000`

4. **Live Reload:** Any changes to markdown files will automatically refresh in the browser

---

## 📝 Adding New Documentation

### Automatic Discovery

Simply add markdown files anywhere in your repository - MkDocs will automatically discover and include them in the site!

Example locations:
- `docs/new-guide.md`
- `docs/tutorials/tutorial-1.md`
- `feature_demos/new-demo.md`

### Manual Navigation (Optional)

To customize the navigation order, edit the `nav:` section in `mkdocs.yml`:

```yaml
nav:
  - Home: index.md
  - Feature Demos:
    - Guide: feature_demos/FEATURE_DEMOS_GUIDE.md
  - Your New Section:
    - docs/your-page.md
```

---

## 🎨 Customization

### Color Scheme

Edit `mkdocs.yml` to change colors:

```yaml
theme:
  palette:
    primary: green  # Change to: blue, indigo, purple, pink, red, etc.
    accent: light-green
```

### Features

Enable/disable features in `mkdocs.yml`:

```yaml
theme:
  features:
    - navigation.tabs
    - navigation.sections
    - search.suggest
    - content.code.copy
```

See [Material for MkDocs Features](https://squidfunk.github.io/mkdocs-material/setup/setting-up-navigation/) for more options.

---

## 🔄 How It Works

1. **You add/edit markdown files** in your repository
2. **Push to GitHub** (`git push`)
3. **GitHub Actions automatically:**
   - Builds the MkDocs site
   - Deploys to `gh-pages` branch (for GitHub Pages) or Netlify
4. **Your documentation is instantly updated** at the public URL
5. **Scalable:** New markdown files are automatically included

---

## 🔗 Sharing the Link

- Share your site URL with anyone
- No GitHub account required to view
- Your repository code remains private
- Only the documentation website is public

**GitHub Pages URL:** `https://YOUR_USERNAME.github.io/AgriTwin-GH/`  
**Netlify URL:** `https://your-site-name.netlify.app`

---

## 📚 Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [Markdown Guide](https://www.markdownguide.org/)

---

## 🐛 Troubleshooting

### Site not deploying?
- Check the "Actions" tab in GitHub for error messages
- Ensure GitHub Pages is enabled in repository Settings → Pages
- Verify the `gh-pages` branch exists

### Broken links?
- Use relative paths: `[Link](../path/to/file.md)`
- Check file paths are correct in `mkdocs.yml`

### Need help?
- Check the workflow logs in the "Actions" tab
- Ensure all paths in `mkdocs.yml` point to existing files

---

**Happy Documenting! 🎉**
