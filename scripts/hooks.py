import shutil

def on_pre_build(config):
    """Copy README.md to docs/index.md before every build so README is the site home page."""
    shutil.copy("README.md", "docs/index.md")
