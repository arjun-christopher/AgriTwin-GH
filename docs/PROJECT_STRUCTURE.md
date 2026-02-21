# AgriTwin-GH Project Structure

This document describes the organization of the AgriTwin-GH project.

## Directory Structure

```
AgriTwin-GH/
├── src/                          # Source code
│   └── agritwin/                 # Main application package
│       ├── __init__.py
│       ├── core/                 # Core functionality
│       │   └── __init__.py
│       ├── models/               # Data models and ML models
│       │   └── __init__.py
│       ├── services/             # Business logic and services
│       │   └── __init__.py
│       ├── api/                  # API endpoints and routes
│       │   └── __init__.py
│       └── utils/                # Utility functions and helpers
│           └── __init__.py
│
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── unit/                     # Unit tests
│   └── integration/              # Integration tests
│
├── data/                         # Data files (see data/README.md)
│   ├── raw/                      # Raw, immutable data
│   ├── processed/                # Cleaned, processed data
│   ├── interim/                  # Intermediate transformations
│   └── external/                 # External reference data
│
├── config/                       # Configuration files
│   ├── settings.yaml             # Application settings
│   └── logging.yaml              # Logging configuration
│
├── scripts/                      # Utility scripts
│   ├── setup.py                  # Setup scripts
│   └── data_processing/          # Data processing scripts
│
├── notebooks/                    # Jupyter notebooks for exploration
│   └── exploratory/              # Exploratory data analysis
│
├── feature_demos/                # Feature demonstrations (existing)
│   ├── *.ipynb                   # Demo notebooks
│   └── data/                     # Demo-specific data
│
├── docs/                         # Documentation
│   ├── index.md                  # Documentation home
│   └── api/                      # API documentation
│
├── logs/                         # Application logs
│   └── .gitkeep                  # Keep directory in git
│
├── mkdocs.yml                    # MkDocs configuration
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup file
├── .gitignore                    # Git ignore rules
└── README.md                     # Project README

```

## Directory Purpose

### `/src/agritwin/`
Main application source code organized by functionality:
- **core/**: Core system components (digital twin engine, simulator)
- **models/**: Data models, ML models, disease risk models
- **services/**: Business logic (sensor data processing, control policies)
- **api/**: REST API endpoints if building a web service
- **utils/**: Helper functions, constants, data generators

### `/tests/`
Test suite following the same structure as `/src/`:
- **unit/**: Fast, isolated unit tests
- **integration/**: Tests for integrated components

### `/data/`
All data files organized by processing stage (see [data/README.md](data/README.md))

### `/config/`
Configuration files for various environments (dev, staging, production)

### `/scripts/`
Standalone scripts for setup, data processing, deployment

### `/notebooks/`
Jupyter notebooks for experimentation and analysis (separate from feature demos)

### `/logs/`
Application runtime logs (excluded from git via .gitignore)

## Development Workflow

1. **Source code** goes in `src/agritwin/`
2. **Tests** mirror the structure in `tests/`
3. **Data files** are organized by stage in `data/`
4. **Configuration** is stored in `config/`
5. **Documentation** is written in `docs/` and built with MkDocs
6. **Experiments** are conducted in `notebooks/`

## Best Practices

- Keep `data/raw/` immutable
- Use descriptive module and file names
- Write tests for all new features
- Document public APIs and complex logic
- Use configuration files instead of hardcoded values
- Add large data files to `.gitignore`
