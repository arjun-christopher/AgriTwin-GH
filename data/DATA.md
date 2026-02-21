# Data Directory

This directory contains all data files for the AgriTwin-GH project.

## Structure

- **raw/**: Raw, immutable data files as received from sensors or external sources
  - Never modify files in this directory
  - Original sensor readings, weather data, etc.

- **processed/**: Cleaned and processed data ready for analysis or modeling
  - Final datasets used by the application
  - Feature-engineered datasets

- **interim/**: Intermediate data transformations
  - Temporary processing stages
  - Data in transformation pipeline

- **external/**: External reference data from third-party sources
  - Weather databases
  - Plant disease databases
  - Reference datasets

## Data Management

- Add data files to `.gitignore` if they are large or contain sensitive information
- Use DVC (Data Version Control) for tracking large datasets
- Document data sources and update dates in this README
