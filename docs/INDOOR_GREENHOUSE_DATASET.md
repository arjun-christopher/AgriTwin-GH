# Indoor Greenhouse Dataset Generation

## Overview

This document describes the methodology and process for generating synthetic indoor greenhouse datasets from outdoor weather data for the AgriTwin-GH project. The datasets simulate passive greenhouse conditions without active environmental control systems, providing realistic baseline data for greenhouse monitoring and digital twin applications.

## Generated Datasets

Two hourly indoor greenhouse datasets have been created:

| Dataset | Location | Records | Period | Size |
|---------|----------|---------|--------|------|
| `dindigul_greenhouse_indoor_2024.csv` | `data/processed/` | 8,784 | Jan 1 - Dec 31, 2024 | 1.03 MB |
| `dindigul_greenhouse_indoor_2025.csv` | `data/processed/` | 8,760 | Jan 1 - Dec 31, 2025 | 1.02 MB |

Both datasets contain **24 hourly records per day** with no missing values.

## Source Data

### Input Files
- `data/external/Weather Data/dindigul_weather_2024.csv`
- `data/external/Weather Data/dindigul_weather_2025.csv`

### Source Data Structure
The original weather files contain daily outdoor measurements with the following key columns:
- `datetime` - Date/time stamp
- `datetimeEpoch` - Unix epoch timestamp
- `temp` - Outdoor temperature (°C)
- `humidity` - Outdoor relative humidity (%)
- `windspeed` - Wind speed (km/h)
- `solarradiation` - Solar radiation (W/m²)
- `sunriseEpoch` - Sunrise time (Unix epoch)
- `sunsetEpoch` - Sunset time (Unix epoch)

## Transformation Methodology

### Step 1: Data Loading and Standardization

Raw daily weather data is loaded and column names are standardized:
- `temp` → `outdoor_temp`
- `humidity` → `outdoor_humidity`
- `windspeed` → `outdoor_windspeed`
- `solarradiation` → `solar`

### Step 2: Temporal Upsampling to Hourly Resolution

Daily records are expanded to hourly resolution (24 records/day):

1. **Complete Hourly Index Creation**: A full hourly datetime index is generated spanning each entire year
2. **Daily Value Distribution**: Each daily measurement is distributed across 24 hourly records
3. **Diurnal Variation Addition**:
   - **Temperature**: Sinusoidal pattern with ±3°C variation, peak at 14:00, minimum at 05:00
   - **Humidity**: Inverse pattern with ±5% variation (clamped 20-100%)
   - **Solar Radiation**: Distributed using sine pattern across daylight hours (06:00-18:00), zero at night

Mathematical formula for diurnal temperature variation:
```
temp_variation = 3 × sin((hour - 5) × π / 12)
hourly_temp = daily_temp + temp_variation
```

### Step 3: Day/Night Classification

A binary `day_night_flag` is computed using actual sunrise/sunset times:
- **Day (1)**: datetimeEpoch between sunriseEpoch and sunsetEpoch
- **Night (0)**: Outside daylight hours

Fallback heuristic (if sunrise/sunset unavailable): Day = 06:00–18:00

## Passive Greenhouse Physics Model

The indoor conditions are derived using a passive greenhouse model that simulates natural greenhouse behavior without active climate control.

### A) Indoor Temperature (°C)

Indoor temperature accounts for solar heating during day and modest heat retention at night:

**Day (solar heating effect):**
```
ΔT = 0.02 × solar_radiation
indoor_temp = outdoor_temp + ΔT
```

**Night (heat retention):**
```
ΔT = 1.5°C
indoor_temp = outdoor_temp + 1.5
```

**Rationale**: Greenhouse glazing traps solar radiation during day (proportional to solar intensity). At night, structural thermal mass provides modest temperature elevation above outdoor conditions.

### B) Indoor Humidity (%)

Humidity is affected by temperature-driven evapotranspiration and condensation:

**Day (drying effect):**
```
indoor_humidity = outdoor_humidity - (0.5 × ΔT) + 0.4
```

**Night (moisture accumulation):**
```
indoor_humidity = outdoor_humidity + 5
```

**Constraints**: Clamped between 30% and 100%

**Rationale**: Daytime heating reduces relative humidity through evapotranspiration. Nighttime cooling increases relative humidity as water vapor condenses on cooler surfaces.

### C) Indoor Air Velocity (m/s)

Indoor air movement is reduced compared to outdoor wind:
```
indoor_air_velocity = outdoor_windspeed × 0.1
```

**Rationale**: Greenhouse structure shields interior from direct wind, reducing air velocity to ~10% of outdoor conditions.

### D) Indoor CO₂ Concentration (ppm)

CO₂ levels fluctuate with photosynthesis (day) and respiration (night):

**Day (photosynthetic depletion):**
```
indoor_CO2 = 400 - (0.05 × solar_radiation)
```

**Night (respiratory accumulation):**
```
indoor_CO2 = 440 ppm
```

**Constraints**: Minimum 300 ppm

**Rationale**: Plant photosynthesis depletes CO₂ during daylight (proportional to light intensity). At night, plant and soil respiration increases CO₂ concentration above ambient levels.

## Derived Environmental Features

### E) Dew Point Temperature (°C)

Approximation of dew point using simplified Magnus formula:
```
dew_point = indoor_temp - ((100 - indoor_humidity) / 5)
```

### F) Vapor Pressure Deficit - VPD (kPa)

Critical metric for plant transpiration and disease risk:

```
SVP = 0.6108 × exp((17.27 × T) / (T + 237.3))  # Saturation Vapor Pressure
AVP = SVP × (RH / 100)                          # Actual Vapor Pressure
VPD = SVP - AVP
```

Where:
- T = indoor_temp (°C)
- RH = indoor_humidity (%)

**Interpretation**:
- Low VPD (<0.4 kPa): Stagnant conditions, high disease risk
- Optimal VPD (0.8-1.2 kPa): Ideal for most crops
- High VPD (>1.5 kPa): Excessive transpiration, plant stress

### G) Leaf Wetness Proxy (binary)

Binary indicator of leaf surface moisture based on sustained high humidity:

```
leaf_wetness_proxy = 1 if (indoor_humidity > 85%) for ≥3 consecutive hours, else 0
```

**Implementation**: Rolling 3-hour window checking for humidity >85%

**Significance**: Extended leaf wetness periods are strong predictors of fungal disease development.

## Output Dataset Specification

### Column Schema

| Column | Type | Unit | Description | Range |
|--------|------|------|-------------|-------|
| `datetime` | datetime | - | Hourly timestamp | - |
| `indoor_temp` | float | °C | Indoor air temperature | 18-43°C |
| `indoor_humidity` | float | % | Relative humidity | 30-100% |
| `indoor_air_velocity` | float | m/s | Air movement speed | 0.1-2.6 m/s |
| `indoor_CO2` | float | ppm | Carbon dioxide concentration | 300-440 ppm |
| `solarradiation` | float | W/m² | Solar radiation intensity | 0-250 W/m² |
| `day_night_flag` | int | - | Day=1, Night=0 | 0 or 1 |
| `vpd` | float | kPa | Vapor pressure deficit | 0-5 kPa |
| `dew_point` | float | °C | Dew point temperature | 10-30°C |
| `leaf_wetness_proxy` | int | - | Leaf wetness indicator | 0 or 1 |

### Data Quality Guarantees

✅ **Temporal Completeness**: Exactly 24 records per calendar day  
✅ **No Missing Values**: All columns fully populated  
✅ **Physical Constraints**: All values within realistic bounds  
✅ **Temporal Continuity**: Chronological ordering preserved  

## Validation Results

### 2024 Dataset (Leap Year)

```
Total Records:        8,784 (366 days × 24 hours)
Date Range:           2024-01-01 00:00:00 to 2024-12-31 23:00:00
Indoor Temperature:   19.8 to 42.8 °C
Indoor Humidity:      43.0 to 100.0%
Indoor CO₂:           384.0 to 440.0 ppm
VPD:                  0.00 to 4.38 kPa
Daylight Hours:       4,435 (50.5%)
Leaf Wetness Hours:   864 (9.8%)
```

### 2025 Dataset

```
Total Records:        8,760 (365 days × 24 hours)
Date Range:           2025-01-01 00:00:00 to 2025-12-31 23:00:00
Indoor Temperature:   18.4 to 41.2 °C
Indoor Humidity:      38.3 to 100.0%
Indoor CO₂:           383.9 to 440.0 ppm
VPD:                  0.00 to 4.76 kPa
Daylight Hours:       4,402 (50.3%)
Leaf Wetness Hours:   502 (5.7%)
```

## Usage Examples

### Loading the Dataset

```python
import pandas as pd

# Load indoor greenhouse data
df_2024 = pd.read_csv('data/processed/dindigul_greenhouse_indoor_2024.csv')
df_2024['datetime'] = pd.to_datetime(df_2024['datetime'])

# Set datetime as index
df_2024.set_index('datetime', inplace=True)

print(f"Loaded {len(df_2024)} hourly records")
```

### Basic Analysis

```python
# Daily aggregations
daily_avg = df_2024.resample('D').mean()

# Temperature statistics
print(f"Mean indoor temp: {df_2024['indoor_temp'].mean():.1f}°C")
print(f"Max indoor temp: {df_2024['indoor_temp'].max():.1f}°C")

# High VPD stress events
high_vpd_hours = (df_2024['vpd'] > 1.5).sum()
print(f"Hours with high VPD stress: {high_vpd_hours}")

# Disease risk periods
leaf_wetness_hours = df_2024['leaf_wetness_proxy'].sum()
print(f"Hours with leaf wetness: {leaf_wetness_hours}")
```

### Time Series Visualization

```python
import matplotlib.pyplot as plt

# Plot one week of data
week_data = df_2024['2024-07-01':'2024-07-07']

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# Temperature
axes[0].plot(week_data.index, week_data['indoor_temp'], label='Indoor')
axes[0].set_ylabel('Temperature (°C)')
axes[0].legend()

# Humidity and VPD
ax_vpd = axes[1].twinx()
axes[1].plot(week_data.index, week_data['indoor_humidity'], 'g-', label='Humidity')
ax_vpd.plot(week_data.index, week_data['vpd'], 'r--', label='VPD')
axes[1].set_ylabel('Humidity (%)', color='g')
ax_vpd.set_ylabel('VPD (kPa)', color='r')

# CO2 with day/night shading
axes[2].plot(week_data.index, week_data['indoor_CO2'])
axes[2].fill_between(week_data.index, 0, 500, 
                      where=week_data['day_night_flag']==0, 
                      alpha=0.2, color='gray', label='Night')
axes[2].set_ylabel('CO₂ (ppm)')
axes[2].legend()

plt.tight_layout()
plt.show()
```

### Disease Risk Assessment

```python
# Calculate daily disease risk score
df_2024['disease_risk'] = (
    (df_2024['leaf_wetness_proxy'] == 1).astype(int) * 0.4 +  # Leaf wetness
    (df_2024['indoor_humidity'] > 80).astype(int) * 0.3 +      # High humidity
    (df_2024['vpd'] < 0.4).astype(int) * 0.3                   # Low VPD
)

daily_risk = df_2024.groupby(df_2024.index.date)['disease_risk'].mean()

# Identify high-risk days
high_risk_days = daily_risk[daily_risk > 0.5]
print(f"High disease risk days: {len(high_risk_days)}")
```

## Processing Pipeline

### Notebook Reference

Complete implementation details available in:  
📓 [`notebooks/generate_passive_greenhouse_data.ipynb`](../notebooks/generate_passive_greenhouse_data.ipynb)

### Pipeline Steps

1. **Load and Standardize** - Import CSV, rename columns, parse datetime
2. **Expand to Hourly** - Create 24 records/day with diurnal patterns
3. **Add Day/Night Flag** - Classify daylight hours using sunrise/sunset
4. **Generate Greenhouse Conditions** - Apply passive physics model
5. **Calculate Derived Features** - Compute VPD, dew point, leaf wetness
6. **Validate and Save** - Quality checks and CSV export

### Reproducibility

To regenerate the datasets:

```bash
# Open Jupyter notebook
jupyter notebook notebooks/generate_passive_greenhouse_data.ipynb

# Run all cells (or use "Run All" from Cell menu)
```

The pipeline will:
- Load outdoor weather data for both years
- Transform to hourly indoor greenhouse conditions
- Validate 24 records/day with no missing values
- Save outputs to `data/processed/`

## Scientific Basis and Assumptions

### Model Assumptions

1. **Passive Operation**: No active heating, cooling, ventilation, or CO₂ enrichment systems
2. **Standard Greenhouse Structure**: Single-layer polyethylene or glass covering
3. **Moderate Plant Density**: Typical vegetable crop canopy coverage
4. **Natural Ventilation**: Passive air exchange through vents/openings
5. **No Irrigation Control**: Natural evapotranspiration only

### Physical Principles

- **Greenhouse Effect**: Solar radiation transmission and infrared re-radiation trapping
- **Thermal Mass**: Floor and structure heat storage/release
- **Boundary Layer**: Reduced wind exposure inside structure
- **Photosynthesis/Respiration**: Plant metabolic CO₂ exchange
- **Evapotranspiration**: Plant water loss affecting humidity

### Limitations

- Simplified linear relationships (actual greenhouse dynamics are non-linear)
- No consideration of crop type, growth stage, or plant density variations
- Uniform spatial conditions (no vertical or horizontal gradients)
- No extreme weather events (storms, frost, heat waves) modeling
- No equipment failures or structural damage scenarios

## Applications

### Suitable Use Cases

✅ **Digital Twin Development**: Baseline greenhouse behavior modeling  
✅ **Control System Testing**: Benchmarking against uncontrolled conditions  
✅ **Disease Risk Prediction**: Training ML models for pathogen outbreak forecasting  
✅ **Growth Stage Simulation**: Environmental condition correlation with plant development  
✅ **Energy Analysis**: Passive vs. active system comparison  
✅ **Irrigation Scheduling**: VPD-based watering optimization  

### Not Recommended For

❌ Active climate control system design  
❌ Precise economic cost modeling  
❌ Specific crop variety performance prediction  
❌ Structural engineering calculations  

## References and Related Documentation

- [Project Structure](PROJECT_STRUCTURE.md) - Repository organization
- [Feature Demos Guide](../feature_demos/FEATURE_DEMOS_GUIDE.md) - Usage examples
- [Data Directory](../data/DATA.md) - Data sources and organization

## Changelog

### Version 1.0 - February 2026
- Initial dataset generation for 2024 and 2025
- Passive greenhouse physics model implementation
- Hourly resolution with diurnal patterns
- VPD and leaf wetness feature engineering

---

**Generated**: February 24, 2026  
**Author**: AgriTwin-GH Data Engineering Team  
**Contact**: arjun-christopher/AgriTwin-GH
