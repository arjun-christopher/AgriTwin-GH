# Frontend UI Reference — AgriTwin-GH

> **Purpose:** Complete reference for the React dashboard UI — pages, components, mock data locations, and the exact API calls needed to swap each piece of static data for live backend data. Use this document when integrating the FastAPI backend.

---

## Table of Contents

1. [Tech Stack & Setup](#1-tech-stack--setup)
2. [Project Structure](#2-project-structure)
3. [Routing & Shell](#3-routing--shell)
4. [Design System & Theming](#4-design-system--theming)
5. [Shared Services Layer](#5-shared-services-layer--srcservicesapijs)
6. [Centralised Mock Data](#6-centralised-mock-data--srcdatamockdatajs)
7. [Pages](#7-pages)
   - [SplashScreen](#71-splashscreen)
   - [HomeDashboard](#72-homedashboard)
   - [DetailedInsights](#73-detailedinsights)
   - [ManualOverride](#74-manualoverride)
8. [Shared Components](#8-shared-components)
9. [API Integration Checklist](#9-api-integration-checklist)
10. [Environment Variables](#10-environment-variables)
11. [Build & Dev Commands](#11-build--dev-commands)

---

## 1. Tech Stack & Setup

| Tool | Version | Role |
|------|---------|------|
| React | 19 | UI framework |
| Vite | 6.4.x | Dev server + bundler |
| Tailwind CSS | v4 | Utility CSS — **no config file**, tokens declared in `src/index.css` via `@theme {}` |
| lucide-react | 0.511.x | Icon library — tree-shaken, no extra config |
| `@tailwindcss/vite` | 4.x | Tailwind v4 Vite plugin (replaces PostCSS) |

### Key points for contributors

- **No `tailwind.config.js`** — Tailwind v4 reads design tokens directly from CSS `@theme {}` block in `src/index.css`. Do not create a config file.
- **No React Router** — navigation is a simple `useState('page')` in `App.jsx`. There are no URL routes; pages are conditionally rendered.
- **No Redux / Zustand** — global state consists of a single `ThemeContext` for dark/light mode only. All other state is local per-page.
- **Module type** — `package.json` sets `"type": "module"`, so all JS files use ES module syntax (`import`/`export`).
- **Icon rule** — `src/data/mockData.js` stores icon identifiers as strings (e.g., `'Fan'`, `'Droplets'`) because that file has no React imports. Each page that consumes these icons defines its own `ICON_MAP = { Fan, Droplets, … }` lookup and resolves them at render time.

### Install & run

```powershell
# From project root
cd src/agritwin_gh/frontend

npm install          # install dependencies
npm run dev          # dev server → http://localhost:5173
npm run build        # production build → dist/
npm run preview      # serve the production build locally
npm run lint         # ESLint
```

> Set `VITE_API_BASE_URL` in `.env.local` before running dev/build to point at the FastAPI server (default falls back to `http://localhost:8000`).

---

## 2. Project Structure

```
src/agritwin_gh/frontend/
├── package.json
├── index.html                   # HTML entry — mounts <div id="root">
└── src/
    ├── main.jsx                 # ReactDOM.createRoot → <App />
    ├── App.jsx                  # Root: ThemeProvider + AppRouter (page state)
    ├── index.css                # @theme tokens + font imports + light-theme overrides
    ├── theme.js                 # (currently unused helper — reserved for future use)
    │
    ├── context/
    │   └── ThemeContext.jsx     # useTheme() → { isDark, toggleTheme }
    │                            # Writes data-theme="light" on <html> for CSS vars
    │
    ├── data/
    │   └── mockData.js          # SINGLE SOURCE of all static data (exported constants)
    │                            # See §6 for full inventory
    │
    ├── services/
    │   └── api.js               # FastAPI stub layer — every function returns mock data now
    │                            # See §5 for full API map
    │
    ├── components/
    │   ├── layout/
    │   │   ├── AppShell.jsx     # Persistent scroll wrapper — renders Header + Navbar + <main>
    │   │   ├── Header.jsx       # Fixed top bar: brand, live UTC clock, theme toggle
    │   │   ├── Navbar.jsx       # Fixed nav below header: Dashboard / Insights / Override links
    │   │   └── Sidebar.jsx      # (file exists, not currently wired into AppShell)
    │   └── ui/
    │       ├── GlassPanel.jsx   # Frosted-glass card container
    │       ├── PanelCard.jsx    # Standard content card with optional header slot
    │       └── StatusCard.jsx   # Crop Health / Weather status tile (Healthy/Warning/Risk)
    │
    └── pages/
        ├── SplashScreen.jsx     # Animated intro — auto-advances to HomeDashboard
        ├── HomeDashboard.jsx    # Main overview page (route key: 'dashboard')
        ├── DetailedInsights.jsx # Deep-dive analysis page (route key: 'insights')
        └── ManualOverride.jsx   # Manual simulation control page (route key: 'override')
```

---

## 3. Routing & Shell

**File:** `src/App.jsx`

Navigation is handled by a single `useState` in `AppRouter`. There is no URL router.

```jsx
const [page, setPage] = useState('splash');
// valid values: 'splash' | 'dashboard' | 'insights' | 'override'
```

- `SplashScreen` renders when `page === 'splash'` and fires `onComplete` after 4.2 s, setting page to `'dashboard'`.
- All other pages are looked up via `PAGE_MAP` and rendered inside `AppShell`.
- `navigate` is passed down as a prop so any page can call `navigate('insights')` etc.

**AppShell layout (`AppShell.jsx`):**

```
┌──────────────────────────────────────────┐  ← fixed, z-50, h-14 (56px)
│  Header (brand + clock + theme toggle)   │
├──────────────────────────────────────────┤  ← fixed, z-40, h-11 (44px), top-14
│  Navbar (Dashboard / Insights / Override)│
├──────────────────────────────────────────┤
│                                          │
│  <main>  pt-25 px-6 md:px-8 pb-16        │  ← scrollable page content
│  (page component renders here)          │
│                                          │
└──────────────────────────────────────────┘
```

**Header (`Header.jsx`):**
- Left: Sprout icon + "AgriTwin-GH" wordmark (clicking it calls `navigate('dashboard')`)
- Right: live UTC date/time (ticks every second via `setInterval`) + dark/light toggle button
- No API data — fully static/local

**Navbar (`Navbar.jsx`):**
- Three nav buttons: Dashboard, Detailed Insights, Manual Override
- Active page shown with primary-colour underline
- No API data — fully static

---

## 4. Design System & Theming

**File:** `src/index.css`

All design tokens are CSS custom properties declared in the `@theme {}` block. Tailwind v4 converts these into utility classes automatically.

### Colour tokens

| Token | Dark value | Light value | Usage |
|-------|-----------|------------|-------|
| `--color-background` | `#0b1326` | `#eef4ea` | Page/body background |
| `--color-surface` | `#171f33` | `#e5ece0` | Panel backgrounds |
| `--color-surface-low` | `#131b2e` | `#e9f0e5` | Recessed areas |
| `--color-surface-deep` | `#060e20` | `#f5f9f3` | Header bar |
| `--color-surface-high` | `#222a3d` | `#d5e0ca` | Elevated cards, hover states |
| `--color-surface-highest` | `#2d3449` | `#c5d4b8` | Badges, tooltips |
| `--color-primary` | `#4be277` | `#22c55e` | Brand green — active states |
| `--color-secondary` | `#89ceff` | `#5db8f5` | Accent blue |
| `--color-danger` | `#f87171` | — | Error / risk |
| `--color-warning` | `#fbbf24` | — | Caution / override mode |
| `--color-on-surface` | `#dae2fd` | — | Primary text |
| `--color-on-surface-variant` | `#bccbb9` | — | Secondary/muted text |
| `--color-on-primary` | `#003915` | — | Text on primary buttons |

### Typography tokens

| Token | Value |
|-------|-------|
| `--font-sans` | Inter (loaded from Google Fonts) |
| `--font-headline` | Space Grotesk (Google Fonts) |
| `--font-mono` | JetBrains Mono / Fira Code |

### Theme switching

`ThemeContext.jsx` toggles `data-theme="light"` on `<html>`. The CSS in `index.css` overrides surface tokens under `[data-theme="light"]`. Dark mode is the default.

### Tailwind class construction rule

**All Tailwind utility classes must be literal strings in source code.** Because Tailwind v4 scans source files at build time, never build class strings dynamically (e.g., `'text-' + color`). Instead, use conditional maps:

```jsx
// CORRECT — both classes appear as literals:
const pill = active ? 'bg-primary/10 text-primary' : 'bg-surface-highest text-on-surface-variant';

// WRONG — class strings won't be detected:
const pill = `bg-${color}/10 text-${color}`;
```

---

## 5. Shared Services Layer — `src/services/api.js`

This file is the **integration boundary** between the frontend and backend. Currently every function returns mock data. When the backend is ready, swap in the `fetch()` call.

**Base URL:** reads `VITE_API_BASE_URL` from `.env.local`, defaults to `http://localhost:8000`.

### Integration steps (per function)

1. Set `VITE_API_BASE_URL=http://your-fastapi-host` in `.env.local`
2. Remove the `import * as mock from '../data/mockData.js'` import once all functions are live
3. For each function, uncomment the `const data = await get(...)` line and delete the `return Promise.resolve(mock.XXX)` line
4. Add error handling (401 → re-authenticate, 503 → show toast/banner)

### Full API function inventory

| Function | HTTP | Endpoint | Returns mock | Backend source |
|----------|------|----------|-------------|----------------|
| `getDtState()` | GET | `/api/dt/state` | `CROP`, `CROP_HEALTH`, `SENSOR_METRICS`, `GROWTH_INTEL` | `dt_core.py` |
| `postDtOverride(params)` | POST | `/api/dt/override` | `{ ok: true }` | `dt_core.py` |
| `postDtPreset(presetId)` | POST | `/api/dt/preset/:id` | `{ ok: true, preset }` | `dt_core.py` |
| `getActuatorState()` | GET | `/api/actuators/state` | `ACTUATOR_STATE` | `mpc_controller.py` |
| `postActuatorSet(actuators)` | POST | `/api/actuators/set` | `{ ok: true }` | `mpc_controller.py` |
| `getWeather()` | GET | `/api/weather/current` | `WEATHER_STATUS`, `OUTDOOR_CURRENT`, `OUTDOOR_FORECAST` | weather forecast model |
| `getDiseaseRisks()` | GET | `/api/intelligence/disease` | `DISEASE_RISKS` | disease progression model |
| `getGrowthIntel()` | GET | `/api/intelligence/growth` | `GROWTH_INTEL` | growth progression model |
| `getMonthlyResources()` | GET | `/api/resources/monthly` | `MONTHLY_RESOURCES`, `MONTHLY_COST` | resource tracking |
| `getLatestMedia()` | GET | `/api/media/latest` | `CROP_IMAGES` | MinIO + PostgreSQL metadata |
| `getStageImages()` | GET | `/api/media/stage-images` | `GROWTH_STAGE_IMAGES` | MinIO |
| `getDiseaseImages()` | GET | `/api/media/disease-scans` | `DISEASE_IMAGES` | MinIO |
| `getSystemHealth()` | GET | `/api/system/health` | `HEALTH_ROWS` | system diagnostics |

### POST body shapes

**`postDtOverride`**
```json
{ "param": "temperature_setpoint", "value": 25.0 }
```

**`postActuatorSet`**
```json
{ "actuators": [{ "id": "fan", "level": 80 }, { "id": "heater", "level": 0 }] }
```

**`postDtPreset`**
```
POST /api/dt/preset/day-cycle
POST /api/dt/preset/night-cycle
POST /api/dt/preset/emergency-flush
```

---

## 6. Centralised Mock Data — `src/data/mockData.js`

All static data used across pages lives here as named exports. Pages import from this file directly, or consume it indirectly via `api.js`. The comments in the file already list the target API endpoint for each constant.

### Data constants inventory

#### `CROP`
```js
{
  current: 'Flowering',      // string — current growth stage name
  currentIndex: 3,           // 0-based index into stages[]
  currentPct: 95,            // % completion of current stage
  daysInStage: 12,           // days elapsed in current stage
  stageDuration: 14,         // total days expected for this stage
  next: 'Fruiting',          // next stage name
  nextInDays: 2,             // estimated days until transition
  stages: ['Germination', 'Seedling', 'Vegetative', 'Flowering', 'Fruiting', 'Harvest'],
}
```
> ⚠️ **Note:** `mockData.js` uses an older 6-stage list. `HomeDashboard.jsx` and `DetailedInsights.jsx` define their own local `CROP` constants with the canonical 6 stages (`Seedling / Early Veg. / Flw. Init. / Flowering / Unripe / Ripe`). When integrating, the API response should use the canonical stage names from `constants.py`.

**→ API:** `GET /api/dt/state` → `.crop`

---

#### `CROP_HEALTH`
```js
{
  status: 'Healthy',        // 'Healthy' | 'Warning' | 'Risk'
  confidence: 99.5,         // classifier confidence % (not shown in current UI)
  detail: '...',            // one-sentence description
  scannedAgo: '2 mins ago', // human-readable "X ago" string
}
```
**→ API:** `GET /api/dt/state` → `.health`

---

#### `WEATHER_STATUS`
```js
{
  status: 'Warning',              // 'Healthy' | 'Warning' | 'Risk'
  condition: 'High External Humidity',
  detail: '...',
  forecast: 'Expected to clear in ~3 hrs',
}
```
**→ API:** `GET /api/weather/current` → `.status`

---

#### `OUTDOOR_CURRENT`
```js
{
  temp: 18.4, humidity: 89, windSpeed: 12, windDir: 'NNE',
  pressure: 1013, uvIndex: 3, dewPoint: 16.8,
  condition: 'Overcast', visibility: 8.2,
}
```
**→ API:** `GET /api/weather/current` → `.current`

---

#### `OUTDOOR_FORECAST`
Array of 6 hourly entries:
```js
{ time: '12:00', high: 22, low: 19, humidity: 74, condition: 'Partly Cloudy', iconKey: 'CloudSun' }
```
> `iconKey` is a lucide icon name. DetailedInsights resolves it via its local `ICON_MAP`.

**→ API:** `GET /api/weather/current` → `.forecast` (array of N entries)

---

#### `ACTUATOR_STATE`
Array of 6 items (older list — pages define their own updated 7-actuator lists inline):
```js
{
  id: 'fan', iconKey: 'Fan', label: 'Ventilation Fan',
  status: 'ON', active: true, level: 75, color: 'primary'
}
```
> **Per-page actuator IDs:** `fan | vent | irrigation | heater | led | co2 | fogger` (7 actuators reflecting MPC `CONTROL_VARIABLES`)

**→ API:** `GET /api/actuators/state`

---

#### `MONTHLY_RESOURCES`
```js
[
  { label: 'Water',  used: 1840, unit: 'L'   },
  { label: 'Energy', used: 312,  unit: 'kWh' },
]
```
**→ API:** `GET /api/resources/monthly` → `.resources`

---

#### `MONTHLY_COST`
```js
{ month: 'April 2026', energy: 43.68, water: 7.36, total: 51.04 }
```
**→ API:** `GET /api/resources/monthly` → `.cost`

---

#### `CROP_IMAGES`
```js
{
  stage: { src: 'https://...', alt: '...', badge: 'Day 12', location: 'GH-04 · Camera 2', captured: '1 hr ago' },
  leaf:  { src: 'https://...', alt: '...', badge: '99.5% Conf.', location: 'Sector B · Leaf #7', captured: '2 mins ago' },
}
```
> Images are currently from `https://picsum.photos` (random placeholder). Replace `src` with MinIO pre-signed URLs.

**→ API:** `GET /api/media/latest`

---

#### `SENSOR_METRICS`
Array of 4 summary chips (temp, humidity, CO₂, light):
```js
{ iconKey: 'Thermometer', label: 'Temperature', value: '24.2', unit: '°C', trend: 'up', trendLabel: '+0.4°', trendColor: 'text-warning' }
```
> `value` is a pre-formatted string. The API should return raw numbers; formatting happens in the component.

**→ API:** `GET /api/sensors/summary`

---

#### `HEALTH_ROWS`
```js
[
  { label: 'Sensor Array',  status: 'Nominal', ok: true  },
  { label: 'Network Link',  status: 'Strong',  ok: true  },
  { label: 'Data Pipeline', status: 'Active',  ok: true  },
  { label: 'Calibration',   status: 'Due: 6d', ok: false },
]
```
**→ API:** `GET /api/system/health`

---

#### `GROWTH_INTEL`
```js
{
  hoursToNextStage: 47,
  transitionProb24h: 18,    // % probability of transitioning within 24h
  stageHistory: [
    { stage: 'Seedling', daysUsed: 14, daysTarget: 14, complete: true },
    ...
  ],
}
```
**→ API:** `GET /api/intelligence/growth`

---

#### `DISEASE_RISKS`
Array of 5 disease entries (canonical `DISEASE_CATEGORIES` from `constants.py`):
```js
{ name: 'Late Blight', pathogen: 'P. infestans', risk24h: 31, severity: 'High' }
```
- `severity`: `'Low' | 'Medium' | 'High'`
- `risk24h`: 0–100 percentage risk score

**→ API:** `GET /api/intelligence/disease`

---

## 7. Pages

---

### 7.1 SplashScreen

**File:** `src/pages/SplashScreen.jsx`  
**Route key:** `'splash'` (internal only — never reachable after first load)

**Purpose:** Animated brand introduction that plays once on app open, then auto-advances to HomeDashboard.

**Mock data:** None — fully static.

**Behaviour:**
- Plays for `4200ms` total (`TIMING.introDuration`)
- Begins fade-out at `3700ms`, fires `onComplete` at `4200ms`
- Content stagger: background glows at 0ms → logo at 280ms → title at 820ms → tagline at 1260ms
- Shows logo, "AgriTwin-GH" headline, and tagline "Precision Greenhouse Intelligence"

**No API integration needed.**

---

### 7.2 HomeDashboard

**File:** `src/pages/HomeDashboard.jsx`  
**Route key:** `'dashboard'`

**Purpose:** Real-time monitoring overview — one-glance status for operators. Three-column layout.

#### Layout

```
┌─────────────────────────────────────────────────────────────┐
│  [Metric chips row: Temp · Humidity · CO₂ · Light]         │
├───────────────┬──────────────────────┬──────────────────────┤
│  LEFT PANEL   │   CENTRE PANEL        │   RIGHT PANEL        │
│  Growth Stage │   Crop Images         │   Actuator Status    │
│  Stage bars   │   Leaf Scan Image     │   Resource Usage     │
│  System Health│                       │   Cost Summary       │
└───────────────┴──────────────────────┴──────────────────────┘
```

#### Mock data constants (defined inline, not from mockData.js)

| Constant | Description | Target API |
|----------|-------------|----------|
| `CROP` | Current growth stage: name, index (0–5), % through stage, days in stage, stage duration, next stage, days to next. `stages[]` array used by `CropStageTrack` component. | `GET /api/dt/state` → `.crop` |
| `CROP_HEALTH` | Health status card: `status` ('Healthy'/'Warning'/'Risk'), `detail` text, `scannedAgo` string. | `GET /api/dt/state` → `.health` |
| `WEATHER` | Weather status card: `status`, `condition`, `detail`, `forecast`. | `GET /api/weather/current` → `.status` |
| `ACTUATORS` | Array of 7 actuator objects: `{ icon, label, status, active, color }`. State-only — no level values shown. | `GET /api/actuators/state` |
| `MONTHLY_RESOURCES` | `[{ label, used, unit }]` — Water and Energy. | `GET /api/resources/monthly` → `.resources` |
| `MONTHLY_COST` | `{ month, energy, water, total }` — cost breakdown. | `GET /api/resources/monthly` → `.cost` |
| `CROP_IMAGES` | Two images (stage + leaf): `{ src, alt, badge, location, captured }`. Images are picsum.photos placeholders. | `GET /api/media/latest` |
| `GROWTH_INTEL` | Stage history bars in left panel: `stageHistory[]` array. Uses abbreviated names for space (e.g., `'Seedl.'`, `'E.Veg.'`). | `GET /api/intelligence/growth` |
| `METRICS` | 4 large sensor metric cards in the top strip: Temp, Humidity, CO₂, Light. Each has `trend` direction + `trendLabel`. | `GET /api/sensors/summary` |
| `HEALTH_ROWS` | 4 system status rows (Sensor Array, Network Link, Data Pipeline, Calibration). | `GET /api/system/health` |

#### Sub-components (file-local, not shared)

| Component | Description |
|-----------|-------------|
| `MetricCard` | Large number + icon + trend arrow. Used in top sensor strip. |
| `CropStageTrack` | Horizontal stage bubble track with connecting line. Filled bubbles for past stages, pulsing ring for current, muted for future. |
| `ActuatorTile` | Compact coloured tile for actuator status. Shows icon + label + ON/OFF pill. 2×4 grid layout. |
| `ActuatorRow` | Slim row variant of actuator status — used in narrower layout contexts. |
| `PlaceholderBlock` | Shimmer-style grey box for sections not yet implemented. |
| `TrendIcon` | Arrow up / arrow down / trend icon based on `trend` string. |

#### Actuator IDs (7, matches MPC `CONTROL_VARIABLES`)

```
fan | vent | irrigation | heater | led | co2 | fogger
```

#### Stage progression constants

```js
// 6 canonical stages:
stages: ['Seedling', 'Early Veg.', 'Flw. Init.', 'Flowering', 'Unripe', 'Ripe']
// Note: HomeDashboard uses abbreviated labels for display space
// ManualOverride uses full names; align API response to full names

// Stage durations (days):
Seedling: 14 | Early Vegetative: 21 | Flowering Initiation: 15 | Flowering: 15 | Unripe: 14 | Ripe: 14
```

---

### 7.3 DetailedInsights

**File:** `src/pages/DetailedInsights.jsx`  
**Route key:** `'insights'`

**Purpose:** In-depth analytical view. Full sensor readings, disease risk scores per pathogen, outdoor weather detail, image galleries, and growth intelligence spotlight.

#### Layout

```
┌──────────────────────────────────────────────────────────┐
│  Section: Key Greenhouse Metrics (8 metric chips)        │
│  Section: Actuator Status (7 pill chips)                 │
│  Section: Growth Intelligence (stage track + spotlight)  │
├───────────────────────┬──────────────────────────────────┤
│  Disease Risk Panel   │  Indoor Sensor Detail            │
│  (5 diseases, bars)   │  (9 sensors, optimal ranges)     │
├───────────────────────┴──────────────────────────────────┤
│  Outdoor Weather (current + 6-slot forecast)             │
├───────────────────────┬──────────────────────────────────┤
│  Stage Camera Gallery │  Leaf Scan Gallery               │
│  (5 recent frames)    │  (5 recent frames + risk label)  │
└───────────────────────┴──────────────────────────────────┘
```

#### Mock data constants (ALL defined inline in the file)

| Constant | What it holds | Target API |
|----------|--------------|----------|
| `CROP` | Same structure as HomeDashboard `CROP`. `stages[]` uses abbreviated names. `currentPct: 80`, `daysInStage: 12`, `stageDuration: 15`. | `GET /api/dt/state` → `.crop` |
| `GROWTH_INTEL` | `hoursToNextStage: 47`, `transitionProb24h: 18`, full `stageHistory[]` with complete stage names. Drives the spotlight stats and history bars. | `GET /api/intelligence/growth` |
| `CROP_HEALTH` | Same structure, `status: 'Healthy'`, `scannedAgo: '2 mins ago'`. | `GET /api/dt/state` → `.health` |
| `ACTUATORS` | 7 actuators with `value` field (e.g., `'75%'`, `'45 L'`, `'0%'`). Shown as pill chips with level values. | `GET /api/actuators/state` |
| `SUMMARY_METRICS` | 8 `MetricChip` entries: Indoor Temp, Humidity, Soil Moisture, CO₂, Light Intensity, VPD, Leaf Wetness, Disease Risk. All from MPC `STATE_VARIABLES`. | `GET /api/sensors/summary` or `GET /api/dt/state` → `.sensors` |
| `DISEASE_RISKS` | 5 disease entries: `{ name, pathogen, risk24h (0–100), severity ('Low'/'Medium'/'High') }`. | `GET /api/intelligence/disease` |
| `OUTDOOR_CURRENT` | Nested: `temp.now`, `humidity.now`, `windspeed.now`, `solarradiation.now`, `conditions.now`, each with `.forecast` counterpart for 24h prediction. | `GET /api/weather/current` → `.current` and `.forecast` |
| `INDOOR_SENSORS` | 9 sensors with full detail: `value`, `unit`, `status`, `rangeMin`, `rangeMax`, `optimal` string (shown as a range label). | `GET /api/sensors/detail` |
| `RECENT_CROP_IMAGES` | Array of 5: `{ src, alt, capturedAgo, stage, confidence }`. Rolling 30-min cadence images from MinIO. | `GET /api/media/stage-images` |
| `RECENT_LEAF_IMAGES` | Array of 5: `{ src, alt, capturedAgo, classification, risk, confidence }`. `classification` from disease classifier. | `GET /api/media/disease-scans` |

#### Disease list (canonical, from `constants.py DISEASE_CATEGORIES`)

```
Late Blight (P. infestans)
Early Blight (A. solani)
Powdery Mildew (L. taurica)
Spider Mites (T. urticae)
Leaf Mold (P. fulva)
```

#### Indoor sensor list (from MPC `STATE_VARIABLES`)

| Label | Unit | Optimal range |
|-------|------|--------------|
| Indoor Temp | °C | 22 – 26 |
| Humidity | % | 60 – 70 |
| Soil Moisture | % | 50 – 75 |
| CO₂ | ppm | 800 – 1000 |
| Light Intensity | lux | 10,000 – 15,000 |
| VPD | kPa | 0.8 – 1.2 |
| Dew Point | °C | 14 – 18 |
| Leaf Wetness | proxy | 0.0 – 0.3 |
| Disease Risk | score | < 0.3 |

#### Sub-components (file-local)

| Component | Description |
|-----------|-------------|
| `SectionHeader` | Section heading with icon badge, eyebrow text, title, and optional subtitle. |
| `MetricChip` | Small stat tile: icon + label + large value + unit. Used in the 8-chip summary row. |
| `ActuatorChip` | Pill chip: icon + label + level value + ON/OFF status. Used in actuator row. |
| `CropStageTrack` | Same as HomeDashboard version (duplicated for independence). |
| `SpotlightStat` | Large spotlight number (hours-to-next / transition %). Two sizes: primary (large) and secondary. |
| `DiseaseBar` | Disease risk row: name + pathogen + percentage bar + severity pill. |
| `SensorRow` | Indoor sensor detail row: icon + label + value + unit + optimal range + status dot. |
| `OutdoorCell` | Weather detail cell: icon + metric label + now value + forecast value. |
| `ForecastSlot` | Hourly forecast column in the weather strip. |
| `ImageFrame` | Camera/scan image frame with overlay badges (time, stage, confidence). |

#### RISK_THEME map (drives disease bar colours)

```js
Low    → bg-primary / text-primary
Medium → bg-warning / text-warning
High   → bg-danger  / text-danger
```

---

### 7.4 ManualOverride

**File:** `src/pages/ManualOverride.jsx`  
**Route key:** `'override'`

**Purpose:** Manual simulation control panel. Operators switch between "Live / Current" mode (read-only display of live state) and "Manual Override" mode (editable inputs). In override mode, all parameters are editable and changes are sent to the DT loop.

#### Layout

```
┌──────────────────────────────────────────────────────────┐
│  Mode pill: [Live / Current]  [Manual Override]          │
│  Unsaved-changes banner (appears when override is dirty) │
├─────────────────────────┬────────────────────────────────┤
│  SIMULATION PARAMETERS  │  ACTUATOR CONTROL              │
│  Growth Stage (select)  │  7 actuator toggle cards       │
│  Day in Stage (number)  │                                │
│  Start Date (text)      │                                │
│  Start Hour (grid pick) │                                │
│  Apply button           │  Reset all → OFF button        │
└─────────────────────────┴────────────────────────────────┘
```

#### Mock data / constants (all file-local)

| Constant | Description | Target API |
|----------|-------------|----------|
| `GROWTH_STAGES` | `['Seedling', 'Early Vegetative', 'Flowering Initiation', 'Flowering', 'Unripe', 'Ripe']` — full names, 6 canonical stages. | `GET /api/dt/state` → `.crop.stage` (to populate current selection) |
| `STAGE_MAX_DAYS` | `{ 'Seedling': 14, 'Early Vegetative': 21, 'Flowering Initiation': 15, 'Flowering': 15, 'Unripe': 14, 'Ripe': 14 }` — clamps the day-in-stage number input. | static — from `TOMATO_GROWTH_STAGE_CLASSIFICATION.md` |
| `ACTUATOR_DEFS` | Static definition of 7 actuators: `{ id, icon, label, accent: { text, bg, ring } }`. Layout and styling only — not the state. | static |
| `INITIAL_ACTUATORS` | Initial state per actuator: `{ active: bool, level: number }`. Mirrors live readings from DetailedInsights. | `GET /api/actuators/state` (to seed initial state on load) |

#### Component state (React `useState`)

| State var | Initial value | Description |
|-----------|--------------|-------------|
| `mode` | `'live'` | `'live'` = read-only / `'override'` = editable |
| `stage` | `'Flowering'` | Selected growth stage in override form |
| `dayInStage` | `12` | Day number input |
| `startDate` | `getLiveDate()` | ISO date string `YYYY-MM-DD` |
| `startHour` | `getLiveHour()` | 0–23 hour integer |
| `actuators` | `INITIAL_ACTUATORS` | Map of `{ [id]: { active, level } }` |
| `saved` | `false` | Drives the "Applied" success banner |
| `confirmReset` | `false` | Drives the "Confirm → Reset all OFF" two-step |

#### Override interactions

| Action | What happens | Target API |
|--------|-------------|----------|
| Switching to "Manual Override" mode | All inputs become enabled | — |
| Changing growth stage | `dayInStage` max clamps to `STAGE_MAX_DAYS[stage]` | — |
| Toggling an actuator | `actuators[id].active` flips | — |
| Clicking "Apply Changes" | Shows 3s "Applied ✓" banner | `POST /api/dt/override` (send `{ stage, dayInStage, startDate, startHour }`) + `POST /api/actuators/set` (send actuator states) |
| Clicking "Reset" → "Confirm Reset" | All actuators set `active: false, level: 0` | `POST /api/actuators/set` with all OFF |
| Switching back to "Live / Current" | Inputs disabled (read-only display continues) | — |

#### Sub-components (file-local)

| Component | Description |
|-----------|-------------|
| `Toggle` | Pill-style on/off toggle switch. Props: `active`, `onToggle`, `disabled`. |
| `ModePill` | Two-segment pill selector: Live / Override. Active segment changes colour (primary for live, warning for override). |
| `Field` | Form field wrapper: label (uppercase, small) + hint text below. |
| `Select` | Styled `<select>` dropdown with chevron icon overlay. |
| `NumberInput` | Number `<input>` with optional `suffix` label. Clamps to `min`/`max`. |
| `HourPicker` | 24-button grid (0–23) for selecting the simulation start hour. |
| `ActuatorCard` | Actuator control tile: icon + label + active status dot + `Toggle`. No slider — toggle only. |

#### Actuator IDs and defaults

```js
fan:         { active: true,  level: 75 }   // Fan Speed
vent:        { active: true,  level: 45 }   // Vent Opening
irrigation:  { active: true,  level: 45 }   // Irrigation
heater:      { active: true,  level: 60 }   // Heater
led:         { active: false, level: 0  }   // LED Intensity
co2:         { active: true,  level: 55 }   // CO₂ Valve
fogger:      { active: false, level: 0  }   // Fogger
```

---

## 8. Shared Components

### `StatusCard` — `src/components/ui/StatusCard.jsx`

Used in HomeDashboard for Crop Health and Weather Status tiles.

```jsx
<StatusCard
  label="Crop Health"
  status="Healthy"       // 'Healthy' | 'Warning' | 'Risk'
  condition="..."        // optional short line
  detail="..."           // one-sentence description
  meta="2 mins ago"      // footer text (scanned time, forecast, etc.)
  variant="crop"         // 'crop' | 'weather' — picks domain icons
/>
```

Status drives the entire card appearance: background colour, border colour, icon, and pulse dot. All classes are literal strings in the `THEME` map inside the component.

---

### `PanelCard` — `src/components/ui/PanelCard.jsx`

Standard content card used throughout HomeDashboard and DetailedInsights.

---

### `GlassPanel` — `src/components/ui/GlassPanel.jsx`

Frosted-glass card container — primarily used for modal-like panel overlays.

---

## 9. API Integration Checklist

When the FastAPI backend is running, use this checklist page by page.

### Global setup
- [ ] Create `src/agritwin_gh/frontend/.env.local` with `VITE_API_BASE_URL=http://your-host:8000`
- [ ] Verify CORS is enabled in the FastAPI app for the frontend origin

### `src/services/api.js`
For each function listed in §5:
- [ ] Uncomment the `const data = await get(...)` line
- [ ] Delete the `return Promise.resolve(mock.XXX)` fallback
- [ ] Add error handling (`try/catch`, user-facing toast/banner)
- [ ] Remove the top-level `import * as mock from '../data/mockData.js'` once all functions are wired

### `HomeDashboard.jsx`
Replace each inline constant with a `useEffect` + API call:
- [ ] `CROP` → `getDtState().then(d => setCrop(d.crop))`
- [ ] `CROP_HEALTH` → `getDtState().then(d => setHealth(d.health))`
- [ ] `WEATHER` → `getWeather().then(d => setWeather(d.status))`
- [ ] `ACTUATORS` → `getActuatorState().then(setActuators)` — note: API returns 7 actuators, map `id` field to find icon from local `ACTUATOR_DEF` lookup
- [ ] `MONTHLY_RESOURCES` + `MONTHLY_COST` → `getMonthlyResources().then(d => { setResources(d.resources); setCost(d.cost) })`
- [ ] `CROP_IMAGES` → `getLatestMedia().then(setImages)` — API returns MinIO pre-signed URLs in the `src` field
- [ ] `GROWTH_INTEL.stageHistory` → `getGrowthIntel().then(d => setGrowthIntel(d))`
- [ ] `METRICS` (Temp/Humidity/CO₂/Light) → `getDtState().then(d => setMetrics(d.sensors))`
- [ ] `HEALTH_ROWS` → `getSystemHealth().then(setHealthRows)`

### `DetailedInsights.jsx`
- [ ] `CROP` → `getDtState().then(d => setCrop(d.crop))`
- [ ] `GROWTH_INTEL` → `getGrowthIntel().then(setGrowthIntel)`
- [ ] `CROP_HEALTH` → `getDtState().then(d => setHealth(d.health))`
- [ ] `ACTUATORS` → `getActuatorState().then(setActuators)` — API must include `value` field (e.g., `'75%'`, `'45 L'`)
- [ ] `SUMMARY_METRICS` → `getDtState().then(d => setMetrics(d.sensors))` — 8 STATE_VARIABLES
- [ ] `DISEASE_RISKS` → `getDiseaseRisks().then(setDiseaseRisks)`
- [ ] `OUTDOOR_CURRENT` → `getWeather().then(d => setOutdoor(d.current))`
- [ ] `INDOOR_SENSORS` → fetch from `GET /api/sensors/detail` (currently no dedicated stub function — add to `api.js`)
- [ ] `RECENT_CROP_IMAGES` → `getStageImages().then(setStageImages)` — API returns rolling 5 entries
- [ ] `RECENT_LEAF_IMAGES` → `getDiseaseImages().then(setLeafImages)` — API returns rolling 5 entries with `classification` and `risk`

### `ManualOverride.jsx`
- [ ] On mount: `getActuatorState()` → seed `INITIAL_ACTUATORS` state from live values
- [ ] On mount: `getDtState()` → populate `stage`, `dayInStage` default to live values
- [ ] "Apply Changes" button: `postDtOverride({ stage, dayInStage, startDate, startHour })` then `postActuatorSet(actuatorPayload)`
- [ ] "Reset" confirmation: `postActuatorSet(allOffPayload)` where every actuator has `active: false, level: 0`

---

## 10. Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `VITE_API_BASE_URL` | No | `http://localhost:8000` | FastAPI backend base URL. Set in `.env.local`. Never commit to version control. |

Create `.env.local` in `src/agritwin_gh/frontend/`:
```
VITE_API_BASE_URL=http://localhost:8000
```

Vite exposes only variables prefixed with `VITE_` to the browser bundle. Do not put secrets here.

---

## 11. Build & Dev Commands

All commands run from `src/agritwin_gh/frontend/`:

```powershell
npm run dev        # Vite dev server with HMR → http://localhost:5173
npm run build      # Production build → dist/ (serves as static assets)
npm run preview    # Serve dist/ locally for production verification
npm run lint       # ESLint (React + React Hooks + React Refresh rules)
```

### Serving the built frontend from FastAPI

After `npm run build`, the `dist/` directory contains the SPA. Serve it as static files from FastAPI:

```python
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app.mount("/", StaticFiles(directory="src/agritwin_gh/frontend/dist", html=True), name="frontend")
```

Or use a reverse proxy (nginx, Caddy) pointing to `dist/` in production.

---

*This document covers the frontend as of the build on **April 4, 2026**. All 1639 Vite-transformed modules build cleanly with no errors or warnings.*
