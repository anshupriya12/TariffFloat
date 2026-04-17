<div align="center">

# 🌴 TariffFloat

### AI-Powered Palm Oil Tariff Impact Simulator

*Evidence-based policy decisions for India's edible oil import security*

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.50.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://phnm4ap3n6nhkxnxgrjguk.streamlit.app)


</div>

---

## What is TariffFloat?

TariffFloat is an interactive web simulator that answers a single, consequential policy question:

> **"If India changes its crude palm oil import tariff by X percentage points — what happens to consumer prices, import volumes, and the country's forex bill?"**

It pulls live market data, runs three trained ML models, and renders the projected impact across a 3–12 month horizon — all through a Streamlit dashboard that any analyst or policymaker can operate without writing a line of code.

---

## The Problem

India imports roughly **94% of its crude palm oil (CPO)** needs, spending **~$6.5 billion annually** on a single commodity. This creates a fragile policy environment:

| Challenge | Detail |
|---|---|
| **Price volatility** | Global CPO prices (driven by Indonesian/Malaysian supply, weather, and energy markets) feed directly into Indian kitchen costs |
| **Forex drain** | Every global price spike erodes India's foreign exchange reserves with limited domestic hedge |
| **Farmer disincentive** | Low tariffs suppress domestic oilseed prices, undercutting the National Mission on Edible Oils (NMEO-OP) |
| **Policy whiplash** | India revised CPO tariffs six times between 2018 and 2025 (7.5% → 10% → 17.5% → 12.5% → 17.5% → 10%) |

TariffFloat gives stakeholders a data-driven sandbox to stress-test tariff scenarios *before* they become policy.

---

## What It Simulates

For any tariff scenario you configure, TariffFloat forecasts four outputs:

| Output | Description | Unit |
|---|---|---|
| **Domestic Price** | Retail CPO price after the tariff change | ₹ / 10 kg |
| **Import Volume** | Expected monthly imports under the new policy | K tonnes / month |
| **Forex Impact** | Cumulative foreign exchange savings or additional cost | USD millions / ₹ crore |
| **Import Dependency** | Share of total supply sourced from imports | % |

All outputs are compared against the current-tariff baseline with percentage deltas, AI-generated policy insights, and a downloadable CSV report.

---

## Pipeline & Architecture

The project runs as a linear three-stage pipeline:

```
╔══════════════════════════════════════════════════════════════╗
║  STAGE 1 — DATA COLLECTION          data_collector.py        ║
║                                                              ║
║  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  ║
║  │  FRED API   │  │ Frankfurter  │  │  CBIC / SEA /      │  ║
║  │ (CPO Price) │  │  (INR/USD)   │  │  PIB / DGCIS       │  ║
║  └──────┬──────┘  └──────┬───────┘  └─────────┬──────────┘  ║
║         └────────────────┴──────────────────────┘            ║
║                           │                                  ║
║                    ┌──────▼──────┐                           ║
║                    │  SQLite DB  │  palm_oil_data.db         ║
║                    └─────────────┘  palm_oil_data.csv        ║
╚══════════════════════════════════════════════════════════════╝
                           │
                           ▼
╔══════════════════════════════════════════════════════════════╗
║  STAGE 2 — ML TRAINING              ml_models.py             ║
║                                                              ║
║  Feature Engineering                                         ║
║  (lags, rolling means, price×tariff, import dependency)      ║
║         │                                                    ║
║         ├──► Random Forest       →  price_model.pkl          ║
║         │    (domestic price)       price_features.pkl       ║
║         │                                                    ║
║         ├──► Gradient Boosting   →  import_model.pkl         ║
║         │    (import volume)        import_features.pkl      ║
║         │                                                    ║
║         └──► Facebook Prophet    →  prophet_model.pkl        ║
║              (12-month forecast)    prophet_forecast.csv     ║
╚══════════════════════════════════════════════════════════════╝
                           │
                           ▼
╔══════════════════════════════════════════════════════════════╗
║  STAGE 3 — DASHBOARD                app.py                   ║
║                                                              ║
║  Live market state  →  Tariff controls  →  Run simulation    ║
║  Forecast charts  →  Policy insights  →  CSV export          ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Stage 1 — Data Collection

`data_collector.py` builds a monthly dataset from **January 2018 to the present day** across six dimensions. Every data type has a live primary source with graceful fallbacks:

### CPO Price (Global)

| Priority | Source | Notes |
|---|---|---|
| 1st | **FRED API** — series `PPOILUSDM` | Malaysia palm oil, USD/metric tonne. Free API key required. |
| 2nd | **IMF** Primary Commodity Prices Excel | Public download, no key needed |
| 3rd | **World Bank Pink Sheet + MPOB** | Documented monthly values 2018–2025, hardcoded |

### INR/USD Exchange Rate

| Priority | Source | Notes |
|---|---|---|
| 1st | **Frankfurter API** — full date range | Daily rates resampled to monthly mean. No key needed. |
| 2nd | Documented annual averages | 2018–2025 compiled from RBI / published sources |

### Other Data

| Data | Source |
|---|---|
| Import volumes | SEA India Annual Reports / DGCIS (Ministry of Commerce) |
| Tariff history | CBIC official notifications (8 rate changes, 2018–2025) |
| Domestic production | PIB India / NMEO-OP / Ministry of Agriculture |

### Domestic Price Derivation

The domestic retail price is not fetched — it is *calculated* from the assembled data:

```
Domestic Price (₹/10 kg)  =  Global Price (USD/tonne)
                              × INR/USD
                              × (1 + Tariff %)
                              × 1.15            ← transport + refining + distribution
                              ÷ 100
```

### Storage

Everything is merged into a single SQLite database (`palm_oil_data.db`) with a `metadata` table storing the last-updated timestamp. A CSV backup (`palm_oil_data.csv`) is written alongside for portability.

---

## Stage 2 — ML Model Training

`ml_models.py` trains three models on the collected dataset via the `PalmOilMLModels` class.

### Feature Engineering

| Category | Features |
|---|---|
| Market inputs | `global_cpo_price_usd_tonne`, `tariff_pct`, `inr_usd` |
| Time | `year`, `month`, `quarter` |
| Price lags | `global_price_lag1`, `global_price_lag2`, `domestic_price_lag1` |
| Volume lags | `import_volume_lag1`, `import_volume_lag2` |
| Rolling averages | `global_price_ma3`, `import_volume_ma3` |
| Interaction | `price_tariff_interaction` = global price × tariff |
| Structural | `import_dependency` = imports ÷ (imports + domestic production) × 100 |

### Model 1 — Domestic Price Predictor

```
Algorithm  : RandomForestRegressor
Target     : domestic_cpo_price_inr_10kg
Params     : 200 trees, max_depth=10, min_samples_leaf=2
Split      : 80/20, shuffle=True  (see P.S.)
CV         : 3-fold R²
Output     : price_model.pkl  |  price_features.pkl
```

### Model 2 — Import Volume Predictor

```
Algorithm  : GradientBoostingRegressor
Target     : import_volume_tonnes (monthly)
Params     : 100 estimators, lr=0.1, max_depth=5
Split      : 80/20, shuffle=False (temporal ordering preserved)
CV         : 5-fold R²
Output     : import_model.pkl  |  import_features.pkl
```

### Model 3 — 12-Month Price Forecast

```
Algorithm  : Facebook Prophet
Target     : domestic_cpo_price_inr_10kg time series
Seasonality: Multiplicative, yearly component
Regressors : global_cpo_price_usd_tonne, tariff_pct, inr_usd
Intervals  : 80 % and 95 % uncertainty bands
Horizon    : 12 months (future regressors held at last observed value)
Output     : prophet_model.pkl  |  prophet_forecast.csv
```

---

## Stage 3 — Streamlit Dashboard

`app.py` wires the models and live data into a four-panel interactive UI.

### Panel 1 — Current Market State
Live KPI cards populated from the SQLite database and Frankfurter API:
- Global CPO price (USD/tonne)
- Active import tariff (%)
- Monthly import volume (K tonnes)
- Live INR/USD rate with delta vs. ₹83 base

### Panel 2 — Policy Simulation Controls
- **Global CPO price input** — editable, defaults to latest data point
- **Base tariff input** — defaults to current active tariff
- **Tariff adjustment slider** — ±10 to +20 percentage points in 0.5pp steps
- **Forecast period selector** — 3 / 6 / 9 / 12 months

### Panel 3 — Simulation Results *(appears after Run Simulation)*

| Section | What You See |
|---|---|
| Impact Metrics | Four KPI cards: domestic price, import volume, forex impact, import dependency — all with % delta vs. baseline |
| Forecast Charts | Side-by-side Plotly line charts — baseline (solid) vs. policy scenario (dashed) |
| Prophet Chart | 12-month price horizon with historical fit, confidence bands, and forecast line |
| Policy Insights | Auto-generated narrative: consumer price alert, import reduction benefit, or neutral impact |

### Panel 4 — Export
One-click CSV download of the full simulation forecast, timestamped and labelled with the simulated tariff rate.

---

## Tech Stack

| Layer | Technology | Version |
|---|---|---|
| UI Framework | Streamlit | 1.50.0 |
| Visualisation | Plotly | 6.3.1 |
| ML — Ensemble | scikit-learn | 1.7.2 |
| ML — Time Series | Facebook Prophet | latest |
| Model I/O | joblib | 1.5.2 |
| Data Processing | pandas | 2.3.3 |
| Numerical | NumPy | 2.3.4 |
| Database | SQLite3 | stdlib |
| HTTP | requests | 2.32.5 |
| Config | python-dotenv | latest |
| External APIs | FRED, Frankfurter, IMF | — |
| Language | Python | 3.10+ |

---

## Project Structure

```
TariffFloat-V1/
│
├── app.py                  # Streamlit dashboard — main entry point
├── data_collector.py       # Stage 1: fetch, merge, and store all data
├── ml_models.py            # Stage 2: feature engineering + train 3 models
│
├── palm_oil_data.db        # SQLite database (generated)
├── palm_oil_data.csv       # CSV backup (generated)
│
├── price_model.pkl         # Random Forest — domestic price
├── price_features.pkl      # Feature list for price model
├── import_model.pkl        # Gradient Boosting — import volume
├── import_features.pkl     # Feature list for import model
├── prophet_model.pkl       # Facebook Prophet — 12-month forecast
├── prophet_forecast.csv    # Prophet output: history + 12 future months
│
├── requirements.txt        # Python dependencies
├── .env                    # API keys (not committed)
└── .gitignore
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- A free [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html)

### 1 — Install dependencies

```bash
git clone <repo-url>
cd TariffFloat-V1
pip install -r requirements.txt
```

> **Prophet on Windows:** `pip install prophet` should work directly.
> On Linux/macOS you may need `pystan` first — see the [Prophet install guide](https://facebook.github.io/prophet/docs/installation.html).

### 2 — Set your API key

Create a `.env` file in the project root:

```env
FRED_API_KEY=your_key_here
```

The FRED key is free and takes 30 seconds to get. Without it, the collector falls back through IMF Excel → documented World Bank values automatically.

### 3 — Collect data

```bash
python data_collector.py
```

Fetches all six data dimensions, computes domestic prices, and saves to `palm_oil_data.db`. Typical runtime: 15–30 seconds.

### 4 — Train models

```bash
python ml_models.py
```

Trains all three models and saves the `.pkl` files. Typical runtime: 1–2 minutes. Requires `prophet` to be installed for Model 3 — if not found, Models 1 and 2 still train successfully.

### 5 — Launch the app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## Usage

1. **Read the market state** — the top row shows the latest values from your database and live exchange rate
2. **Set your scenario** — adjust the global price if you want to model a different market environment, set the base tariff, and drag the slider to simulate a tariff cut or hike
3. **Pick a forecast horizon** — 3, 6, 9, or 12 months
4. **Hit Run Simulation** — results render in ~1–2 seconds
5. **Read the charts** — solid line = current tariff baseline, dashed line = your policy scenario
6. **Check the insights panel** — automated narrative flags consumer price risk or import reduction benefits
7. **Export** — download the timestamped CSV for reports or further analysis

---

## Data Coverage

| Variable | Period | Frequency | Source |
|---|---|---|---|
| Global CPO price | Jan 2018 – present | Monthly | FRED / IMF / World Bank |
| Import tariff | Jan 2018 – present | Monthly | CBIC notifications |
| India import volume | Jan 2018 – present | Monthly | SEA India / DGCIS |
| Domestic production | Jan 2018 – present | Monthly | PIB / NMEO-OP |
| INR/USD rate | Jan 2018 – present | Monthly | Frankfurter API |
| Domestic CPO price | Jan 2018 – present | Monthly | Derived (formula) |

---

## API Reference

| API | Key Required | Rate Limit | Fallback |
|---|---|---|---|
| [FRED](https://fred.stlouisfed.org) | Yes — free | 120 req/min | IMF Excel → documented values |
| [Frankfurter](https://www.frankfurter.app) | No | Generous | ₹83.0 constant |
| [IMF Commodity Prices](https://www.imf.org/external/np/res/commod/index.aspx) | No | Public download | Documented World Bank values |

---
## Deployment — Streamlit Cloud

TariffFloat is deployed on Streamlit Community Cloud.

🔗 **Live app:** https://phnm4ap3n6nhkxnxgrjguk.streamlit.app

To deploy your own instance:
1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select your fork, set main file to `app.py`
4. Add `FRED_API_KEY` in Advanced Settings → Secrets
5. Click Deploy
---

## P.S. — Key Design Decisions

**Why `shuffle=True` for the price model split?**
The dataset has only ~90 monthly rows. A strict chronological split places the entire 2022 commodity spike (CPO hit $1,589/tonne in March 2022) exclusively in the test set — a distribution the training set never saw — which produces `R² < 0`. Shuffling ensures both splits see the full price range. Lag features already encode temporal context, so row ordering is not strictly required for valid evaluation. The import model uses `shuffle=False` because import volumes have stronger serial autocorrelation where leakage matters more.

**Why is `landed_cost_inr` excluded from price model features?**
`landed_cost_inr = global_price × inr_usd × (1 + tariff%) / 100` is algebraically near-identical to the target `domestic_cpo_price_inr_10kg` (which adds only a 1.15 markup). Including it is data leakage and would produce artificially inflated R² with no real predictive value.

**Why Prophet over ARIMA or LSTM?**
Prophet handles structural changepoints (tariff policy shifts), multiplicative seasonality, missing months, and named external regressors — all with minimal hyperparameter tuning on a small dataset. LSTM would require far more data to generalise. ARIMA doesn't natively accept external regressors without ARIMAX, adding complexity for marginal gain on ~90 data points.

**Why SQLite over a remote database?**
TariffFloat runs as a single-container app with no infrastructure dependencies — both locally and on Hugging Face Spaces. SQLite provides ACID-compliant storage, metadata timestamping, and SQL queries without a database server. Swapping to PostgreSQL later is a one-line change (replace `sqlite3.connect(...)` with a SQLAlchemy engine).

**Data disclaimer**
Import volumes, domestic production, and pre-2025 CPO prices are compiled from published government and commodity reports (SEA India, DGCIS, World Bank Pink Sheet, MPOB). Intended for policy simulation and research purposes only. Do not use as primary data for financial or commercial decisions.

---

## License

MIT — see `LICENSE`.

---

<div align="center">
<i>TariffFloat &nbsp;|&nbsp; Built for evidence-based analysis of India's edible oil import security</i>
</div>
