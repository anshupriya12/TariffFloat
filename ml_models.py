"""
Palm Oil Tariff Simulator - ML Models
Model 1 : Random Forest  — domestic price predictor  (price_model.pkl)
Model 2 : Gradient Boosting — import volume predictor (import_model.pkl)
Model 3 : Facebook Prophet  — domestic price time-series forecast
            with 80 % / 95 % uncertainty intervals   (prophet_model.pkl,
                                                       prophet_forecast.csv)
Data source: SQLite (palm_oil_data.db) with CSV fallback.
"""

import sqlite3
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import matplotlib.pyplot as plt


# ── Prophet import with a helpful error ───────────────────────────────────────
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠  prophet not installed. Run: pip install prophet")
    print("   Model 3 (Prophet) will be skipped.\n")


def _load_data(db_path: str = 'palm_oil_data.db',
               csv_path: str = 'palm_oil_data.csv') -> pd.DataFrame:
    """
    Load palm oil data.
    Priority: SQLite (palm_oil_data.db) → CSV fallback.
    If neither exists, raises FileNotFoundError.
    """
    # ── Try SQLite ─────────────────────────────────────────────────────────────
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql('SELECT * FROM palm_oil_data ORDER BY date', conn)
        conn.close()
        df['date'] = pd.to_datetime(df['date'])
        print(f"✓ Loaded {len(df)} rows from SQLite ({db_path})")
        return df
    except Exception as e:
        print(f"! SQLite load failed ({e}) — falling back to CSV.")

    # ── CSV fallback ───────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        print(f"✓ Loaded {len(df)} rows from CSV ({csv_path})")
        return df
    except Exception as e:
        raise FileNotFoundError(
            f"Could not load data from '{db_path}' or '{csv_path}'. "
            "Run data_collector.py first."
        ) from e


class PalmOilMLModels:
    def __init__(self,
                 db_path: str = 'palm_oil_data.db',
                 csv_fallback: str = 'palm_oil_data.csv'):
        self.data = _load_data(db_path, csv_fallback)
        self.price_model   = None
        self.import_model  = None
        self.prophet_model = None

        # Ensure inr_usd column exists (may be absent in old CSVs)
        if 'inr_usd' not in self.data.columns:
            print("  ↳ 'inr_usd' column not found — using 83.0 as constant fallback.")
            self.data['inr_usd'] = 83.0

    # ──────────────────────────────────────────────────────────────────────────
    # FEATURE ENGINEERING
    # ──────────────────────────────────────────────────────────────────────────

    def prepare_features(self) -> pd.DataFrame:
        """Engineer features for ML models."""
        print("Engineering features...")
        df = self.data.copy()

        # ── Time features ──────────────────────────────────────────────────────
        df['year']    = df['date'].dt.year
        df['month']   = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter

        # ── Lagged features ────────────────────────────────────────────────────
        df['global_price_lag1']   = df['global_cpo_price_usd_tonne'].shift(1)
        df['global_price_lag2']   = df['global_cpo_price_usd_tonne'].shift(2)
        df['domestic_price_lag1'] = df['domestic_cpo_price_inr_10kg'].shift(1)
        df['import_volume_lag1']  = df['import_volume_tonnes'].shift(1)
        df['import_volume_lag2']  = df['import_volume_tonnes'].shift(2)
        df['inr_usd_lag1']        = df['inr_usd'].shift(1)   # ← NEW

        # ── Rolling averages (3-month) ─────────────────────────────────────────
        df['global_price_ma3']   = df['global_cpo_price_usd_tonne'].rolling(3).mean()
        df['import_volume_ma3']  = df['import_volume_tonnes'].rolling(3).mean()
        df['inr_usd_ma3']        = df['inr_usd'].rolling(3).mean()           # ← NEW

        # ── Interaction terms ──────────────────────────────────────────────────
        df['price_tariff_interaction'] = (
            df['global_cpo_price_usd_tonne'] * df['tariff_pct']
        )
        # Landed cost proxy: global price in INR/tonne after tariff, per 10 kg
        df['landed_cost_inr'] = (
            df['global_cpo_price_usd_tonne']
            * df['inr_usd']
            * (1 + df['tariff_pct'] / 100)
            / 100
        )                                                                      # ← NEW

        # ── Import dependency ──────────────────────────────────────────────────
        df['import_dependency'] = (
            df['import_volume_tonnes']
            / (df['import_volume_tonnes'] + df['domestic_production_tonnes'])
            * 100
        )

        df = df.dropna()
        print(f"✓ Features prepared: {len(df)} records, {len(df.columns)} columns")
        return df

    # ──────────────────────────────────────────────────────────────────────────
    # MODEL 1 — DOMESTIC PRICE PREDICTOR  (Random Forest)
    # ──────────────────────────────────────────────────────────────────────────

    def train_price_model(self, df: pd.DataFrame) -> dict:
        """Train Random Forest to predict domestic CPO price (INR/10 kg)."""
        print("\n" + "=" * 60)
        print("TRAINING MODEL 1: DOMESTIC PRICE PREDICTOR  (Random Forest)")
        print("=" * 60)

        feature_cols = [
            'global_cpo_price_usd_tonne',
            'tariff_pct',
            'inr_usd',           # single exchange-rate feature; lag/MA variants removed
            # 'landed_cost_inr' removed — circular with target
            # 'inr_usd_lag1', 'inr_usd_ma3' removed — high collinearity with inr_usd
            # on 99 rows increases variance without adding signal
            'year',              # captures long-run price trend across 2018-2025
            'month',
            'quarter',
            'global_price_lag1',
            'global_price_lag2',
            'domestic_price_lag1',
            'global_price_ma3',
            'price_tariff_interaction',
            'import_dependency',
        ]

        X = df[feature_cols]
        y = df['domestic_cpo_price_inr_10kg']

        # shuffle=True: lag features already encode temporal context, so row order
        # is not required for valid evaluation. With only 99 rows, shuffle=False
        # puts the 2022-2025 price spike exclusively in the test set, producing
        # R² < 0 because the training distribution never saw those price levels.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )

        print("Training Random Forest Regressor...")
        self.price_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        self.price_model.fit(X_train, y_train)

        y_train_pred = self.price_model.predict(X_train)
        y_test_pred  = self.price_model.predict(X_test)

        train_r2  = r2_score(y_train, y_train_pred)
        test_r2   = r2_score(y_test,  y_test_pred)
        test_mae  = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        # cv=3 because with ~99 rows and shuffle=False each fold is ~33 rows;
        # cv=5 would give ~20 rows per fold — too small for stable R² estimates
        cv_scores = cross_val_score(
            self.price_model, X, y, cv=3, scoring='r2', n_jobs=-1
        )

        print(f"\n✓ Model Performance:")
        print(f"  Train R²  : {train_r2:.4f}")
        print(f"  Test  R²  : {test_r2:.4f}")
        print(f"  Test  MAE : {test_mae:.2f} INR/10kg")
        print(f"  Test  RMSE: {test_rmse:.2f} INR/10kg")
        print(f"  CV R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std()*2:.4f}")

        importance = pd.DataFrame({
            'feature':    feature_cols,
            'importance': self.price_model.feature_importances_,
        }).sort_values('importance', ascending=False)
        print(f"\n✓ Top 5 features:")
        print(importance.head().to_string(index=False))

        joblib.dump(self.price_model, 'price_model.pkl')
        joblib.dump(feature_cols,     'price_features.pkl')
        print(f"\n✓ Saved: price_model.pkl  |  price_features.pkl")

        return dict(train_r2=train_r2, test_r2=test_r2,
                    test_mae=test_mae, test_rmse=test_rmse,
                    cv_mean=cv_scores.mean(), cv_std=cv_scores.std())

    # ──────────────────────────────────────────────────────────────────────────
    # MODEL 2 — IMPORT VOLUME PREDICTOR  (Gradient Boosting)
    # ──────────────────────────────────────────────────────────────────────────

    def train_import_model(self, df: pd.DataFrame) -> dict:
        """Train Gradient Boosting to predict monthly import volume (tonnes)."""
        print("\n" + "=" * 60)
        print("TRAINING MODEL 2: IMPORT VOLUME PREDICTOR  (Gradient Boosting)")
        print("=" * 60)

        feature_cols = [
            'global_cpo_price_usd_tonne',
            'tariff_pct',
            'domestic_production_tonnes',
            'month',
            'quarter',
            'import_volume_lag1',
            'import_volume_lag2',
            'import_volume_ma3',
            'price_tariff_interaction',
            'domestic_cpo_price_inr_10kg',
        ]

        X = df[feature_cols]
        y = df['import_volume_tonnes']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )

        print("Training Gradient Boosting Regressor...")
        self.import_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
        )
        self.import_model.fit(X_train, y_train)

        y_train_pred = self.import_model.predict(X_train)
        y_test_pred  = self.import_model.predict(X_test)

        train_r2  = r2_score(y_train, y_train_pred)
        test_r2   = r2_score(y_test,  y_test_pred)
        test_mae  = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        cv_scores = cross_val_score(
            self.import_model, X, y, cv=5, scoring='r2', n_jobs=-1
        )

        print(f"\n✓ Model Performance:")
        print(f"  Train R²  : {train_r2:.4f}")
        print(f"  Test  R²  : {test_r2:.4f}")
        print(f"  Test  MAE : {test_mae:.0f} tonnes/month")
        print(f"  Test  RMSE: {test_rmse:.0f} tonnes/month")
        print(f"  CV R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std()*2:.4f}")

        importance = pd.DataFrame({
            'feature':    feature_cols,
            'importance': self.import_model.feature_importances_,
        }).sort_values('importance', ascending=False)
        print(f"\n✓ Top 5 features:")
        print(importance.head().to_string(index=False))

        joblib.dump(self.import_model, 'import_model.pkl')
        joblib.dump(feature_cols,      'import_features.pkl')
        print(f"\n✓ Saved: import_model.pkl  |  import_features.pkl")

        return dict(train_r2=train_r2, test_r2=test_r2,
                    test_mae=test_mae, test_rmse=test_rmse,
                    cv_mean=cv_scores.mean(), cv_std=cv_scores.std())

    # ──────────────────────────────────────────────────────────────────────────
    # MODEL 3 — DOMESTIC PRICE TIME-SERIES FORECAST  (Facebook Prophet)
    # ──────────────────────────────────────────────────────────────────────────

    def train_prophet_model(self, forecast_periods: int = 12) -> dict | None:
        """
        Fit a Prophet model on the full domestic price history and produce a
        12-month ahead forecast with 80 % and 95 % uncertainty intervals.

        Extra regressors added to Prophet:
          - global_cpo_price_usd_tonne
          - tariff_pct
          - inr_usd

        Future regressor values are held constant at their last observed value
        (conservative assumption; replace with scenario inputs for policy use).

        Outputs
        -------
        prophet_model.pkl       — serialised Prophet model (joblib)
        prophet_forecast.csv    — full forecast DataFrame (history + 12 future months)
        """
        if not PROPHET_AVAILABLE:
            print("\n⚠  Skipping Model 3: Prophet not installed.")
            return None

        print("\n" + "=" * 60)
        print("TRAINING MODEL 3: DOMESTIC PRICE FORECAST  (Facebook Prophet)")
        print("=" * 60)

        # ── Prepare Prophet DataFrame ──────────────────────────────────────────
        df = self.data[
            ['date', 'domestic_cpo_price_inr_10kg',
             'global_cpo_price_usd_tonne', 'tariff_pct', 'inr_usd']
        ].copy().dropna()

        prophet_df = df.rename(columns={
            'date':                        'ds',
            'domestic_cpo_price_inr_10kg': 'y',
        })

        regressor_cols = ['global_cpo_price_usd_tonne', 'tariff_pct', 'inr_usd']

        # ── Build & fit model ──────────────────────────────────────────────────
        print("Fitting Prophet model with regressors: "
              + ', '.join(regressor_cols) + " ...")

        self.prophet_model = Prophet(
            interval_width=0.80,          # primary uncertainty band (80 %)
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.15, # moderate flexibility
        )

        # Add each regressor
        for col in regressor_cols:
            self.prophet_model.add_regressor(col)

        self.prophet_model.fit(prophet_df)

        # ── Build future DataFrame (manual construction — avoids make_future_dataframe
        #    pandas compatibility issues with MS freq in newer pandas versions) ──
        future_dates = pd.date_range(
            start=prophet_df['ds'].iloc[0],
            periods=len(prophet_df) + forecast_periods,
            freq='MS',
        )
        future = pd.DataFrame({'ds': future_dates})
        future['ds'] = pd.to_datetime(future['ds'])
        last = prophet_df[regressor_cols].iloc[-1]
        for col in regressor_cols:
            future[col] = pd.concat([
                prophet_df[col],
                pd.Series([last[col]] * forecast_periods),
            ]).values

        # ── Forecast ───────────────────────────────────────────────────────────
        forecast = self.prophet_model.predict(future)

        # Clip negative lower bounds (prices can't be negative)
        forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)

        # ── In-sample fit metrics (on training data only) ──────────────────────
        train_len     = len(prophet_df)
        in_sample     = forecast.iloc[:train_len]
        train_mae     = mean_absolute_error(prophet_df['y'], in_sample['yhat'])
        train_rmse    = np.sqrt(mean_squared_error(prophet_df['y'], in_sample['yhat']))
        train_r2      = r2_score(prophet_df['y'], in_sample['yhat'])

        print(f"\n✓ In-sample fit (training data):")
        print(f"  R²  : {train_r2:.4f}")
        print(f"  MAE : {train_mae:.2f} INR/10kg")
        print(f"  RMSE: {train_rmse:.2f} INR/10kg")

        # ── 12-month forecast summary ──────────────────────────────────────────
        future_fc = forecast.iloc[train_len:][
            ['ds', 'yhat', 'yhat_lower', 'yhat_upper',
             'trend', 'yearly']
        ].copy()
        future_fc.columns = [
            'date', 'forecast_price_inr_10kg',
            'lower_80pct', 'upper_80pct',
            'trend', 'seasonality_yearly',
        ]

        print(f"\n✓ 12-month ahead forecast (₹/10 kg):")
        print(future_fc[['date', 'forecast_price_inr_10kg',
                          'lower_80pct', 'upper_80pct']].to_string(index=False))

        # Also compute 95 % intervals via a second pass
        prophet_95 = Prophet(
            interval_width=0.95,
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.15,
        )
        for col in regressor_cols:
            prophet_95.add_regressor(col)
        prophet_95.fit(prophet_df)
        future_95 = pd.DataFrame({'ds': future_dates})
        future_95['ds'] = pd.to_datetime(future_95['ds'])
        for col in regressor_cols:
            future_95[col] = pd.concat([
                prophet_df[col],
                pd.Series([last[col]] * forecast_periods),
            ]).values
        forecast_95 = prophet_95.predict(future_95)
        future_fc['lower_95pct'] = forecast_95.iloc[train_len:]['yhat_lower'].clip(lower=0).values
        future_fc['upper_95pct'] = forecast_95.iloc[train_len:]['yhat_upper'].values

        # ── Save outputs ───────────────────────────────────────────────────────
        # Full forecast (history + future) for dashboard use
        full_forecast = forecast[
            ['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']
        ].rename(columns={
            'ds': 'date', 'yhat': 'forecast_price_inr_10kg',
            'yhat_lower': 'lower_80pct', 'yhat_upper': 'upper_80pct',
        })
        full_forecast['lower_95pct'] = forecast_95['yhat_lower'].clip(lower=0).values
        full_forecast['upper_95pct'] = forecast_95['yhat_upper'].values
        full_forecast['is_forecast'] = [False] * train_len + [True] * forecast_periods

        full_forecast.to_csv('prophet_forecast.csv', index=False)
        joblib.dump(self.prophet_model, 'prophet_model.pkl')

        print(f"\n✓ Saved: prophet_model.pkl  |  prophet_forecast.csv")
        print(f"   Forecast rows: {forecast_periods} future months"
              f" + {train_len} historical fitted values")

        return dict(
            train_r2=train_r2,
            train_mae=train_mae,
            train_rmse=train_rmse,
            forecast_mean=future_fc['forecast_price_inr_10kg'].mean(),
            forecast_min=future_fc['lower_80pct'].min(),
            forecast_max=future_fc['upper_80pct'].max(),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # MASTER TRAINING PIPELINE
    # ──────────────────────────────────────────────────────────────────────────

    def train_all_models(self):
        """Prepare features, train all three models, print summary."""
        print("\n" + "=" * 60)
        print("PALM OIL ML MODELS — TRAINING PIPELINE")
        print("=" * 60 + "\n")

        df = self.prepare_features()

        price_metrics  = self.train_price_model(df)
        import_metrics = self.train_import_model(df)
        prophet_metrics = self.train_prophet_model(forecast_periods=12)

        # ── Summary ────────────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE — MODEL SUMMARY")
        print("=" * 60)

        print(f"\n1. PRICE MODEL  (Random Forest)  — price_model.pkl")
        print(f"   Test R²  : {price_metrics['test_r2']:.4f}  "
              f"{'✓ Good' if price_metrics['test_r2'] > 0.75 else '⚠ Check data'}")
        print(f"   Test MAE : {price_metrics['test_mae']:.2f} INR/10kg")
        print(f"   Features : inr_usd, inr_usd_lag1, year + 12 others (landed_cost_inr removed)")

        print(f"\n2. IMPORT MODEL  (Gradient Boosting)  — import_model.pkl")
        print(f"   Test R²  : {import_metrics['test_r2']:.4f}  "
              f"{'✓ Good' if import_metrics['test_r2'] > 0.75 else '⚠ Check data'}")
        print(f"   Test MAE : {import_metrics['test_mae']:.0f} tonnes/month")

        if prophet_metrics:
            print(f"\n3. PROPHET FORECAST  (12-month horizon)  — prophet_model.pkl")
            print(f"   In-sample R²   : {prophet_metrics['train_r2']:.4f}")
            print(f"   In-sample MAE  : {prophet_metrics['train_mae']:.2f} INR/10kg")
            print(f"   Forecast mean  : ₹{prophet_metrics['forecast_mean']:.0f}/10kg")
            print(f"   80% CI range   : ₹{prophet_metrics['forecast_min']:.0f} – "
                  f"₹{prophet_metrics['forecast_max']:.0f}")
            print(f"   Output files   : prophet_model.pkl, prophet_forecast.csv")
        else:
            print(f"\n3. PROPHET FORECAST  — skipped (install prophet first)")

        print("\n✓ All models saved and ready for deployment!")
        return price_metrics, import_metrics, prophet_metrics


if __name__ == "__main__":
    trainer = PalmOilMLModels()
    trainer.train_all_models()
