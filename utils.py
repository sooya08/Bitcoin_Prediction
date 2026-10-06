
"""
Bitcoin Predictor - utilities and forecasting model

Required packages:
    pip install flask pandas numpy yfinance scikit-learn xgboost joblib
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
import yfinance as yf

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "bitcoin_data.csv")
RETURN_MODEL_PATH = os.path.join(
    BASE_DIR, "bitcoin_return_model.joblib"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# DATA FETCHING
# ============================================================

def fetch_bitcoin_data(force_refresh=False):
    """
    Load historical BTC-USD daily data.
    Uses a local CSV cache and downloads from Yahoo Finance
    when needed. Returns a DataFrame with standard column names.
    """
    if not force_refresh and os.path.exists(DATA_PATH):
        try:
            data = pd.read_csv(DATA_PATH)

            if {"date", "close"}.issubset(data.columns):
                data["date"] = pd.to_datetime(
                    data["date"], errors="coerce", utc=True
                ).dt.tz_convert(None)

                data["close"] = pd.to_numeric(
                    data["close"], errors="coerce"
                )

                data = data.dropna(subset=["date", "close"])
                data = data[data["close"] > 0]
                data = data.sort_values("date")
                data = data.drop_duplicates("date")

                # Refresh old or small datasets.
                if len(data) >= 100:
                    latest = data["date"].max()
                    age = (
                        pd.Timestamp.now().normalize()
                        - latest.normalize()
                    ).days

                    if age <= 3:
                        return data.reset_index(drop=True)

        except Exception as exc:
            logger.warning("Could not read cached data: %s", exc)

    try:
        logger.info("Downloading BTC-USD historical data...")

        downloaded = yf.download(
            "BTC-USD",
            period="max",
            interval="1d",
            auto_adjust=False,
            progress=False
        )

        if downloaded is None or downloaded.empty:
            logger.error("Yahoo Finance returned no Bitcoin data.")
            return _load_cached_data()

        # Newer yfinance versions can return MultiIndex columns.
        if isinstance(downloaded.columns, pd.MultiIndex):
            downloaded.columns = downloaded.columns.get_level_values(0)

        downloaded = downloaded.reset_index()

        # Normalize Yahoo Finance column names.
        downloaded.columns = [
            str(column).strip().lower().replace(" ", "_")
            for column in downloaded.columns
        ]

        rename = {
            "datetime": "date",
            "adj_close": "adj_close"
        }

        downloaded = downloaded.rename(columns=rename)

        if "date" not in downloaded.columns:
            if "index" in downloaded.columns:
                downloaded = downloaded.rename(
                    columns={"index": "date"}
                )
            else:
                raise ValueError("Downloaded data has no date column.")

        if "close" not in downloaded.columns:
            raise ValueError("Downloaded data has no close column.")

        if "adj_close" not in downloaded.columns:
            downloaded["adj_close"] = downloaded["close"]

        if "volume" not in downloaded.columns:
            downloaded["volume"] = np.nan

        for column in ["open", "high", "low", "close",
                       "adj_close", "volume"]:
            if column not in downloaded.columns:
                downloaded[column] = np.nan

        downloaded["date"] = pd.to_datetime(
            downloaded["date"], errors="coerce", utc=True
        ).dt.tz_convert(None)

        numeric_columns = [
            "open", "high", "low", "close", "adj_close", "volume"
        ]

        for column in numeric_columns:
            downloaded[column] = pd.to_numeric(
                downloaded[column], errors="coerce"
            )

        downloaded = downloaded.dropna(subset=["date", "close"])
        downloaded = downloaded[downloaded["close"] > 0]
        downloaded = downloaded.sort_values("date")
        downloaded = downloaded.drop_duplicates("date")
        downloaded = downloaded[
            ["date", "open", "high", "low",
             "close", "adj_close", "volume"]
        ].reset_index(drop=True)

        if downloaded.empty:
            logger.error("No valid Bitcoin rows were downloaded.")
            return _load_cached_data()

        downloaded.to_csv(DATA_PATH, index=False)
        logger.info("Saved %d daily Bitcoin records.", len(downloaded))

        return downloaded

    except Exception as exc:
        logger.exception("Bitcoin download failed: %s", exc)
        return _load_cached_data()


def _load_cached_data():
    """Load available cached data when a download fails."""
    if not os.path.exists(DATA_PATH):
        return None

    try:
        data = pd.read_csv(DATA_PATH)
        data["date"] = pd.to_datetime(
            data["date"], errors="coerce", utc=True
        ).dt.tz_convert(None)

        data["close"] = pd.to_numeric(
            data["close"], errors="coerce"
        )

        data = data.dropna(subset=["date", "close"])
        data = data[data["close"] > 0]
        return data.sort_values("date").reset_index(drop=True)

    except Exception as exc:
        logger.error("Could not load cached data: %s", exc)
        return None


# ============================================================
# LIVE PRICE AND USD/INR EXCHANGE RATE
# ============================================================

def get_live_price():
    """
    Returns (bitcoin_price_usd, bitcoin_price_inr).
    Falls back to the latest cached BTC price if needed.
    """
    try:
        ticker = yf.Ticker("BTC-USD")
        history = ticker.history(period="5d", interval="1d")

        if not history.empty:
            usd = float(history["Close"].dropna().iloc[-1])
            rate = get_usd_inr_rate()
            return usd, usd * rate

    except Exception as exc:
        logger.warning("Live Bitcoin price unavailable: %s", exc)

    data = fetch_bitcoin_data()

    if data is not None and not data.empty:
        usd = float(data["close"].iloc[-1])
        return usd, usd * get_usd_inr_rate()

    return None, None


def get_usd_inr_rate():
    """Get the USD-to-INR exchange rate, with a fallback value."""
    try:
        ticker = yf.Ticker("USDINR=X")
        history = ticker.history(period="5d", interval="1d")

        if not history.empty:
            rate = float(history["Close"].dropna().iloc[-1])

            if np.isfinite(rate) and rate > 0:
                return rate

    except Exception as exc:
        logger.warning("Could not retrieve USD/INR rate: %s", exc)

    # Fallback only; update if a live rate cannot be retrieved.
    return 85.0


# ============================================================
# EXPLORATORY DATA ANALYSIS / SEASONALITY
# ============================================================

def get_seasonality_data():
    """
    Calculate average monthly Bitcoin returns by calendar year.
    Returns a dictionary keyed by year, suitable for JSON output.
    """
    data = fetch_bitcoin_data()

    if data is None or data.empty:
        return {}

    try:
        df = data[["date", "close"]].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna().sort_values("date")

        df = df.set_index("date")

        # Month-end prices and month-over-month percentage returns.
        monthly_close = df["close"].resample("ME").last()
        monthly_returns = monthly_close.pct_change() * 100

        monthly = monthly_returns.dropna().to_frame("return")
        monthly["year"] = monthly.index.year
        monthly["month_idx"] = monthly.index.month

        pivot = monthly.pivot(
            index="year",
            columns="month_idx",
            values="return"
        )

        pivot = pivot.reindex(columns=range(1, 13))
        pivot.columns = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
        ]

        pivot = pivot.sort_index(ascending=False)

        # Missing months remain null instead of being presented as zero.
        return {
            str(year): {
                month: (
                    None if pd.isna(value) else round(float(value), 2)
                )
                for month, value in row.items()
            }
            for year, row in pivot.iterrows()
        }

    except Exception as exc:
        logger.exception("Seasonality calculation failed: %s", exc)
        return {}


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def _make_return_features(close):
    """
    Generate features from historical daily closing prices.
    The target is the following day's log return.
    """
    close = pd.Series(
        np.asarray(close, dtype=float)
    ).reset_index(drop=True)

    returns = np.log(close / close.shift(1))
    features = pd.DataFrame(index=close.index)

    for lag in [1, 2, 3, 5, 7, 14, 21, 30]:
        features[f"return_lag_{lag}"] = returns.shift(lag - 1)

    for window in [3, 7, 14, 30, 60]:
        features[f"momentum_{window}"] = (
            close / close.shift(window) - 1
        )

    for window in [7, 14, 30]:
        features[f"volatility_{window}"] = (
            returns.rolling(window).std()
        )

    for window in [7, 14, 30, 60]:
        average = close.rolling(window).mean()
        features[f"ma_ratio_{window}"] = close / average - 1

    target = np.log(close.shift(-1) / close)

    result = features.replace([np.inf, -np.inf], np.nan)
    result["target"] = target
    result["current_close"] = close

    return result.dropna().copy()


# ============================================================
# TRAIN AND EVALUATE MODEL
# ============================================================

def train_xgb_model():
    """
    Chronological holdout evaluation.
    Returns y_test and predictions for compatibility with app.py.
    """
    data = fetch_bitcoin_data()

    if data is None or data.empty or "close" not in data.columns:
        return {"error": "No Bitcoin closing-price data is available."}

    data = data.copy()
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data = data.dropna(subset=["close"])
    data = data[data["close"] > 0]
    data = data.sort_values("date").reset_index(drop=True)

    if len(data) < 150:
        return {
            "error": "At least 150 daily closing prices are required."
        }

    dataset = _make_return_features(data["close"])

    if len(dataset) < 60:
        return {"error": "Not enough rows to train the model."}

    feature_columns = [
        column for column in dataset.columns
        if column not in ["target", "current_close"]
    ]

    X = dataset[feature_columns]
    y = dataset["target"]

    split = int(len(dataset) * 0.80)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    if len(X_train) < 30 or X_test.empty:
        return {"error": "Not enough training or test rows."}

    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=5.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    try:
        model.fit(X_train, y_train)

        predicted_returns = model.predict(X_test)

        current_prices = (
            dataset["current_close"].iloc[split:].to_numpy()
        )

        actual_prices = current_prices * np.exp(y_test.to_numpy())
        predicted_prices = current_prices * np.exp(predicted_returns)

        rmse = float(np.sqrt(
            mean_squared_error(actual_prices, predicted_prices)
        ))

        mae = float(
            mean_absolute_error(actual_prices, predicted_prices)
        )

        # Naive baseline: next day's price equals today's price.
        baseline_rmse = float(np.sqrt(
            mean_squared_error(actual_prices, current_prices)
        ))

        # Save the new model separately from the old model format.
        joblib.dump(
            {
                "model": model,
                "features": feature_columns
            },
            RETURN_MODEL_PATH
        )

        logger.info("Test RMSE: $%.2f", rmse)
        logger.info("Test MAE: $%.2f", mae)
        logger.info("Naive baseline RMSE: $%.2f", baseline_rmse)

        return {
            "rmse": rmse,
            "mae": mae,
            "baseline_rmse": baseline_rmse,
            "y_test": actual_prices.tolist(),
            "predictions": predicted_prices.tolist()
        }

    except Exception as exc:
        logger.exception("Model training failed: %s", exc)
        return {"error": f"Model training failed: {exc}"}


# ============================================================
# FUTURE PRICE FORECAST
# ============================================================

def predict_next_days(days=30):
    """
    Recursively predict future daily prices from predicted log returns.
    Forecasts are estimates, not guaranteed market prices.
    """
    if not os.path.exists(RETURN_MODEL_PATH):
        logger.warning(
            "Return model is missing. Train the model on the Analysis page."
        )
        return None

    data = fetch_bitcoin_data()

    if data is None or data.empty or "close" not in data.columns:
        return None

    data = data.copy()
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data = data.dropna(subset=["close"])
    data = data[data["close"] > 0]
    data = data.sort_values("date").reset_index(drop=True)

    if len(data) < 100:
        logger.error("Not enough historical data for forecasting.")
        return None

    try:
        bundle = joblib.load(RETURN_MODEL_PATH)
        model = bundle["model"]
        feature_columns = bundle["features"]

        days = max(1, min(int(days), 365))

        closes = data["close"].astype(float).tolist()
        current_price = float(closes[-1])
        prices = []

        for _ in range(days):
            features = _make_return_features(closes)

            if features.empty:
                return None

            latest = features[feature_columns].iloc[[-1]]
            predicted_return = float(model.predict(latest)[0])

            # Safety guard against extreme single-day extrapolation.
            predicted_return = float(np.clip(
                predicted_return, -0.10, 0.10
            ))

            next_price = closes[-1] * np.exp(predicted_return)

            if not np.isfinite(next_price) or next_price <= 0:
                return None

            prices.append(round(float(next_price), 2))
            closes.append(float(next_price))

        last_date = pd.to_datetime(data["date"].iloc[-1])

        dates = [
            (last_date + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(1, days + 1)
        ]

        history = data.tail(30)
        next_day_price = prices[0]
        price_diff = next_day_price - current_price

        return {
            "dates": dates,
            "prices": prices,
            "history_dates": pd.to_datetime(
                history["date"]
            ).dt.strftime("%Y-%m-%d").tolist(),
            "history_prices": history["close"].astype(float).tolist(),
            "current_price": round(current_price, 2),
            "next_day_price": round(next_day_price, 2),
            "price_diff": round(price_diff, 2),
            "diff_percent": round(
                price_diff / current_price * 100, 2
            )
        }

    except Exception as exc:
        logger.exception("Forecast generation failed: %s", exc)
        return None