import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import joblib
import os
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

MODEL_PATH = 'bitcoin_model.json'
SCALER_PATH = 'scaler.save'
DATA_CACHE_PATH = 'bitcoin_data.pkl'
SEQ_LENGTH = 10

def fetch_bitcoin_data(force_refresh=False):
    try:
        # Load from local CSV if not forcing a refresh
        if not force_refresh and os.path.exists('bitcoin_data.csv'):
            print("Loading from current local dataset (bitcoin_data.csv)...")
            data = pd.read_csv('bitcoin_data.csv')
            data['date'] = pd.to_datetime(data['date']).dt.date
            return data
        
        print("Fetching fresh data from Yahoo Finance...")
        btc = yf.Ticker("BTC-USD")
        end_date = datetime.now()
        start_date = "2014-01-01"
        data = btc.history(start=start_date, end=end_date).reset_index()
        
        # Standardize columns
        data.columns = [c.lower() for c in data.columns]
        if 'date' not in data.columns:
            data['date'] = data.index
            
        data['date'] = pd.to_datetime(data['date']).dt.date
        
        # Match Yahoo Finance columns exactly
        if 'adj close' in data.columns:
            data['adj_close'] = data['adj close']
        else:
            data['adj_close'] = data['close']
            
        # Ensure volume is numeric and drop unnecessary columns
        data['volume'] = pd.to_numeric(data['volume'], errors='coerce').fillna(0)
        cols_to_drop = ['dividends', 'stock splits', 'adj close']
        data = data.drop(columns=[c for c in cols_to_drop if c in data.columns])

        # Comprehensive Feature Engineering ("Fetch Everything")
        data['returns'] = data['close'].pct_change()
        data['ma7'] = data['close'].rolling(window=7).mean()
        data['ma14'] = data['close'].rolling(window=14).mean()
        data['ma30'] = data['close'].rolling(window=30).mean()
        data['ema7'] = data['close'].ewm(span=7, adjust=False).mean()
        data['ema30'] = data['close'].ewm(span=30, adjust=False).mean()
        data['daily_range'] = data['high'] - data['low']
        data['volatility'] = data['returns'].rolling(window=30).std()
        
        data.dropna(inplace=True)
        
        # Save to disk
        data.to_pickle(DATA_CACHE_PATH)
        data.to_csv('bitcoin_data.csv', index=False)
        
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def get_live_price():
    try:
        # Use yfinance for reliable live price
        btc = yf.Ticker("BTC-USD")
        curr = btc.fast_info['last_price']
        
        # Fetch USD-INR for conversion
        ticker_inr = yf.Ticker("INR=X")
        inr_rate = ticker_inr.fast_info['last_price']
        
        return curr, curr * inr_rate
    except:
        return None, None

def train_xgb_model():
    data = fetch_bitcoin_data()
    if data is None or data.empty:
        return {"error": "No data available"}
    
    # Selection of Features (Current Dataset Columns)
    features = ['open', 'high', 'low', 'close', 'volume', 'ema7', 'volatility']
    # Ensure columns exist
    features = [f for f in features if f in data.columns]
    
    # Prepare data for training
    df_features = data[features].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_features)
    
    # Save scaler
    joblib.dump(scaler, SCALER_PATH)

    # Train/Test Split
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]
    
    def create_sequences(dataset, seq_length):
        X, y = [], []
        target_idx = features.index('close')
        for i in range(len(dataset) - seq_length):
            X.append(dataset[i:i + seq_length])
            y.append(dataset[i + seq_length, target_idx])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_data, SEQ_LENGTH)
    X_test, y_test = create_sequences(test_data, SEQ_LENGTH)

    # Flatten X for XGBoost
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    model = XGBRegressor(
        n_estimators=2000, 
        learning_rate=0.01, 
        max_depth=8, 
        subsample=0.8, 
        colsample_bytree=0.8, 
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train_flat, y_train)
    
    # Save Model
    model.save_model(MODEL_PATH)
    
    # Evaluate
    test_preds = model.predict(X_test_flat)
    
    target_idx = features.index('close')
    def inv_transform_single_col(values, col_idx, total_cols):
        dummy = np.zeros((len(values), total_cols))
        dummy[:, col_idx] = values.flatten()
        inv = scaler.inverse_transform(dummy)
        return inv[:, col_idx]

    test_preds_inv = inv_transform_single_col(test_preds, target_idx, len(features))
    y_test_inv = inv_transform_single_col(y_test, target_idx, len(features))
    
    mse = mean_squared_error(y_test_inv, test_preds_inv)
    rmse = np.sqrt(mse)
    
    return {
        "mse": mse, 
        "rmse": rmse, 
        "y_test": y_test_inv.flatten().tolist(),
        "predictions": test_preds_inv.flatten().tolist(),
        "dates": data['date'].iloc[-len(y_test_inv):].astype(str).tolist()
    }

def predict_next_days(days=10):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None
        
    data = fetch_bitcoin_data()
    if data is None:
        return None
        
    # Features used for the model
    features = ['open', 'high', 'low', 'close', 'volume', 'ema7', 'volatility']
    features = [f for f in features if f in data.columns]
    
    model = XGBRegressor()
    model.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # Get last sequence
    last_data = data[features].tail(SEQ_LENGTH).values
    scaled_last = scaler.transform(last_data)
    
    predictions = []
    current_batch = scaled_last.reshape(1, SEQ_LENGTH, len(features))
    
    target_idx = features.index('close')
    try:
        open_idx = features.index('open')
        high_idx = features.index('high')
        low_idx = features.index('low')
        ema_idx = features.index('ema7') if 'ema7' in features else -1
    except:
        open_idx = high_idx = low_idx = ema_idx = -1
    
    for _ in range(days):
        # Flatten for prediction
        curr_flat = current_batch.reshape(1, -1)
        pred = model.predict(curr_flat)[0]
        
        # Create next row based on prediction
        new_row = current_batch[0, -1, :].copy()
        
        # Smarter recursive update:
        if open_idx != -1: new_row[open_idx] = new_row[target_idx] # Next Open = current Close
        new_row[target_idx] = pred 
        if high_idx != -1: new_row[high_idx] = max(pred, new_row[open_idx]) * 1.002 # Estimate
        if low_idx != -1: new_row[low_idx] = min(pred, new_row[open_idx]) * 0.998 # Estimate
        
        # Simple EMA update (approximate)
        if ema_idx != -1:
            alpha = 2 / (7 + 1)
            new_row[ema_idx] = (pred * alpha) + (new_row[ema_idx] * (1 - alpha))
        
        predictions.append(pred)
        
        # Update current batch
        new_row_reshaped = new_row.reshape(1, 1, len(features))
        current_batch = np.append(current_batch[:, 1:, :], new_row_reshaped, axis=1)

    # Inverse transform predictions
    def inv_transform_single_col(values, col_idx, total_cols):
        dummy = np.zeros((len(values), total_cols))
        dummy[:, col_idx] = values.flatten()
        inv = scaler.inverse_transform(dummy)
        return inv[:, col_idx]

    prices = inv_transform_single_col(np.array(predictions).reshape(-1, 1), target_idx, len(features))
    
    # Generate future dates
    last_date = data['date'].iloc[-1]
    if isinstance(last_date, str):
        last_date = datetime.strptime(last_date, '%Y-%m-%d').date()
        
    dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(days)]
    
    current_price = float(data['close'].iloc[-1])
    next_day_price = float(prices[0])
    
    # CoinCodex style: Pass last 30 days history + predictions for plotting
    history_last_30 = data[['date', 'close']].tail(30)
    history_dates = history_last_30['date'].astype(str).tolist()
    history_prices = history_last_30['close'].tolist()

    return {
        "dates": dates,
        "prices": [round(p, 2) for p in prices.tolist()],
        "history_dates": history_dates,
        "history_prices": [round(p, 2) for p in history_prices],
        "current_price": round(current_price, 2),
        "next_day_price": round(next_day_price, 2),
        "price_diff": round(next_day_price - current_price, 2),
        "diff_percent": round(((next_day_price - current_price) / current_price) * 100, 2)
    }

def get_seasonality_data():
    data = fetch_bitcoin_data()
    if data is None:
        return None
    
    df = data.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # Calculate monthly returns: (Last Close of month / Last Close of prev month) - 1
    # For simplicity, we can use pct_change on daily and then resample or just pivot.
    df.set_index('date', inplace=True)
    monthly_prices = df['close'].resample('ME').last()
    monthly_returns = monthly_prices.pct_change() * 100
    
    res = []
    for date, ret in monthly_returns.items():
        if pd.isna(ret): continue
        res.append({
            'year': date.year,
            'month': date.strftime('%b'),
            'month_idx': date.month,
            'return': round(ret, 2)
        })
    
    # Pivot for heatmap
    pivot_df = pd.DataFrame(res).pivot(index='year', columns='month_idx', values='return')
    pivot_df.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pivot_df = pivot_df.sort_index(ascending=False) # Recent years on top
    
    return pivot_df.fillna(0).to_dict(orient='index')
