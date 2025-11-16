# =============================================================================
# AIR QUALITY FORECASTING MODELS
# =============================================================================
# Comprehensive module containing all forecasting models and utilities
# Models: Naive, ARIMA/SARIMA, Prophet, LSTM, CNN
# Features: Evaluation metrics, feature engineering, data preparation, visualization
# =============================================================================

# Suppress TensorFlow warnings before any imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU to avoid threading issues

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from math import sqrt
import warnings
warnings.filterwarnings('ignore')

# Lazy import flag for TensorFlow - will be imported only when needed
TF_AVAILABLE = None  # Will be set to True/False on first use

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: Statsmodels not available. ARIMA models disabled.")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Warning: Prophet not available. Prophet model disabled.")

from sklearn.preprocessing import RobustScaler, MinMaxScaler


# =============================================================================
# LAZY LOADING HELPER FOR TENSORFLOW
# =============================================================================

def _ensure_tensorflow_available():
    """Lazy load TensorFlow only when needed (called by LSTM/CNN functions)"""
    global TF_AVAILABLE
    
    if TF_AVAILABLE is None:  # First time calling - try to import
        try:
            global Sequential, LSTM, Dense, Dropout, Conv1D, Flatten, Adam, EarlyStopping, tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, Flatten
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.callbacks import EarlyStopping
            import tensorflow as tf
            TF_AVAILABLE = True
            tf.get_logger().setLevel('ERROR')
            return True
        except ImportError:
            TF_AVAILABLE = False
            return False
    
    return TF_AVAILABLE is True


# =============================================================================
# SECTION 1: DATA PREPARATION & TRAIN-TEST SPLIT
# =============================================================================

def train_test_split_timeseries(ts_data, test_size=0.15):
    """
    Split time series into train and test sets (respecting temporal order).
    
    Parameters:
    -----------
    ts_data : pd.Series
        Time series data with datetime index
    test_size : float
        Proportion of data for testing (default: 0.15)
    
    Returns:
    --------
    train : pd.Series
        Training data
    test : pd.Series
        Test data
    """
    split_idx = int(len(ts_data) * (1 - test_size))
    train = ts_data.iloc[:split_idx]
    test = ts_data.iloc[split_idx:]
    
    return train, test


def prepare_data_for_deep_learning(train_data, test_data, lookback=1):
    """
    Prepare data for LSTM/CNN models with scaling.
    
    Parameters:
    -----------
    train_data : pd.Series or np.ndarray
        Training time series
    test_data : pd.Series or np.ndarray
        Test time series
    lookback : int
        Number of timesteps to look back
    
    Returns:
    --------
    dict : Dictionary with scaled data and scaler object
    """
    # Clean data
    train_clean = train_data.copy()
    test_clean = test_data.copy()
    
    if isinstance(train_clean, pd.Series):
        train_clean = train_clean.replace([np.inf, -np.inf], np.nan).dropna()
        test_clean = test_clean[~(test_clean.replace([np.inf, -np.inf], np.nan).isna())]
    
    # Robust scaling
    scaler = RobustScaler(quantile_range=(5, 95))
    scaled_train = scaler.fit_transform(train_clean.values.reshape(-1, 1)).flatten()
    scaled_test = scaler.transform(test_clean.values.reshape(-1, 1)).flatten()
    
    # Fallback to MinMaxScaler if issues
    if np.isnan(scaled_train).any() or np.isinf(scaled_train).any():
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_train = scaler.fit_transform(train_clean.values.reshape(-1, 1)).flatten()
        scaled_test = scaler.transform(test_clean.values.reshape(-1, 1)).flatten()
    
    return {
        'train': train_clean,
        'test': test_clean,
        'scaled_train': scaled_train,
        'scaled_test': scaled_test,
        'scaler': scaler
    }


# =============================================================================
# SECTION 2: FEATURE ENGINEERING FOR DEEP LEARNING
# =============================================================================

def engineer_features_from_single_point(live_value, train_data=None):
    """
    Transform single AQI observation into rich 13-feature vector.
    
    Features include:
    - Raw value (normalized)
    - Temporal: day_sin/cos, hour_sin/cos, month_sin/cos (cyclical encoding)
    - Statistical: z-score, min-max, percentile rank
    - Domain: anomaly flags, deviation from typical
    
    Parameters:
    -----------
    live_value : float
        Current AQI observation
    train_data : pd.Series
        Training data for statistical context
    
    Returns:
    --------
    np.ndarray : Feature vector of shape (13,)
    
    Feature Vector (13 dimensions):
    [0] raw value
    [1-2] day_sin, day_cos
    [3-4] hour_sin, hour_cos
    [5-6] month_sin, month_cos
    [7] z-score normalized
    [8] min-max normalized
    [9] percentile rank
    [10] is unusually high
    [11] is unusually low
    [12] deviation from typical day
    """
    if train_data is None:
        raise ValueError("train_data required for feature engineering")
    
    today = pd.Timestamp.now()
    
    # ===== TEMPORAL FEATURES (Cyclical Encoding) =====
    day_of_week = today.dayofweek
    day_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_cos = np.cos(2 * np.pi * day_of_week / 7)
    
    hour = today.hour
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    
    month = today.month
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    # ===== STATISTICAL CONTEXT =====
    train_mean = train_data.mean()
    train_std = train_data.std()
    train_min = train_data.min()
    train_max = train_data.max()
    
    # Normalization approaches
    normalized_value = (live_value - train_mean) / train_std if train_std > 0 else 0
    normalized_min_max = (live_value - train_min) / (train_max - train_min) if train_max > train_min else 0.5
    
    # Percentile rank
    percentile_rank = np.mean(train_data <= live_value)
    
    # Anomaly detection (±2 std dev)
    is_unusually_high = 1.0 if live_value > train_mean + 2*train_std else 0.0
    is_unusually_low = 1.0 if live_value < train_mean - 2*train_std else 0.0
    
    # Deviation from typical for this day of week
    typical_for_day = train_data[train_data.index.dayofweek == day_of_week].mean()
    deviation_from_typical = live_value - typical_for_day
    
    # ===== COMBINE ALL FEATURES =====
    feature_vector = np.array([
        live_value,
        day_sin, day_cos,
        hour_sin, hour_cos,
        month_sin, month_cos,
        normalized_value,
        normalized_min_max,
        percentile_rank,
        is_unusually_high,
        is_unusually_low,
        deviation_from_typical,
    ])
    
    return feature_vector


def create_feature_enriched_sequences(data, lookback=1, forecast_horizon=3, train_data=None):
    """
    Create sequences with engineered features for multi-day forecasting.
    
    Transforms raw time series into sequences where:
    - Input: (lookback, 13 features) - 13 engineered features per timestep
    - Output: (forecast_horizon,) - next 3 days of values
    
    Parameters:
    -----------
    data : pd.Series
        Time series data (cleaned and normalized)
    lookback : int
        Number of timesteps to look back (default: 1)
    forecast_horizon : int
        Number of days to forecast (default: 3)
    train_data : pd.Series
        Full training data for statistical context
    
    Returns:
    --------
    X : np.ndarray of shape (num_sequences, lookback, 13)
        Input sequences with engineered features
    y : np.ndarray of shape (num_sequences, forecast_horizon)
        Target values (next 3 days)
    """
    if train_data is None:
        train_data = data.copy()
    
    train_mean = train_data.mean()
    train_std = train_data.std()
    train_min = train_data.min()
    train_max = train_data.max()
    
    X, y = [], []
    
    for i in range(len(data) - lookback - forecast_horizon + 1):
        
        # ===== PREPARE INPUTS: Engineered features for each lookback timestep =====
        lookback_features = []
        
        for t in range(lookback):
            current_idx = i + t
            current_value = data.iloc[current_idx]
            current_date = data.index[current_idx]
            
            # Temporal features - handle both DatetimeIndex and numeric index
            if isinstance(current_date, pd.Timestamp):
                day_of_week = current_date.dayofweek
                month = current_date.month
                hour = current_date.hour if hasattr(current_date, 'hour') else 0
            else:
                # If index is numeric, use modular arithmetic as proxy
                day_of_week = current_idx % 7
                month = (current_idx // 30) % 12 + 1
                hour = (current_idx % 24)
            
            day_sin = np.sin(2 * np.pi * day_of_week / 7)
            day_cos = np.cos(2 * np.pi * day_of_week / 7)
            
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)
            
            # Statistical features
            normalized_value = (current_value - train_mean) / train_std if train_std > 0 else 0
            normalized_min_max = (current_value - train_min) / (train_max - train_min) if train_max > train_min else 0.5
            percentile_rank = np.mean(train_data <= current_value)
            
            # Anomaly detection
            is_unusually_high = 1.0 if current_value > train_mean + 2*train_std else 0.0
            is_unusually_low = 1.0 if current_value < train_mean - 2*train_std else 0.0
            
            # Deviation from typical day
            typical_for_day = train_data[train_data.index.dayofweek == day_of_week].mean()
            deviation_from_typical = current_value - typical_for_day
            
            # Combine all features
            features = np.array([
                current_value,
                day_sin, day_cos,
                hour_sin, hour_cos,
                month_sin, month_cos,
                normalized_value,
                normalized_min_max,
                percentile_rank,
                is_unusually_high,
                is_unusually_low,
                deviation_from_typical,
            ])
            
            lookback_features.append(features)
        
        X.append(np.array(lookback_features))
        
        # ===== PREPARE TARGETS: Next forecast_horizon days =====
        target_start = i + lookback
        target_end = target_start + forecast_horizon
        target = data.iloc[target_start:target_end].values
        
        y.append(target)
    
    return np.array(X), np.array(y)


# =============================================================================
# SECTION 3: EVALUATION METRICS
# =============================================================================

def evaluate_forecast(actual, predicted, model_name="Model"):
    """
    Calculate comprehensive evaluation metrics for forecasts.
    
    Metrics:
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - MAPE: Mean Absolute Percentage Error
    - sMAPE: Symmetric Mean Absolute Percentage Error
    
    Parameters:
    -----------
    actual : array-like
        Actual values
    predicted : array-like
        Predicted values
    model_name : str
        Name of model for reporting
    
    Returns:
    --------
    dict : Dictionary with all metrics
    """
    mae = mean_absolute_error(actual, predicted)
    rmse = sqrt(mean_squared_error(actual, predicted))
    mape = mean_absolute_percentage_error(actual, predicted)
    
    # sMAPE: Symmetric Mean Absolute Percentage Error
    numerator = np.abs(actual - predicted)
    denominator = np.abs(actual) + np.abs(predicted)
    smape_values = np.divide(numerator, denominator, where=denominator!=0, out=np.zeros_like(numerator))
    smape = 2 * np.mean(smape_values)
    
    metrics = {
        'Model': model_name,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'sMAPE': smape
    }
    
    return metrics


# =============================================================================
# SECTION 4: MODEL 1 - NAIVE FORECAST (Baseline)
# =============================================================================

def train_naive_model(train_data, test_data):
    """
    Train naive forecast model (repeat last week pattern).
    
    Strategy:
    - Use last 7 days from training data
    - Repeat this weekly pattern to cover entire test period
    
    Parameters:
    -----------
    train_data : pd.Series or np.ndarray
        Training data
    test_data : pd.Series or np.ndarray
        Test data (used only to determine forecast horizon)
    
    Returns:
    --------
    dict : Dictionary with model info and predictions
    """
    last_week_train = train_data.iloc[-7:].values if isinstance(train_data, pd.Series) else train_data[-7:]
    forecast_horizon = len(test_data)
    
    # Tile the 7-day pattern
    naive_forecast = np.tile(last_week_train, int(np.ceil(forecast_horizon / 7)))[:forecast_horizon]
    
    return {
        'model_name': 'Naive',
        'forecast': naive_forecast,
        'strategy': 'Repeat last 7 days pattern'
    }


def predict_naive(model_output, horizon=1):
    """
    Make prediction with naive model (already computed).
    
    Parameters:
    -----------
    model_output : dict
        Output from train_naive_model
    horizon : int
        Forecast horizon (not used for naive model)
    
    Returns:
    --------
    np.ndarray : Predictions
    """
    return model_output['forecast']


# =============================================================================
# SECTION 5: MODEL 2 - ARIMA/SARIMA
# =============================================================================

def train_arima_model(train_data, test_data, order=(1, 1, 1), 
                     seasonal_order=(0, 0, 0, 12)):
    """
    Train ARIMA/SARIMA model for time series forecasting.
    
    Uses rolling window with periodic refitting for better forecast quality.
    
    Parameters:
    -----------
    train_data : pd.Series
        Training data
    test_data : pd.Series
        Test data
    order : tuple
        ARIMA order (p, d, q)
    seasonal_order : tuple
        SARIMA seasonal order (P, D, Q, s)
    
    Returns:
    --------
    dict : Dictionary with model and predictions
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("Statsmodels required for ARIMA. Install with: pip install statsmodels")
    
    forecast_test = []
    history = train_data.copy()
    batch_size = max(1, len(test_data) // 25)
    fitted = None
    
    try:
        for idx, test_val in enumerate(test_data):
            # Refit model at batch intervals
            if idx % batch_size == 0:
                try:
                    fitted = SARIMAX(
                        history,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    ).fit(disp=False)
                except:
                    pass  # Use previous fit if current fails
            
            # Make prediction
            if fitted is not None:
                forecast_step = fitted.get_forecast(steps=1).predicted_mean.values[0]
            else:
                forecast_step = history.mean()
            
            forecast_test.append(forecast_step)
            history = pd.concat([history, pd.Series([test_val], index=[test_data.index[idx]])])
        
        return {
            'model_name': f'ARIMA{order}',
            'forecast': np.array(forecast_test),
            'fitted_model': fitted,
            'order': order,
            'seasonal_order': seasonal_order
        }
    
    except Exception as e:
        raise RuntimeError(f"ARIMA training failed: {e}")


def predict_arima(model_output, horizon=1):
    """
    Make predictions with ARIMA model.
    
    Parameters:
    -----------
    model_output : dict
        Output from train_arima_model
    horizon : int
        Forecast horizon
    
    Returns:
    --------
    np.ndarray : Predictions
    """
    if model_output['fitted_model'] is None:
        raise ValueError("Model not properly fitted")
    
    forecast = model_output['fitted_model'].get_forecast(steps=horizon)
    return forecast.predicted_mean.values


# =============================================================================
# SECTION 6: MODEL 3 - PROPHET
# =============================================================================

def train_prophet_model(train_data, yearly_seasonality=True, 
                       changepoint_prior_scale=0.05):
    """
    Train Facebook Prophet model for time series forecasting.
    
    Prophet is particularly good at:
    - Handling seasonality
    - Capturing trend changes
    - Robust to missing data
    
    Parameters:
    -----------
    train_data : pd.Series
        Training data with datetime index
    yearly_seasonality : bool
        Whether to include yearly seasonality
    changepoint_prior_scale : float
        Flexibility of trend changes (higher = more flexible)
    
    Returns:
    --------
    dict : Dictionary with Prophet model and info
    """
    if not PROPHET_AVAILABLE:
        raise ImportError("Prophet required. Install with: pip install prophet")
    
    # Prepare data in Prophet format
    train_prophet = pd.DataFrame({
        'ds': train_data.index,
        'y': train_data.values
    })
    
    # Initialize and fit model
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
        interval_width=0.95
    )
    
    model.fit(train_prophet)
    
    return {
        'model_name': 'Prophet',
        'model': model,
        'train_data': train_prophet
    }


def predict_prophet(model_output, horizon=7):
    """
    Make predictions with Prophet model.
    
    Parameters:
    -----------
    model_output : dict
        Output from train_prophet_model
    horizon : int
        Number of steps to forecast ahead
    
    Returns:
    --------
    np.ndarray : Predicted values
    """
    model = model_output['model']
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=horizon)
    forecast = model.predict(future)
    
    # Return only the forecasted portion
    return forecast['yhat'].values[-horizon:]


# =============================================================================
# SECTION 7: MODEL 4 - LSTM (Long Short-Term Memory)
# =============================================================================

def train_lstm_model(train_data, test_data, lookback=1, forecast_horizon=3,
                    epochs=50, batch_size=16, validation_split=0.2):
    """
    Train LSTM model for time series forecasting with 3-day ahead predictions.
    
    Architecture:
    - LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(16) → Dense(3)
    - Input: (lookback=1, features=13) with engineered features
    - Output: 3-day forecast
    
    Parameters:
    -----------
    train_data : pd.Series
        Training time series
    test_data : pd.Series
        Test time series (used for validation)
    lookback : int
        Number of timesteps to look back (default: 1)
    forecast_horizon : int
        Number of days to forecast (default: 3)
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    validation_split : float
        Proportion of training data for validation
    
    Returns:
    --------
    dict : Dictionary with model, predictions, and metadata
    """
    # Lazy load TensorFlow on first use
    if not _ensure_tensorflow_available():
        raise ImportError("TensorFlow required for LSTM. Install with: pip install tensorflow")
    
    # Data preparation
    data_prep = prepare_data_for_deep_learning(train_data, test_data, lookback)
    
    # Create feature-enriched sequences
    X_enriched, y_enriched = create_feature_enriched_sequences(
        data_prep['train'],
        lookback=lookback,
        forecast_horizon=forecast_horizon,
        train_data=data_prep['train']
    )
    
    # Split into train/test (85/15)
    train_size = int(len(X_enriched) * 0.85)
    X_train = X_enriched[:train_size]
    y_train = y_enriched[:train_size]
    X_test = X_enriched[train_size:]
    y_test = y_enriched[train_size:]
    
    # Build model
    model = Sequential([
        LSTM(64, activation='relu', 
             input_shape=(X_train.shape[1], X_train.shape[2]), 
             return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='relu', return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(forecast_horizon, activation='linear')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    # Early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=0
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stop],
        verbose=0
    )
    
    # Make predictions
    y_pred = model.predict(X_test, verbose=0)
    
    return {
        'model_name': 'LSTM',
        'model': model,
        'history': history,
        'predictions': y_pred,
        'y_test': y_test,
        'X_test': X_test,
        'scaler': data_prep['scaler'],
        'lookback': lookback,
        'forecast_horizon': forecast_horizon
    }


def predict_lstm(model_output, X_new):
    """
    Make predictions with trained LSTM model.
    
    Parameters:
    -----------
    model_output : dict
        Output from train_lstm_model
    X_new : np.ndarray
        New input data with shape (samples, lookback, features=13)
    
    Returns:
    --------
    np.ndarray : Predictions with shape (samples, forecast_horizon)
    """
    model = model_output['model']
    return model.predict(X_new, verbose=0)


# =============================================================================
# SECTION 8: MODEL 5 - CNN (Convolutional Neural Network)
# =============================================================================

def train_cnn_model(train_data, test_data, lookback=1, forecast_horizon=3,
                   epochs=50, batch_size=16, validation_split=0.2):
    """
    Train CNN model for time series forecasting with 3-day ahead predictions.
    
    Architecture:
    - Conv1D(32, kernel=3) → Dropout(0.2) → Conv1D(64, kernel=3) → Dropout(0.2) 
      → Flatten → Dense(16) → Dense(3)
    - Input: (lookback=1, features=13) with engineered features
    - Output: 3-day forecast
    
    Parameters:
    -----------
    train_data : pd.Series
        Training time series
    test_data : pd.Series
        Test time series (used for validation)
    lookback : int
        Number of timesteps to look back (default: 1)
    forecast_horizon : int
        Number of days to forecast (default: 3)
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    validation_split : float
        Proportion of training data for validation
    
    Returns:
    --------
    dict : Dictionary with model, predictions, and metadata
    """
    # Lazy load TensorFlow on first use
    if not _ensure_tensorflow_available():
        raise ImportError("TensorFlow required for CNN. Install with: pip install tensorflow")
    
    # Data preparation
    data_prep = prepare_data_for_deep_learning(train_data, test_data, lookback)
    
    # Create feature-enriched sequences
    X_enriched, y_enriched = create_feature_enriched_sequences(
        data_prep['train'],
        lookback=lookback,
        forecast_horizon=forecast_horizon,
        train_data=data_prep['train']
    )
    
    # Split into train/test (85/15)
    train_size = int(len(X_enriched) * 0.85)
    X_train = X_enriched[:train_size]
    y_train = y_enriched[:train_size]
    X_test = X_enriched[train_size:]
    y_test = y_enriched[train_size:]
    
    # Build model
    model = Sequential([
        Conv1D(32, kernel_size=1, activation='relu',
               input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        Conv1D(64, kernel_size=1, activation='relu'),
        Dropout(0.2),
        Flatten(),
        Dense(16, activation='relu'),
        Dense(forecast_horizon, activation='linear')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    # Early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=0
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stop],
        verbose=0
    )
    
    # Make predictions
    y_pred = model.predict(X_test, verbose=0)
    
    return {
        'model_name': 'CNN',
        'model': model,
        'history': history,
        'predictions': y_pred,
        'y_test': y_test,
        'X_test': X_test,
        'scaler': data_prep['scaler'],
        'lookback': lookback,
        'forecast_horizon': forecast_horizon
    }


def predict_cnn(model_output, X_new):
    """
    Make predictions with trained CNN model.
    
    Parameters:
    -----------
    model_output : dict
        Output from train_cnn_model
    X_new : np.ndarray
        New input data with shape (samples, lookback, features=13)
    
    Returns:
    --------
    np.ndarray : Predictions with shape (samples, forecast_horizon)
    """
    model = model_output['model']
    return model.predict(X_new, verbose=0)



# =============================================================================
# END OF MODELS MODULE
# =============================================================================