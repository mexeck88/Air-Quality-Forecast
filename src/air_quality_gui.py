"""
Air Quality Forecasting GUI Application

A user-friendly interface for:
- Selecting cities and viewing AQI forecasts
- Inputting personal health information
- Running model training and predictions
- Displaying comprehensive visualizations
- Receiving personalized health recommendations
"""

# Suppress TensorFlow and Keras warnings/output before importing anything
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU to avoid threading issues

import PySimpleGUI as sg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Suppress Tkinter threading warnings that occur when TensorFlow cleans up in background threads
import logging
logging.getLogger('tkinter').setLevel(logging.ERROR)

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))


# Import from project modules
from data_handler import load_saved_epa_data
from models import (
    train_naive_model,
    predict_naive,
    train_lstm_model,
    predict_lstm,
    train_prophet_model,
    predict_prophet,
    train_arima_model,
    predict_arima,
    train_cnn_model,
    predict_cnn,
    evaluate_forecast,
    prepare_data_for_deep_learning,
    predict_3day_ahead_lstm_cnn,
    create_feature_enriched_sequences
)
from visualizations import (
    plot_3day_forecast,
    plot_predictions_comparison,
    model_results_bar_chart
)

from recommendations import get_aqi_level, get_health_recommendation

# ========================
# TRAINING CLASS
# ========================

class ModelTrainer:
    """Class for training air quality prediction models using LSTM, Prophet, and ARIMA"""

    def __init__(self, city, pollutant='PM2.5', data_path=None):
        """
        Initialize the model trainer
        
        Parameters:
        -----------
        city : str
            City name for training data
        pollutant : str
            Pollutant to forecast (default: 'PM2.5')
        data_path : str, optional
            Path to EPA data file
        """
        self.city = city
        self.pollutant = pollutant
        self.data_path = data_path or Path(__file__).parent.parent / 'data' / 'epa_aqs_data_2025_cleaned.csv'
        
        self.train_data = None
        self.test_data = None
        self.lstm_model = None
        self.prophet_model = None
        self.arima_model = None
        self.naive_model = None
        self.cnn_model = None
        self.scaler = None
        self.metrics = {}

    def load_data(self):
        """Load EPA data for the specified city and pollutant, with fallback to any available pollutant"""
        try:
            epa_data = load_saved_epa_data(str(self.data_path))
            
            # Extract city name from "City, State" format
            city_name = self.city.split(',')[0].strip()
            
            # Filter for city
            city_data = epa_data[epa_data['city'].str.lower() == city_name.lower()]
            if city_data.empty:
                raise ValueError(f"No data found for city: {city_name}")
            
            # Filter for pollutant with fallback logic
            pollutant_variations = {
                'PM2.5': ['PM2.5', 'PM2.5 - Local Conditions', 'PM25'],
                'Ozone': ['Ozone', 'O3'],
                'NO2': ['NO2', 'Nitrogen Dioxide'],
                'SO2': ['SO2', 'Sulfur Dioxide'],
                'CO': ['CO', 'Carbon Monoxide'],
            }
            variations = pollutant_variations.get(self.pollutant, [self.pollutant])
            pollutant_pattern = '|'.join(variations)
            
            pollutant_data = city_data[
                city_data['parameter'].str.contains(pollutant_pattern, case=False, regex=True, na=False)
            ]
            
            # Fallback to any available pollutant if selected one not found
            if pollutant_data.empty:
                available_pollutants = city_data['parameter'].unique()
                if len(available_pollutants) == 0:
                    raise ValueError(f"No pollutant data found for {city_name}")
                
                # Use the first available pollutant
                fallback_pollutant = available_pollutants[0]
                print(f"  Warning: {self.pollutant} not available for {city_name}")
                print(f"  Falling back to: {fallback_pollutant}")
                self.pollutant = fallback_pollutant
                
                pollutant_data = city_data[city_data['parameter'] == fallback_pollutant]
            
            # Aggregate to daily values
            pollutant_data_copy = pollutant_data.copy()
            pollutant_data_copy['datetime'] = pd.to_datetime(pollutant_data_copy['datetime'])
            pollutant_data_copy = pollutant_data_copy.sort_values('datetime')
            
            ts_data = pollutant_data_copy.groupby('datetime')['aqi'].mean()
            
            # Split into train/test
            split_idx = int(len(ts_data) * 0.8)
            self.train_data = ts_data.iloc[:split_idx]
            self.test_data = ts_data.iloc[split_idx:]
            
            print(f" Data loaded for {city_name} - {self.pollutant}")
            print(f"  Training samples: {len(self.train_data)}")
            print(f"  Test samples: {len(self.test_data)}")
            
            return True
            
        except Exception as e:
            print(f" Error loading data: {e}")
            return False

    def train_lstm(self, epochs=50, batch_size=16):
        """Train LSTM model for 3-day forecasting"""
        if self.train_data is None:
            self.load_data()
        
        try:
            print(f"\nTraining LSTM model... (If this is the first forecast, this may take a moment)")
            
            # Prepare data
            data_prep = prepare_data_for_deep_learning(self.train_data, self.test_data, lookback=1)
            self.scaler = data_prep['scaler']
            
            # Create feature-enriched sequences
            X_enriched, y_enriched = create_feature_enriched_sequences(
                pd.Series(data_prep['scaled_train']),
                lookback=1,
                forecast_horizon=3,
                train_data=self.train_data
            )
            
            # Train LSTM
            lstm_output = train_lstm_model(
                self.train_data,
                self.test_data,
                lookback=1,
                forecast_horizon=3,
                epochs=epochs,
                batch_size=batch_size
            )
            
            self.lstm_model = lstm_output
            
            # LSTM now returns rolling 1-day predictions (not 3-day sequences)
            y_pred = lstm_output['predictions']  # Shape: (test_size,)
            y_actual = lstm_output['y_test']      # Shape: (test_size,)
            
            # Evaluate using same method as notebook (sklearn metrics for consistency)
            from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
            from math import sqrt as sqrt_fn
            
            mae = mean_absolute_error(y_actual, y_pred)
            rmse = sqrt_fn(mean_squared_error(y_actual, y_pred))
            mape = mean_absolute_percentage_error(y_actual, y_pred)
            
            # sMAPE calculation
            numerator = np.abs(y_actual - y_pred)
            denominator = np.abs(y_actual) + np.abs(y_pred)
            smape_values = np.divide(numerator, denominator, where=denominator!=0, out=np.zeros_like(numerator))
            smape = 2 * np.mean(smape_values)
            
            # Create a new metrics dict for this model (don't use shared self.metrics)
            metrics_dict = {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'sMAPE': smape
            }
            
            # Get 3-day forecast using the new prediction function
            # Use the last training value as the "live" input for forward prediction
            try:
                live_value = float(self.train_data.iloc[-1])
                forecast_3day = predict_3day_ahead_lstm_cnn(
                    self.lstm_model['model'],
                    live_value,
                    self.train_data
                )
            except (ValueError, RuntimeError, TypeError, AttributeError) as e:
                print(f"Warning: Could not generate 3-day forecast: {e}")
                # Fallback: use last 3 rolling predictions
                if len(y_pred) >= 3:
                    forecast_3day = y_pred[-3:]
                elif len(y_pred) > 0:
                    last_val = y_pred[-1]
                    forecast_3day = np.concatenate([y_pred[-(min(len(y_pred), 3)-1):], [last_val] * max(0, 3 - len(y_pred))])
                else:
                    forecast_3day = np.array([45.2, 45.1, 46.3])
            
            print("LSTM training complete!")
            print(f"  RMSE: {metrics_dict['RMSE']:.4f}")
            
            return {
                'model_name': 'LSTM',
                'metrics': metrics_dict,
                'y_pred': y_pred,
                'y_actual': lstm_output['y_test'],
                'forecast_3day': forecast_3day,
                'current_aqi': float(self.train_data.iloc[-1]),
                'model': self.lstm_model
            }
            
        except (ValueError, RuntimeError, TypeError, AttributeError) as e:
            print(f"LSTM training failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def train_prophet(self):
        """Train Prophet model for time series forecasting"""
        if self.train_data is None:
            self.load_data()
        
        try:
            print("Training Prophet model...")
            
            prophet_output = train_prophet_model(self.train_data)
            
            # Forecast for test period
            y_pred = predict_prophet(prophet_output, len(self.test_data))
            y_actual = self.test_data.values
            
            # Calculate all required metrics using same method as notebook (sklearn metrics for consistency)
            from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
            from math import sqrt as sqrt_fn
            
            mae = mean_absolute_error(y_actual, y_pred)
            rmse = sqrt_fn(mean_squared_error(y_actual, y_pred))
            mape = mean_absolute_percentage_error(y_actual, y_pred)
            
            numerator = np.abs(y_actual - y_pred)
            denominator = np.abs(y_actual) + np.abs(y_pred)
            smape_values = np.divide(numerator, denominator, where=denominator!=0, out=np.zeros_like(numerator))
            smape = 2 * np.mean(smape_values)
            
            # Create a new metrics dict for this model (don't use shared self.metrics)
            metrics_dict = {
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape,
                'sMAPE': smape
            }
            
            print("Prophet training complete!")
            
            return {
                'model_name': 'Prophet',
                'metrics': metrics_dict,
                'y_pred': y_pred,
                'y_actual': self.test_data.values,
                'model': prophet_output
            }
            
        except (ValueError, RuntimeError, TypeError, AttributeError) as e:
            print(f"Prophet training failed: {e}")
            return None

    def train_naive(self):
        """Train Naive baseline model (repeats last 7-day pattern)"""
        if self.train_data is None:
            self.load_data()
        
        try:
            print("Training Naive baseline model...")
            
            naive_output = train_naive_model(self.train_data, self.test_data)
            y_pred = predict_naive(naive_output, len(self.test_data))
            y_actual = self.test_data.values
            
            # Calculate all required metrics using same method as notebook (sklearn metrics for consistency)
            from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
            from math import sqrt as sqrt_fn
            
            mae = mean_absolute_error(y_actual, y_pred)
            rmse = sqrt_fn(mean_squared_error(y_actual, y_pred))
            mape = mean_absolute_percentage_error(y_actual, y_pred)
            
            numerator = np.abs(y_actual - y_pred)
            denominator = np.abs(y_actual) + np.abs(y_pred)
            smape_values = np.divide(numerator, denominator, where=denominator!=0, out=np.zeros_like(numerator))
            smape = 2 * np.mean(smape_values)
            
            # Create a new metrics dict for this model (don't use shared self.metrics)
            metrics_dict = {
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape,
                'sMAPE': smape
            }
            
            print("Naive model training complete!")
            print(f"  RMSE: {metrics_dict['RMSE']:.4f}")
            print(f"  MAE: {metrics_dict['MAE']:.4f}")
            
            return {
                'model_name': 'Naive',
                'metrics': metrics_dict,
                'y_pred': y_pred,
                'y_actual': y_actual,
                'model': naive_output
            }
            
        except (ValueError, RuntimeError, TypeError, AttributeError) as e:
            print(f"Naive model training failed: {e}")
            return None

    def train_arima(self, order=(1, 1, 1), seasonal_order=(0, 0, 0, 12)):
        """Train ARIMA/SARIMA model for time series forecasting"""
        if self.train_data is None:
            self.load_data()
        
        try:
            print("Training ARIMA model...")
            
            arima_output = train_arima_model(
                self.train_data,
                self.test_data,
                order=order,
                seasonal_order=seasonal_order
            )
            
            y_pred = predict_arima(arima_output, len(self.test_data))
            y_actual = self.test_data.values
            
            # Calculate all required metrics using same method as notebook (sklearn metrics for consistency)
            from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
            from math import sqrt as sqrt_fn
            
            mae = mean_absolute_error(y_actual, y_pred)
            rmse = sqrt_fn(mean_squared_error(y_actual, y_pred))
            mape = mean_absolute_percentage_error(y_actual, y_pred)
            
            numerator = np.abs(y_actual - y_pred)
            denominator = np.abs(y_actual) + np.abs(y_pred)
            smape_values = np.divide(numerator, denominator, where=denominator!=0, out=np.zeros_like(numerator))
            smape = 2 * np.mean(smape_values)
            
            # Create a new metrics dict for this model (don't use shared self.metrics)
            metrics_dict = {
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape,
                'sMAPE': smape
            }
            
            print("ARIMA model training complete!")
            print(f"  RMSE: {metrics_dict['RMSE']:.4f}")
            print(f"  MAE: {metrics_dict['MAE']:.4f}")
            
            return {
                'model_name': 'ARIMA',
                'metrics': metrics_dict,
                'y_pred': y_pred,
                'y_actual': y_actual,
                'model': arima_output
            }
            
        except (ValueError, RuntimeError, TypeError, AttributeError) as e:
            print(f"ARIMA model training failed: {e}")
            return None

    def train_cnn(self, epochs=50, batch_size=16):
        """Train CNN model for 3-day forecasting"""
        if self.train_data is None:
            self.load_data()
        
        try:
            print("Training CNN model with feature enrichment...")
            
            # Prepare data with scaling
            data_prep = prepare_data_for_deep_learning(self.train_data, self.test_data, lookback=1)
            self.scaler = data_prep['scaler']
            
            # Create feature-enriched sequences
            X_enriched, _ = create_feature_enriched_sequences(
                pd.Series(data_prep['scaled_train']),
                lookback=1,
                forecast_horizon=3,
                train_data=self.train_data
            )
            
            print(f"  Data enriched: {X_enriched.shape[0]} sequences with {X_enriched.shape[2]} features")
            
            # Train CNN
            cnn_output = train_cnn_model(
                self.train_data,
                self.test_data,
                lookback=1,
                forecast_horizon=3,
                epochs=epochs,
                batch_size=batch_size
            )
            
            self.cnn_model = cnn_output
            
            # CNN now returns rolling 1-day predictions (not 3-day sequences)
            y_pred = cnn_output['predictions']  # Shape: (test_size,)
            y_actual = cnn_output['y_test']     # Shape: (test_size,)
            
            # Calculate all required metrics using same method as notebook (sklearn metrics for consistency)
            from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
            from math import sqrt as sqrt_fn
            
            mae = mean_absolute_error(y_actual, y_pred)
            rmse = sqrt_fn(mean_squared_error(y_actual, y_pred))
            mape = mean_absolute_percentage_error(y_actual, y_pred)
            
            # sMAPE calculation
            numerator = np.abs(y_actual - y_pred)
            denominator = np.abs(y_actual) + np.abs(y_pred)
            smape_values = np.divide(numerator, denominator, where=denominator!=0, out=np.zeros_like(numerator))
            smape = 2 * np.mean(smape_values)
            
            # Create a new metrics dict for this model (don't use shared self.metrics)
            metrics_dict = {
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape,
                'sMAPE': smape
            }
            
            # Get 3-day forecast using the new prediction function
            # Use the last training value as the "live" input for forward prediction
            try:
                live_value = float(self.train_data.iloc[-1])
                forecast_3day = predict_3day_ahead_lstm_cnn(
                    self.cnn_model['model'],
                    live_value,
                    self.train_data
                )
            except (ValueError, RuntimeError, TypeError, AttributeError) as e:
                print(f"Warning: Could not generate 3-day forecast: {e}")
                # Fallback: use last 3 rolling predictions
                if len(y_pred) >= 3:
                    forecast_3day = y_pred[-3:]
                elif len(y_pred) > 0:
                    last_val = y_pred[-1]
                    forecast_3day = np.concatenate([y_pred[-(min(len(y_pred), 3)-1):], [last_val] * max(0, 3 - len(y_pred))])
                else:
                    forecast_3day = np.array([45.2, 45.1, 46.3])
            
            print("CNN training complete!")
            print(f"  RMSE: {metrics_dict['RMSE']:.4f}")
            print(f"  MAE: {metrics_dict['MAE']:.4f}")
            print(f"  MAPE: {metrics_dict['MAPE']:.2f}%")
            print(f"  sMAPE: {metrics_dict['sMAPE']:.2f}%")
            
            return {
                'model_name': 'CNN',
                'metrics': metrics_dict,
                'y_pred': y_pred,
                'y_actual': y_actual,
                'forecast_3day': forecast_3day,
                'current_aqi': float(self.train_data.iloc[-1]),
                'model': self.cnn_model
            }
            
        except (ValueError, RuntimeError, TypeError, AttributeError) as e:
            print(f"CNN training failed: {e}")
            import traceback
            traceback.print_exc()
            return None


# Configure PySimpleGUI theme
sg.theme('LightBlue2')
sg.set_options(font=('Arial', 10))

# Cache EPA data at module load to avoid redundant I/O during GUI initialization
_EPA_DATA_CACHE = None
_EPA_DATA_CACHE_PATH = None

def _load_epa_data_cache(data_path):
    """Load EPA data once and cache it"""
    global _EPA_DATA_CACHE, _EPA_DATA_CACHE_PATH
    if _EPA_DATA_CACHE is None or _EPA_DATA_CACHE_PATH != str(data_path):
        _EPA_DATA_CACHE = load_saved_epa_data(str(data_path))
        _EPA_DATA_CACHE_PATH = str(data_path)
    return _EPA_DATA_CACHE

def get_available_pollutants(city, data_path=None):
    """
    Get available pollutants for a selected city from EPA data.
    Uses cached EPA data to avoid redundant file I/O.
    
    Parameters:
    -----------
    city : str
        City name in format "City, State"
    data_path : str, optional
        Path to EPA data file
    
    Returns:
    --------
    list : List of available pollutants for the city
    """
    try:
        if data_path is None:
            data_path = Path(__file__).parent.parent / 'data' / 'epa_aqs_data_2025_cleaned.csv'
        
        # Use cached EPA data instead of loading from disk
        epa_data = _load_epa_data_cache(data_path)
        
        # Extract city name from "City, State" format
        city_name = city.split(',')[0].strip()
        
        # Filter for city
        city_data = epa_data[epa_data['city'].str.lower() == city_name.lower()]
        
        if city_data.empty:
            return AVAILABLE_POLLUTANTS  # Fallback to all pollutants
        
        # Get unique pollutants for this city
        available = city_data['parameter'].unique().tolist()
        
        # Sort to maintain consistent order
        available = sorted([p for p in available if p in AVAILABLE_POLLUTANTS])
        
        return available if available else AVAILABLE_POLLUTANTS
        
    except Exception as e:
        print(f"Error getting available pollutants: {e}")
        return AVAILABLE_POLLUTANTS  # Fallback to all pollutants


# Available cities from EPA data (with state abbreviations)
# Only includes cities that actually exist in epa_aqs_data_2025_cleaned.csv
AVAILABLE_CITIES = [
    "Alsip, IL",
    "Baytown, TX",
    "Blue Point, NY",
    "Boston, MA",
    "Buckeye, AZ",
    "Cave Creek, AZ",
    "Chandler, AZ",
    "Channelview, TX",
    "Chelsea, MA",
    "Chicago, IL",
    "Cicero, IL",
    "Compton, CA",
    "Deer Park, TX",
    "Des Plaines, IL",
    "Evanston, IL",
    "Fort McDowell, AZ",
    "Fountain Hills, AZ",
    "Gilbert, AZ",
    "Glendale, AZ",
    "Glendora, CA",
    "Houston, TX",
    "Lancaster, CA",
    "Lansing, IL",
    "Lemont, IL",
    "Long Beach, CA",
    "Los Angeles, CA",
    "McCook, IL",
    "Mesa, AZ",
    "New York, NY",
    "Northbrook, IL",
    "Pasadena, CA",
    "Philadelphia, PA",
    "Phoenix, AZ",
    "Pico Rivera, CA",
    "Pomona, CA",
    "Reseda, CA",
    "Santa Clarita, CA",
    "Schiller Park, IL",
    "Scottsdale, AZ",
    "Seabrook, TX",
    "Signal Hill, CA",
    "Summit, IL",
    "Surprise, AZ",
    "Tempe, AZ",
    "Tomball, TX",
    "West Los Angeles, CA"
]

# Available pollutants for forecasting
AVAILABLE_POLLUTANTS = [
    'CO',
    'NO2',
    'Ozone',
    'PM10',
    'PM2.5',
    'SO2'
]

class AirQualityGUI:
    """Main GUI application class for Air Quality Forecasting"""
    
    def __init__(self):
        """Initialize the GUI application"""
        self.model_trainer = None
        self.current_forecast = None
        self.is_training = False
        self.training_result = None
        self.window = None
        
    def create_window(self):
        """Create the main application window"""
        
        # Get initial pollutants for default city
        default_city = 'Los Angeles'
        initial_pollutants = get_available_pollutants(default_city)
        default_pollutant = initial_pollutants[0] if initial_pollutants else 'PM2.5'
        
        # Define the layout
        layout = [
            # Title
            [sg.Text('Air Quality Forecasting & Health Recommendation System',
                    font=('Arial', 16, 'bold'),
                    justification='center',
                    expand_x=True)],
            [sg.Text('_' * 100, expand_x=True)],
            
            # User Input Section
            [sg.Frame('User Information', [
                [sg.Text('City:', size=(15, 1), font=('Arial', 10, 'bold')),
                 sg.Combo(AVAILABLE_CITIES, default_value='Los Angeles, CA',
                         key='-CITY-', size=(30, 1), readonly=True)],
                
                [sg.Text('Pollutant:', size=(15, 1), font=('Arial', 10, 'bold')),
                 sg.Combo(initial_pollutants, default_value=default_pollutant,
                         key='-POLLUTANT-', size=(30, 1), readonly=True)],
                
                [sg.Text('Age (optional):', size=(15, 1), font=('Arial', 10, 'bold')),
                 sg.Input(key='-AGE-', size=(30, 1), default_text='')],
                
                [sg.Text('Respiratory Issues:', size=(15, 1), font=('Arial', 10, 'bold')),
                 sg.Combo(['None', 'Mild', 'Moderate', 'Severe'],
                         default_value='None',
                         key='-RESPIRATORY-', size=(30, 1), readonly=True)],
            ], expand_x=True, font=('Arial', 10, 'bold'))],
            
            # Forecast Button Section
            [sg.Frame('Actions', [
                [sg.Button('FORECAST & ANALYZE', size=(25, 2), button_color=('white', '#0066CC')),
                 sg.Button('Clear Results', size=(25, 1)),
                 sg.Button('Exit', size=(25, 1), button_color=('white', 'red'))],
                
                [sg.ProgressBar(100, orientation='h', size=(75, 20),
                               key='-PROGRESS-', visible=False)],
                
                [sg.Text('', key='-STATUS-', font=('Arial', 10),
                        text_color='blue', expand_x=True)],
            ], expand_x=True, font=('Arial', 10, 'bold'))],
            
            # Results Tabs Section
            [sg.TabGroup([[
                sg.Tab('Forecast Results', [
                    [sg.Canvas(key='-CANVAS-FORECAST-', size=(1100, 650))],
                ]),
                sg.Tab('Model Performance', [
                    [sg.Canvas(key='-CANVAS-METRICS-', size=(1100, 650))],
                ]),
                sg.Tab('5-Panel Analysis', [
                    [sg.Canvas(key='-CANVAS-5PANEL-', size=(1100, 750))],
                ]),
                sg.Tab('Health Recommendation', [
                    [sg.Frame('Personalized Health Guidance', [
                        [sg.Text('', key='-RECOMMENDATION-TEXT-',
                                size=(100, 15),
                                font=('Arial', 11),
                                text_color='black',
                                background_color='white')],
                    ], expand_x=True, expand_y=True)],
                ]),
            ]], expand_x=True, expand_y=True, key='-TABGROUP-')],
        ]
        
        self.window = sg.Window('Air Quality Forecasting System',
                               layout,
                               size=(1400, 1000),
                               finalize=True,
                               resizable=True)
        
        return self.window
    
    def update_status(self, message, color='blue'):
        """Update status message"""
        if self.window:
            self.window['-STATUS-'].update(message, text_color=color)
    
    def show_error(self, message):
        """Show error dialog"""
        sg.popup_error(message, title='Error')
        self.update_status(f'Error: {message}', color='red')
    
    def train_model_thread(self, city, pollutant):
        """Train all models in a separate thread"""
        try:
            self.update_status(f'Initializing model trainer for {city} ({pollutant})...', 'blue')
            
            self.model_trainer = ModelTrainer(city, pollutant=pollutant)
            
            self.update_status(f'Loading data for {city}...', 'blue')
            if not self.model_trainer.load_data():
                self.show_error(f'No data available for {city} with {pollutant}')
                self.is_training = False
                return
            
            # Train all models and collect results
            all_results = []
            
            # 1. Naive Model
            self.update_status(f'Training Naive baseline model...', 'blue')
            naive_result = self.model_trainer.train_naive()
            if naive_result:
                all_results.append(naive_result)
            
            # 2. ARIMA Model
            self.update_status(f'Training ARIMA model...', 'blue')
            arima_result = self.model_trainer.train_arima()
            if arima_result:
                all_results.append(arima_result)
            
            # 3. Prophet Model
            self.update_status(f'Training Prophet model...', 'blue')
            prophet_result = self.model_trainer.train_prophet()
            if prophet_result:
                all_results.append(prophet_result)
            
            # 4. LSTM Model
            self.update_status(f'Training LSTM model... (If this is the first forecast, this may take a moment)', 'blue')
            lstm_result = self.model_trainer.train_lstm()
            if lstm_result:
                all_results.append(lstm_result)
            
            # 5. CNN Model
            self.update_status(f'Training CNN model...', 'blue')
            cnn_result = self.model_trainer.train_cnn()
            if cnn_result:
                all_results.append(cnn_result)
            
            # Find the best performing model using intelligent metric selection
            # Use MAPE as primary metric when available and RMSE values are similar
            best_result = None
            best_metric_value = float('inf')
            best_metric_name = 'RMSE'
            
            # Collect all RMSE and MAPE values
            rmse_values = []
            mape_values = []
            for result in all_results:
                if result and 'metrics' in result:
                    rmse_values.append(result['metrics'].get('RMSE', float('inf')))
                    mape_values.append(result['metrics'].get('MAPE', float('inf')))
            
            # Determine which metric to use for selection
            use_mape = False
            if rmse_values and mape_values:
                # Calculate coefficient of variation for RMSE
                valid_rmse = [r for r in rmse_values if r < float('inf')]
                if valid_rmse:
                    rmse_mean = np.mean(valid_rmse)
                    rmse_std = np.std(valid_rmse)
                    rmse_cv = rmse_std / rmse_mean if rmse_mean > 0 else float('inf')
                    
                    # If RMSE values are very similar (CV < 0.05 or 5% variation), use MAPE instead
                    if rmse_cv < 0.05:
                        use_mape = True
                        best_metric_name = 'MAPE'
            
            # Select best model based on chosen metric
            for result in all_results:
                if result and 'metrics' in result:
                    if use_mape:
                        metric_value = result['metrics'].get('MAPE', float('inf'))
                    else:
                        metric_value = result['metrics'].get('RMSE', float('inf'))
                    
                    if metric_value < best_metric_value:
                        best_metric_value = metric_value
                        best_result = result
            
            # Use best performing model for forecast display
            self.training_result = best_result if best_result else (lstm_result if lstm_result else all_results[0] if all_results else None)
            
            # Add all results to training_result for display in Model Performance tab
            if self.training_result:
                self.training_result['all_results'] = all_results
            
            best_model_name = self.training_result.get('model_name', 'Unknown') if self.training_result else 'Unknown'
            self.update_status(f'Model training complete! Best model: {best_model_name} ({best_metric_name}: {best_metric_value:.4f})', 'green')
            self.is_training = False
            
        except (ValueError, RuntimeError, TypeError, AttributeError, IOError) as e:
            self.show_error(f'Training failed: {str(e)}')
            self.is_training = False
    
    def forecast_and_analyze(self, city, pollutant, age='', respiratory='None'):
        """Run forecast analysis for selected city and pollutant"""
        
        if self.is_training:
            self.show_error('Training is already in progress. Please wait.')
            return
        
        # Validate age if provided
        if age and age != '':
            try:
                age = int(age)
                if age < 0 or age > 150:
                    self.show_error('Please enter a valid age between 0 and 150')
                    return
            except ValueError:
                self.show_error('Age must be a number')
                return
        
        # Start training in a separate thread
        self.is_training = True
        training_thread = threading.Thread(target=self.train_model_thread, args=(city, pollutant))
        training_thread.daemon = True
        training_thread.start()
        
        # While training, show progress
        while self.is_training:
            self.window.read(timeout=100)
        
        # After training, display results
        if self.training_result:
            self.display_results(city, pollutant, age, respiratory)
    
    def display_results(self, city, pollutant, age, respiratory):
        """Display forecast and analysis results"""
        
        try:
            # Get model name for display
            model_name = self.training_result.get('model_name', 'LSTM')
            
            # Tab 1: 3-Day Forecast
            self.update_status(f'Generating 3-day forecast visualization for {pollutant}...', 'blue')
            fig_forecast, _ = plot_3day_forecast(
                live_value=self.training_result.get('current_aqi', 45.2),
                predictions_3day=self.training_result.get('forecast_3day', [44.2, 45.1, 46.3]),
                pollutant=pollutant,
                title=f'3-Day {pollutant} Forecast for {city} (Model: {model_name})',
                figsize=(14, 8.5)
            )
            self.draw_figure(fig_forecast, '-CANVAS-FORECAST-')
            
            # Tab 2: Model Performance Rankings
            self.update_status('Generating model performance rankings...', 'blue')
            all_results = self.training_result.get('all_results', [])
            city_name = city.split(',')[0].strip() if ',' in city else city
            
            print(f"DEBUG: all_results = {all_results}")
            print(f"DEBUG: type(all_results) = {type(all_results)}")
            print(f"DEBUG: len(all_results) = {len(all_results) if all_results else 0}")
            
            fig_metrics = model_results_bar_chart(
                all_results=all_results,
                pollutant_focus=pollutant,
                city_focus=city_name,
                figsize=(14, 8.5)
            )
            self.draw_figure(fig_metrics, '-CANVAS-METRICS-')
            
            # Tab 3: Detailed Predictions Comparison
            self.update_status('Generating detailed forecast analysis...', 'blue')
            y_actual = self.training_result.get('y_actual', np.array([45.0, 46.0, 45.5]))
            y_pred = self.training_result.get('y_pred', np.array([44.9, 46.1, 45.3]))
            fig_comparison = plot_predictions_comparison(
                y_actual=y_actual,
                y_pred=y_pred,
                title_prefix='Detailed Forecast Analysis',
                pollutant=pollutant,
                figsize=(14, 10)
            )
            self.draw_figure(fig_comparison, '-CANVAS-5PANEL-')
            
            # Tab 4: Health Recommendation
            forecast_day1 = self.training_result.get('forecast_3day', [45.2])[0]
            aqi_level = get_aqi_level(forecast_day1)
            
            age_int = int(age) if age and age != '' else None
            recommendation = get_health_recommendation(
                aqi_value=forecast_day1,
                age=age_int,
                respiratory_issues=respiratory
            )
            
            self.window['-RECOMMENDATION-TEXT-'].update(recommendation)
            
            self.update_status(f'Analysis complete! {city} - {pollutant}: {forecast_day1:.1f} ({aqi_level})', 'green')
            
        except (ValueError, KeyError, RuntimeError, TypeError, AttributeError) as e:
            self.show_error(f'Error displaying results: {str(e)}')
    
    def draw_figure(self, fig, canvas_key):
        """Draw matplotlib figure on canvas with padding to prevent cutoff"""
        try:
            # Clear previous canvas
            canvas_elem = self.window[canvas_key]
            if canvas_elem.tk_canvas:
                for widget in canvas_elem.tk_canvas.winfo_children():
                    widget.destroy()
            
            # Draw new figure - don't apply additional subplots_adjust as figures already have it
            figure_agg = FigureCanvasTkAgg(fig, canvas_elem.tk_canvas)
            figure_agg.draw()
            # Use fill='both' and expand=1 to use all available space
            figure_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
            
            # Close figure to free memory
            plt.close(fig)
            
        except (ValueError, KeyError, RuntimeError, TypeError, AttributeError) as e:
            self.show_error(f'Error drawing figure: {str(e)}')
    
    def clear_results(self):
        """Clear all results and reset the GUI"""
        self.training_result = None
        self.model_trainer = None
        
        # Clear all canvases
        for key in ['-CANVAS-FORECAST-', '-CANVAS-METRICS-', '-CANVAS-5PANEL-']:
            try:
                canvas_elem = self.window[key]
                if canvas_elem.tk_canvas:
                    for widget in canvas_elem.tk_canvas.winfo_children():
                        widget.destroy()
            except (ValueError, KeyError, RuntimeError, AttributeError):
                pass
        
        self.window['-RECOMMENDATION-TEXT-'].update('')
        self.update_status('Results cleared. Ready for new forecast.', 'blue')
    
    def run(self):
        """Run the GUI application"""
        
        self.create_window()
        
        while True:
            event, values = self.window.read(timeout=100)
            
            if event == sg.WINDOW_CLOSED or event == 'Exit':
                break
            
            elif event == '-CITY-':
                # When city selection changes, update available pollutants
                selected_city = values['-CITY-']
                available_pollutants = get_available_pollutants(selected_city)
                
                # Update the pollutant dropdown with available options
                self.window['-POLLUTANT-'].update(
                    values=available_pollutants,
                    value=available_pollutants[0] if available_pollutants else 'PM2.5'
                )
            
            elif event == 'FORECAST & ANALYZE':
                city = values['-CITY-']
                pollutant = values['-POLLUTANT-']
                age = values['-AGE-']
                respiratory = values['-RESPIRATORY-']
                
                self.forecast_and_analyze(city, pollutant, age, respiratory)
            
            elif event == 'Clear Results':
                self.clear_results()
        
        self.window.close()


def main():
    """Main entry point for the GUI application"""
    app = AirQualityGUI()
    app.run()


if __name__ == '__main__':
    main()
