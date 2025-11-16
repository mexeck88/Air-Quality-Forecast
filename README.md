# Air Quality Forecasting and Recommendation System

This project provides an end-to-end system for air quality forecasting, model evaluation, and personalized health recommendations, complete with a graphical user interface (GUI) for easy interaction. The system supports city selection, forecasting via LSTM and classical models, and EPA-based health recommendations tailored to user profiles.

---

# Features

## 1. City Selection
Select from multiple U.S. cities such as Los Angeles, New York, Chicago, Houston, Phoenix, Philadelphia, and Boston.

## 2. Personalized Health Profile
Users may enter:
- Age (optional)
- Respiratory conditions: None, Mild, Moderate, Severe

## 3. Air Quality Forecasting
- LSTM-based AQI forecasting (default: 3-day)
- Automatic model tuning and evaluation pipeline
- Baseline and statistical models:
  - Naive Forecast
  - Moving Average
  - ARIMA / SARIMA / Prophet
  - LSTM / GRU / Transformer

## 4. GUI with Multi-Tab Results
The GUI contains:
- Forecast Results: 3-day AQI visualization
- Model Performance: MAE, RMSE, MAPE, R²
- 5-Panel Analysis Dashboard
- Health Recommendations based on AQI + user profile

---

# Project Structure

```
air_quality/
├── README.md               # Main project documentation
├── requirements.txt
│
├── data/
│   └── epa_aqs_data_2025_cleaned.csv
│
├── src/
│   ├── air_quality_gui.py      # GUI
│   ├── data_handler.py         # Data loading, cleaning, API access
│   ├── models.py               # Forecasting models
│   ├── visualizations.py       # Plotting utilities
│   └── recommendations.py      # Health advisory logic
│
└── notebooks/
    ├── 01_data_collection_POC.ipynb
    ├── 02_model_POC.ipynb
```

---

# Installation

```
pip3 install -r requirements.txt
python -m pip install --upgrade --extra-index-url https://PySimpleGUI.net/install PySimpleGUI
```

Systems with NVIDIA GPUs will automatically use CUDA for model training and inference when supported.

---

# Running the GUI

Ensure your dataset is located at:

```
data/epa_aqs_data_2025_cleaned.csv
```

Run the application:

```
python src/air_quality_gui.py
```

---

# Core Components

## 1. Data Handling (`src/data_handler.py`)
Handles:
- EPA API queries
- Real-time AirNow feeds
- Dataset loading (EPA, Kaggle)
- Preprocessing, validation, resampling
- AQI calculation from pollutant measurements

## 2. Forecasting Models (`src/models/`)
Includes:
- Baseline models
- ARIMA / SARIMA / Prophet
- LSTM / GRU / Transformer
- Model evaluation and tuning utilities

## 3. Visualizations (`src/visualizations.py`)
- Time series forecasting plots
- Model metric charts
- Full 5-panel analysis dashboard

## 4. Health Recommendations (`src/recommendations.py`)
AQI thresholds follow EPA standards:

```
0–50:      Good
51–100:    Moderate
101–150:   Unhealthy for Sensitive Groups
151–200:   Unhealthy
201–300:   Very Unhealthy
300+:      Hazardous
```

Personalization includes:
- Age-sensitive recommendations
- Respiratory condition severity
- Exposure risk assessment

---

# Report Format

1. Introduction  
2. Data Collection and Preprocessing  
3. Modeling Approaches  
4. Evaluation Metrics and Results  
5. Health Recommendation System  
6. GUI System and User Interaction  
7. Conclusion
