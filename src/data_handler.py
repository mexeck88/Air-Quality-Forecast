# ============================================================================
# Data Collection, Cleaning, and Preprocessing for Air Quality Data
# ============================================================================
# 
# PRIMARY DATA SOURCE: EPA AQS (Environmental Protection Agency Air Quality System)
#   - Official, verified air quality data
#   - Available data: 2023-2025 (cleaned daily aggregates)
#   - Pollutants: PM2.5, PM10, Ozone, NO2, SO2, CO
#   - Update frequency: 1-3 month delay from real-time
#   - Used for: Time series analysis, SARIMA/Prophet/LSTM modeling
#
# SUPPLEMENTARY DATA SOURCE: Kaggle 2016 Dataset
#   - Historical data for 2016 (Jan-May 2016)
#   - Pollutants: NO2, O3, SO2, CO (no PM2.5)
#   - Use case: Supplementary historical context only
#   - Status: Deprecated for core modeling (EPA data takes priority)
#
# REAL-TIME DATA: AirNow API
#   - Current air quality observations (few hours delay)
#   - Used for: Real-time monitoring and recommendations
#
# KEY FUNCTIONS:
#   - load_saved_epa_data(): Load EPA cleaned CSV data (PRIMARY)
#   - load_pollution_us_kaggle(): Load historical Kaggle data (supplementary)
#   - analyze_autocorrelation(): ACF/PACF analysis using EPA data
#   - extract_timeseries(): Flexible time series extraction (EPA priority)
#
# DATA STRUCTURE:
#   EPA format: [datetime, city, state, parameter_code, parameter, aqi, ...]
#   Kaggle format: [datetime, city, state, NO2_AQI, O3_AQI, SO2_AQI, CO_AQI]
#   Pollutant names are standardized during EPA data cleaning:
#     - "PM2.5 - Local Conditions" → "PM2.5"
#     - "Carbon monoxide" → "CO"
#     - "Nitrogen dioxide (NO2)" → "NO2"
#     - "Sulfur dioxide" → "SO2"
#     - "Ozone" → "Ozone"
#     - "PM10 Total 0-10um STP" → "PM10"
#
# ============================================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

import requests

# =================================================================
# API INITIALIZATION
#==================================================================
EPA_EMAIL = "matthew.eckert117@gmail.com"  # Your EPA AQS email
EPA_API_KEY = "sandheron75"  # Your EPA AQS key from registration email
EPA_BASE_URL = "https://aqs.epa.gov/data/api"

# AirNow API Configuration
AIRNOW_API_KEY = "F1BBFDBA-F113-4DD5-B113-99EFED624D60"  # Your AirNow API key
AIRNOW_BASE_URL = "https://www.airnowapi.org/aq/observation/zipCode/current/"

# Common pollutant parameter codes for EPA AQS
POLLUTANT_CODES = {
    'PM2.5': '88101',
    'PM10': '81102', 
    'Ozone': '44201',
    'NO2': '42602',
    'SO2': '42401',
    'CO': '42101'
}

# Major metropolitan areas for testing
TEST_LOCATIONS = {
    'Los Angeles': {'zip': '90210', 'state': '06', 'county': '037'},
    'New York': {'zip': '10001', 'state': '36', 'county': '061'},
    'Chicago': {'zip': '60601', 'state': '17', 'county': '031'},
    'Houston': {'zip': '77001', 'state': '48', 'county': '201'},
    'Phoenix': {'zip': '85001', 'state': '04', 'county': '013'},
    'Philadelphia': {'zip': '19019', 'state': '42', 'county': '101'},
    'Boston': {'zip': '02101', 'state': '25', 'county': '025'}
}



class LiveDataCollector:
    """Standalone class for collecting live air quality data."""
    
    def collect_airnow_data(self, zip_code, api_key, distance=25):
        """
        Collect current air quality data from AirNow API for a specific zip code.
        
        Parameters:
        -----------
        zip_code : str
            US zip code to query
        api_key : str
            AirNow API key
        distance : int
            Distance in miles from zip code (default 25)
        
        Returns:
        --------
        pd.DataFrame : Air quality data
        """
        url = "https://www.airnowapi.org/aq/observation/zipCode/current/"
        
        params = {
            'format': 'application/json',
            'zipCode': zip_code,
            'distance': distance,
            'API_KEY': api_key
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                print(f"No data returned for zip code {zip_code}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            df['zip_code'] = zip_code
            df['datetime'] = pd.to_datetime(df['DateObserved'] + ' ' + df['HourObserved'].astype(str) + ':00')
            
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching AirNow data for {zip_code}: {e}")
            return pd.DataFrame()
    
    def collect_airnow_data_multiple_locations(self, zip_codes, api_key, distance=25):
        """
        Collect AirNow data for multiple zip codes.
        
        Parameters:
        -----------
        zip_codes : list
            List of US zip codes
        api_key : str
            AirNow API key
        distance : int
            Distance in miles from zip code
        
        Returns:
        --------
        pd.DataFrame : Combined air quality data
        """
        all_data = []
        
        for zip_code in zip_codes:
            print(f"Fetching AirNow data for {zip_code}...")
            df = self.collect_airnow_data(zip_code, api_key, distance)
            if not df.empty:
                all_data.append(df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f" Successfully collected data for {len(all_data)} locations")
            return combined_df
        else:
            print("No data collected from AirNow API")
            return pd.DataFrame()
    
    # def collect_airnow_historical(self, zip_code, api_key, hours_back=24, distance=25):
    #     """
    #     Collect historical AirNow data for the last N hours.
        
    #     Uses the AirNow observation history endpoint to get data from the past
    #     hours with all available pollutants.
        
    #     Parameters:
    #     -----------
    #     zip_code : str
    #         US zip code to query
    #     api_key : str
    #         AirNow API key
    #     hours_back : int
    #         Number of hours of historical data to retrieve (default: 24)
    #     distance : int
    #         Distance in miles from zip code (default 25)
        
    #     Returns:
    #     --------
    #     pd.DataFrame : Air quality data for all pollutants over time period
    #     """
    #     # Calculate date range
    #     end_time = datetime.now()
    #     start_time = end_time - timedelta(hours=hours_back)
        
    #     print(f"\n  Fetching historical AirNow data:")
    #     print(f"    From: {start_time}")
    #     print(f"    To: {end_time}")
    #     print(f"    Hours: {hours_back}")
        
    #     # The history endpoint
    #     url = "https://www.airnowapi.org/aq/observation/zipCode/historical"
        
    #     params = {
    #         'zipCode': zip_code,
    #         'date': start_time.strftime('%Y-%m-%d'),
    #         'format': 'application/json',
    #         'API_KEY': api_key
    #     }
        
    #     try:
    #         response = requests.get(url, params=params)
    #         response.raise_for_status()
            
    #         # Debug: Check raw response
    #         print(f"  HTTP Status: {response.status_code}")
    #         print(f"  Response length: {len(response.text)} characters")
            
    #         if not response.text or response.text.strip() == '':
    #             print(f"  Empty response from API")
    #             return pd.DataFrame()
            
    #         # Try to parse JSON
    #         try:
    #             data = response.json()
    #         except ValueError as json_error:
    #             print(f"  JSON parse error: {json_error}")
    #             print(f"  First 200 chars of response: {response.text[:200]}")
    #             return pd.DataFrame()
            
    #         if not data or (isinstance(data, list) and len(data) == 0):
    #             print(f"  No historical data returned for zip code {zip_code}")
    #             return pd.DataFrame()
            
    #         # Convert to DataFrame
    #         df = pd.DataFrame(data)
    #         df['zip_code'] = zip_code
    #         df['datetime'] = pd.to_datetime(df['DateObserved'] + ' ' + df['HourObserved'].astype(str) + ':00')
            
    #         print(f"  Successfully retrieved {len(df)} historical records")
    #         print(f"  Unique pollutants: {df['ParameterName'].nunique()}")
    #         print(f"  Pollutants: {df['ParameterName'].unique().tolist()}")
            
    #         return df
            
    #     except requests.exceptions.RequestException as e:
    #         print(f"  Error fetching historical AirNow data for {zip_code}: {e}")
    #         return pd.DataFrame()
    
    def collect_aqs_data(self, state_code, county_code, parameter_code, 
                         api_key, api_email, begin_date=None, end_date=None, 
                         data_type='daily'):
        """
        Collect data from EPA AQS API.
        
        Parameters:
        -----------
        state_code : str
            2-digit state FIPS code
        county_code : str
            3-digit county FIPS code
        parameter_code : str
            EPA parameter code (e.g., '88101' for PM2.5)
        api_key : str
            EPA AQS API key
        api_email : str
            EPA AQS API email address
        begin_date : str
            Start date in YYYYMMDD format (default: 2023 data)
        end_date : str
            End date in YYYYMMDD format (default: 2023 data)
        data_type : str
            'daily', 'annual', or 'quarterly' (default: 'daily')
        
        Returns:
        --------
        pd.DataFrame : Air quality data
        """
        # Set default dates to 2023 (more likely to have complete data)
        if end_date is None:
            end_date = '20231231'
        if begin_date is None:
            begin_date = '20230101'
        
        # EPA AQS API requirement: end date must be in the same year as begin date
        # (except for annual data which uses only the year portion)
        if data_type != 'annual':
            begin_year = begin_date[:4]
            end_year = end_date[:4]
            if begin_year != end_year:
                print(f"Warning: Adjusting end_date to be in same year as begin_date ({begin_year})")
                end_date = f"{begin_year}1231"
        
        # Select appropriate endpoint based on data_type
        if data_type == 'annual':
            url = "https://aqs.epa.gov/data/api/annualData/byCounty"
        elif data_type == 'quarterly':
            url = "https://aqs.epa.gov/data/api/quarterlyData/byCounty"
        else:
            url = "https://aqs.epa.gov/data/api/dailyData/byCounty"
        
        params = {
            'email': api_email,
            'key': api_key,
            'param': parameter_code,
            'bdate': begin_date,
            'edate': end_date,
            'state': state_code,
            'county': county_code
        }
        
        try:
            print(f"  Querying {data_type} data: {begin_date} to {end_date}")
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            result = response.json()
            
            # Check for API status
            if 'Header' in result and len(result['Header']) > 0:
                status = result['Header'][0].get('status', 'Unknown')
                rows = result['Header'][0].get('rows', 0)
                
                if status.lower() == 'success' and rows > 0:
                    print(f"  ✓ API Status: {status} - {rows} rows")
                elif 'no data matched' in status.lower():
                    print(f"   API Status: {status}")
                    return pd.DataFrame()
                else:
                    print(f"   API Status: {status} - {rows} rows")
                    return pd.DataFrame()
            

            if 'Data' not in result or not result['Data']:
                print(f"  No data available in response")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(result['Data'])
            
            # Parse datetime based on data type
            if data_type == 'daily' and 'date_local' in df.columns:
                df['datetime'] = pd.to_datetime(df['date_local'])
            elif data_type == 'annual' and 'year' in df.columns:
                df['datetime'] = pd.to_datetime(df['year'], format='%Y')
            elif data_type == 'quarterly' and 'year' in df.columns and 'quarter' in df.columns:
                # Create quarter start date
                df['datetime'] = pd.to_datetime(df['year'].astype(str) + '-' + 
                                                ((df['quarter'].astype(int) - 1) * 3 + 1).astype(str) + '-01')
            
            print(f"  Retrieved {len(df)} measurements")
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"  HTTP Error: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"  Error processing AQS data: {e}")
            return pd.DataFrame()
    
    def collect_aqs_data_multiple_locations(self, locations, parameter_code, 
                                           api_key, api_email, begin_date=None, end_date=None):
        """
        Collect AQS data for multiple locations.
        
        Parameters:
        -----------
        locations : dict
            Dictionary with location info: {'name': {'state': 'XX', 'county': 'XXX'}}
        parameter_code : str
            EPA parameter code
        api_key : str
            EPA AQS API key
        begin_date : str
            Start date in YYYYMMDD format
        end_date : str
            End date in YYYYMMDD format
        
        Returns:
        --------
        pd.DataFrame : Combined air quality data
        """
        all_data = []
        
        for location_name, location_info in locations.items():
            print(f"Fetching AQS data for {location_name}...")
            df = self.collect_aqs_data(
                state_code=location_info['state'],
                county_code=location_info['county'],
                parameter_code=parameter_code,
                api_key=api_key,
                api_email=api_email,
                begin_date=begin_date,
                end_date=end_date
            )
            if not df.empty:
                df['location_name'] = location_name
                all_data.append(df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f" Successfully collected AQS data for {len(all_data)} locations")
            return combined_df
        else:
            print("No data collected from AQS API")
            return pd.DataFrame()


class DataProcessor:
    """Standalone class for processing air quality data."""
    
    def clean_data(self, df):
        """Clean and validate air quality data."""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Remove null values
        df = df.dropna()
        
        # Ensure datetime is properly formatted
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        
        return df
    
    def standardize_units(self, df):
        """Standardize pollutant units."""
        # This would contain unit conversion logic
        return df
    
    def standardize_airnow_data(self, df):
        """Standardize AirNow data to common format."""
        if df.empty:
            return df
        
        # Extract category name from the nested dictionary
        category_name = df['Category'].apply(lambda x: x.get('Name') if isinstance(x, dict) else 'Unknown')
        
        standardized = pd.DataFrame({
            'datetime': df['datetime'],
            'location': df['ReportingArea'],
            'state_code': df['StateCode'],
            'zip_code': df['zip_code'],
            'parameter': df['ParameterName'].str.lower().str.replace('.', '', regex=False),
            'aqi': df['AQI'],
            'category': category_name,
            'latitude': df['Latitude'],
            'longitude': df['Longitude'],
            'source': 'AirNow'
        })
        
        return standardized
    
    def standardize_aqs_data(self, df):
        """Standardize AQS data to common format."""
        if df.empty:
            return df
        
        standardized = pd.DataFrame({
            'datetime': df['datetime'],
            'location': df['local_site_name'] if 'local_site_name' in df.columns else df.get('site_number', 'Unknown'),
            'state_code': df['state_code'],
            'county_code': df['county_code'],
            'site_num': df['site_number'],
            'parameter_code': df['parameter_code'],
            'parameter': df['parameter'],
            'value': df['arithmetic_mean'],  # Main daily average value
            'value_max': df['first_max_value'],  # Peak value for the day
            'max_hour': df['first_max_hour'],  # Hour when peak occurred
            'unit': df['units_of_measure'],
            'latitude': df['latitude'],
            'longitude': df['longitude'],
            'city': df['city'] if 'city' in df.columns else 'Unknown',
            'aqi': df['aqi'] if 'aqi' in df.columns else None,
            'source': 'EPA_AQS'
        })
        
        return standardized


class ExploratoryAnalysis:
    """Standalone class for exploratory data analysis."""
    
    def generate_summary_stats(self, df):
        """Generate summary statistics for air quality data."""
        return df.describe()
    
    def plot_time_series(self, df, pollutant, city=None):
        """Plot time series for a specific pollutant."""
        plt.figure(figsize=(12, 6))
        
        if city:
            data = df[df['city'] == city]
            plt.plot(data['datetime'], data['value'])
            plt.title(f'{pollutant} Time Series - {city}')
        else:
            for city_name in df['city'].unique():
                city_data = df[df['city'] == city_name]
                plt.plot(city_data['datetime'], city_data['value'], label=city_name)
            plt.legend()
            plt.title(f'{pollutant} Time Series - All Cities')
        
        plt.xlabel('Date')
        plt.ylabel('Concentration')
        plt.grid(True, alpha=0.3)
        plt.show()


# Initialize components
live_collector = LiveDataCollector()
data_processor = DataProcessor()
analyzer = ExploratoryAnalysis()


def load_kaggle_data(file_path):
    """
    Load historic air quality data from Kaggle dataset.
    Converts raw pollutant concentrations to AQI using EPA breakpoint tables.
    
    Parameters:
    -----------
    file_path : str
        Path to the Kaggle CSV file
        
    Returns:
    --------
    pd.DataFrame : Historic air quality data with pollutants converted to AQI scale (0-500)
    
    Notes:
    ------
    Kaggle data contains raw concentrations in standard units:
    - PM2.5, PM10: µg/m³
    - O3, NO2, SO2, CO: ppm/ppb
    
    These are converted to EPA AQI (0-500 scale) for consistency with EPA data.
    """
    try:
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        
        print(f"Successfully loaded {len(df)} records from Kaggle dataset")
        print(f"Columns: {df.columns.tolist()}\n")
        
        # Map Kaggle pollutant columns to EPA parameter codes
        pollutant_mapping = {
            'PM2.5': '88101',
            'PM10': '81102',
            'O3': '44201',
            'NO2': '42602',
            'SO2': '42401',
            'CO': '42101'
        }
        
        # Convert each pollutant to AQI
        for pollutant, param_code in pollutant_mapping.items():
            if pollutant in df.columns:
                # Create temporary DataFrame for conversion
                temp_df = pd.DataFrame({
                    'parameter_code': param_code,
                    'arithmetic_mean': df[pollutant],
                    'aqi': np.nan
                })
                
                # Convert to AQI using EPA breakpoints
                converted = convert_EPA_raw_to_aqi(temp_df)
                
                # Replace raw values with AQI values
                count_converted = converted['aqi'].notna().sum()
                df[pollutant] = converted['aqi']
                print(f"✓ {pollutant}: {count_converted} values converted to AQI")
        
        print(f"\n Data loading complete. All pollutants now in AQI scale (0-500)")
        return df
        
    except FileNotFoundError:
        print(f" File not found: {file_path}")
        return None
    except Exception as e:
        print(f" Error loading file: {e}")
        import traceback
        traceback.print_exc()
        return None

def _get_pollutant_units(pollutant):
    """Get units for pollutant."""
    units_map = {
        'pm25': 'µg/m³', 'pm10': 'µg/m³', 'ozone': 'ppm',
        'no2': 'ppb', 'so2': 'ppb', 'co': 'ppm'
    }
    return units_map.get(pollutant, 'unknown')


def _get_city_coords(city):
    """Get coordinates for city."""
    coords = {
        'Los Angeles': (34.0522, -118.2437),
        'New York': (40.7128, -74.0060),
        'Chicago': (41.8781, -87.6298),
        'Houston': (29.7604, -95.3698),
        'Phoenix': (33.4484, -112.0740),
        'Philadelphia': (39.9526, -75.1652),
        'Boston': (42.3601, -71.0589)
    }
    return coords.get(city, (0, 0))


def load_saved_epa_data(filename):
    """
    Load previously saved EPA AQS data from CSV file.
    
    The EPA data has already been cleaned and aggregated to daily values.
    Pollutant names have been standardized during cleaning:
      - "PM2.5 - Local Conditions" → "PM2.5"
      - "Carbon monoxide" → "CO"
      - "Nitrogen dioxide (NO2)" → "NO2"
      - "Ozone" → "Ozone"
      - "PM10 Total 0-10um STP" → "PM10"
      - "Sulfur dioxide" → "SO2"
    
    This function loads and validates the data.
    
    Parameters:
    -----------
    filename : str
        Path to EPA AQS CSV file (already cleaned with standardized names)
        
    Returns:
    --------
    pd.DataFrame : EPA AQS data with columns:
        - datetime: Date and time
        - city: City name
        - state: State
        - parameter_code: EPA pollutant parameter code
        - parameter: Standardized pollutant name (e.g., 'PM2.5', 'Ozone')
        - arithmetic_mean: Average concentration value
        - aqi: AQI value (0-500)
        - observation_count: Number of observations aggregated
        - units_of_measure: Unit of measurement
    """
    try:
        df = pd.read_csv(filename)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Sort by date
        df = df.sort_values('datetime').reset_index(drop=True)
        
        print(f"Loaded EPA AQS data: {len(df)} records")
        print(f"Date range: {df['datetime'].min().date()} to {df['datetime'].max().date()}")
        
        # Summary statistics
        print(f"\nEPA Data Summary:")
        print(f"  Unique cities: {df['city'].nunique()}")
        print(f"  Unique pollutants: {df['parameter_code'].nunique()}")
        print(f"  Unique dates: {df['datetime'].nunique()}")
        print(f"  Total records: {len(df)}")
        
        # Display available pollutants
        pollutants = df['parameter'].unique()
        print(f"  Available pollutants: {list(pollutants)}")
        
        return df
        
    except FileNotFoundError:
        print(f"[ERROR] File not found: {filename}")
        print(f"Make sure the EPA data file has been downloaded and cleaned.")
        return pd.DataFrame()
    except Exception as e:
        print(f"[ERROR] Error loading file: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
    

def convert_EPA_raw_to_aqi(df):
    """
    Convert raw EPA pollutant measurements to AQI values using EPA breakpoint tables.
    
    The EPA calculates AQI using pollutant-specific breakpoint concentrations.
    This function applies the standard EPA AQI calculation formula.
    
    Formula: AQI = ((AQI_high - AQI_low) / (BP_high - BP_low)) * (concentration - BP_low) + AQI_low
    
    Where:
    - AQI_high/low: AQI breakpoints for the corresponding category
    - BP_high/low: Breakpoint concentrations for the pollutant
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing EPA pollutant data with columns:
        - 'parameter_code': EPA parameter code (e.g., '88101' for PM2.5)
        - 'value': Pollutant concentration value
        - Optionally 'unit': Unit of measurement
        
    Returns:
    --------
    pd.DataFrame : DataFrame with added 'aqi' column
    
    Pollutant codes:
    - '88101': PM2.5 (µg/m³, 24-hr avg)
    - '81102': PM10 (µg/m³, 24-hr avg)
    - '44201': Ozone (ppm, 8-hr avg)
    - '42602': NO2 (ppb, 1-hr avg)
    - '42401': SO2 (ppb, 1-hr avg)
    - '42101': CO (ppm, 8-hr avg)
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # EPA AQI Breakpoint Tables
    # Format: 'parameter_code': [(concentration_breakpoint_low, concentration_breakpoint_high, aqi_low, aqi_high), ...]
    
    BREAKPOINT_TABLES = {
        # PM2.5 (24-hour average, µg/m³)
        '88101': [
            (0.0, 12.0, 0, 50),
            (12.1, 35.4, 51, 100),
            (35.5, 55.4, 101, 150),
            (55.5, 150.4, 151, 200),
            (150.5, 250.4, 201, 300),
            (250.5, float('inf'), 301, 500),
        ],
        # PM10 (24-hour average, µg/m³)
        '81102': [
            (0.0, 54.0, 0, 50),
            (54.1, 154.0, 51, 100),
            (154.1, 254.0, 101, 150),
            (254.1, 354.0, 151, 200),
            (354.1, 424.0, 201, 300),
            (424.1, float('inf'), 301, 500),
        ],
        # O3 (8-hour average, ppm)
        '44201': [
            (0.000, 0.054, 0, 50),
            (0.055, 0.070, 51, 100),
            (0.071, 0.085, 101, 150),
            (0.086, 0.105, 151, 200),
            (0.106, 0.200, 201, 300),
            (0.201, float('inf'), 301, 500),
        ],
        # NO2 (1-hour average, ppb)
        '42602': [
            (0.0, 53.0, 0, 50),
            (54.0, 100.0, 51, 100),
            (101.0, 360.0, 101, 150),
            (361.0, 649.0, 151, 200),
            (650.0, 1249.0, 201, 300),
            (1250.0, float('inf'), 301, 500),
        ],
        # SO2 (1-hour average, ppb)
        '42401': [
            (0.0, 35.0, 0, 50),
            (36.0, 75.0, 51, 100),
            (76.0, 185.0, 101, 150),
            (186.0, 304.0, 151, 200),
            (305.0, 604.0, 201, 300),
            (605.0, float('inf'), 301, 500),
        ],
        # CO (8-hour average, ppm)
        '42101': [
            (0.0, 4.4, 0, 50),
            (4.5, 9.4, 51, 100),
            (9.5, 12.4, 101, 150),
            (12.5, 15.4, 151, 200),
            (15.5, 30.4, 201, 300),
            (30.5, float('inf'), 301, 500),
        ],
    }
    
    def calculate_aqi(param_code, concentration):
        """Calculate AQI for a given parameter code and concentration."""
        if pd.isna(concentration) or concentration < 0:
            return np.nan
        
        if param_code not in BREAKPOINT_TABLES:
            # Unknown parameter code
            return np.nan
        
        breakpoints = BREAKPOINT_TABLES[param_code]
        
        # Find the appropriate breakpoint range
        for bp_low, bp_high, aqi_low, aqi_high in breakpoints:
            if bp_low <= concentration <= bp_high:
                # Apply AQI formula
                if bp_high == bp_low:  # Avoid division by zero
                    return aqi_low
                
                aqi = ((aqi_high - aqi_low) / (bp_high - bp_low)) * (concentration - bp_low) + aqi_low
                return round(aqi, 1)
        
        # Concentration exceeds highest breakpoint
        return np.nan
    
    # Initialize aqi column if it doesn't exist
    if 'aqi' not in df.columns:
        df['aqi'] = np.nan
    
    # Apply conversion only to rows missing AQI data
    # Find rows with missing AQI
    missing_aqi_mask = df['aqi'].isna()
    
    # Get parameter code column (might be named differently)
    param_col = None
    for col in ['parameter_code', 'ParameterCode', 'parameter', 'Parameter']:
        if col in df.columns:
            param_col = col
            break
    
    # Get value column - try different variations
    value_col = None
    for col in ['arithmetic_mean', 'Arithmetic_Mean', 'value', 'Value', 'mean_value', 'Mean_Value']:
        if col in df.columns:
            value_col = col
            break
    
    if param_col is None:
        print(f"Warning: Could not find parameter_code column for AQI conversion")
        print(f"Available columns: {df.columns.tolist()}")
        return df
    
    if value_col is None:
        print(f"Warning: Could not find value column for AQI conversion")
        print(f"Looked for: arithmetic_mean, value, mean_value")
        print(f"Available columns: {df.columns.tolist()}")
        return df
    
    # Calculate AQI for rows with missing values
    if missing_aqi_mask.sum() > 0:
        df.loc[missing_aqi_mask, 'aqi'] = df.loc[missing_aqi_mask].apply(
            lambda row: calculate_aqi(row[param_col], row[value_col]),
            axis=1
        )
        print(f"  Calculated AQI for {missing_aqi_mask.sum()} rows with missing values")
        print(f"  Used parameter column: '{param_col}'")
        print(f"  Used value column: '{value_col}'")
    else:
        print("All rows already have AQI values")
    
    return df


# ================================================================
# AUTOCORRELATION ANALYSIS FUNCTION AND HYPERPARAMETERS
# ================================================================

try:
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.stattools import acf, pacf
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "statsmodels"])
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.stattools import acf, pacf

def analyze_autocorrelation(epa_data, city_name, pollutant_name, nlags=30):
    """
    Compute and visualize autocorrelation (ACF) and partial autocorrelation (PACF)
    for a specific pollutant in a city using EPA cleaned data.
    
    Parameters:
    -----------
    epa_data : pd.DataFrame
        EPA cleaned data with columns: datetime, city, parameter, aqi
    city_name : str
        City to analyze
    pollutant_name : str
        Pollutant to analyze (e.g., 'PM2.5', 'Ozone', 'NO2')
    nlags : int
        Number of lags to display (default 30)
    
    Returns:
    --------
    dict : Dictionary containing ACF and PACF values
    """
    print(f"\n{'='*70}")
    print(f"ANALYZING: {pollutant_name} in {city_name}")
    print(f"{'='*70}")
    
    results = {}
    
    try:
        # Filter data for the city
        city_data = epa_data[epa_data['city'] == city_name].copy()
        
        if city_data.empty:
            print(f" No data found for city: {city_name}")
            print(f"  Available cities: {epa_data['city'].unique()}")
            return results
        
        # Filter for the specific pollutant
        # EPA data has 'parameter' column with full names
        pollutant_filter_map = {
            'PM2.5': ['PM2.5', 'PM25', 'PM2.5 - Local Conditions'],
            'Ozone': ['Ozone', 'O3'],
            'NO2': ['NO2', 'Nitrogen Dioxide'],
            'SO2': ['SO2', 'Sulfur Dioxide'],
            'CO': ['CO', 'Carbon Monoxide'],
        }
        
        if pollutant_name not in pollutant_filter_map:
            print(f" Pollutant '{pollutant_name}' not recognized")
            print(f"  Available: {list(pollutant_filter_map.keys())}")
            return results
        
        # Filter to the specific pollutant
        city_data = city_data[city_data['parameter'].str.contains('|'.join(pollutant_filter_map[pollutant_name]), case=False, na=False)]
        
        if city_data.empty:
            print(f" Pollutant '{pollutant_name}' not found for {city_name}")
            print(f"  Available pollutants: {epa_data[epa_data['city'] == city_name]['parameter'].unique()}")
            return results
        
        # Sort by date and prepare time series
        city_data['datetime'] = pd.to_datetime(city_data['datetime'])
        city_data = city_data.sort_values('datetime')
        
        # Aggregate to daily averages (EPA might have multiple values per day per pollutant)
        city_data['date_only'] = city_data['datetime'].dt.date
        daily_data = city_data.groupby('date_only')['aqi'].mean()
        
        # Remove NaN values
        ts_data = daily_data.dropna()
        
        if len(ts_data) < nlags + 1:
            print(f"Not enough data points ({len(ts_data)}) for {nlags} lags. Using {len(ts_data)//2} lags instead.")
            nlags = len(ts_data) // 2
        
        print(f"✓ Time series prepared: {len(ts_data)} daily observations")
        print(f"  Date range: {daily_data.index.min()} to {daily_data.index.max()}")
        print(f"  Mean: {ts_data.mean():.2f} | Std: {ts_data.std():.2f}")
        
        # Compute ACF and PACF
        acf_values = acf(ts_data, nlags=nlags, fft=True)
        pacf_values = pacf(ts_data, nlags=nlags, method='ywmle')
        
        results = {
            'ts_data': ts_data,
            'acf_values': acf_values,
            'pacf_values': pacf_values,
            'nlags': nlags,
            'city': city_name,
            'pollutant': pollutant_name
        }
        
        # Print interpretation
        print(f"\n ACF & PACF INTERPRETATION:")
        print(f"-" * 70)
        
        # Find significant lags (beyond confidence interval)
        confidence_interval = 1.96 / np.sqrt(len(ts_data))
        print(f"  Confidence Interval: ±{confidence_interval:.3f}")
        
        significant_acf = np.where(np.abs(acf_values[1:]) > confidence_interval)[0] + 1
        significant_pacf = np.where(np.abs(pacf_values[1:]) > confidence_interval)[0] + 1
        
        print(f"\n  Significant ACF lags (first 5): {significant_acf[:5].tolist()}")
        print(f"  Significant PACF lags (first 5): {significant_pacf[:5].tolist()}")
        
        if len(significant_acf) > 0:
            print(f"  → Strong autocorrelation suggests {pollutant_name} is PERSISTENT")
            print(f"    (Today's value strongly predicts tomorrow's value)")
        else:
            print(f"  → Weak autocorrelation suggests {pollutant_name} is RANDOM")
            print(f"    (Today's value has little predictive power for tomorrow)")
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # ACF plot
        plot_acf(ts_data, lags=nlags, ax=axes[0], alpha=0.05)
        axes[0].set_title(f'Autocorrelation (ACF) - {pollutant_name} in {city_name}', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Lag (days)')
        axes[0].set_ylabel('ACF')
        axes[0].grid(True, alpha=0.3)
        
        # PACF plot
        plot_pacf(ts_data, lags=nlags, ax=axes[1], alpha=0.05, method='ywmle')
        axes[1].set_title(f'Partial Autocorrelation (PACF) - {pollutant_name} in {city_name}', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Lag (days)')
        axes[1].set_ylabel('PACF')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f" Error in autocorrelation analysis: {e}")
        import traceback
        traceback.print_exc()
    
    return results


def determine_sarima_parameters(ts_data, nlags=30):
    """
    Determine SARIMA(p,d,q)(P,D,Q,s) parameters from ACF/PACF analysis.
    Only outputs the recommended parameters to try.
    """
    from statsmodels.tsa.stattools import adfuller
    
    ts_data = ts_data.dropna()
    
    if len(ts_data) < nlags + 1:
        nlags = len(ts_data) // 2
    
    # Compute ACF and PACF
    acf_values = acf(ts_data, nlags=nlags, fft=True)
    pacf_values = pacf(ts_data, nlags=nlags, method='ywmle')
    
    # Calculate confidence interval
    confidence_interval = 1.96 / np.sqrt(len(ts_data))
    
    # Find significant lags
    significant_acf = np.where(np.abs(acf_values[1:]) > confidence_interval)[0] + 1
    significant_pacf = np.where(np.abs(pacf_values[1:]) > confidence_interval)[0] + 1
    
    # =====================================================================
    # Determine d (differencing)
    # =====================================================================
    adf_result = adfuller(ts_data, autolag='AIC')
    d = 0 if adf_result[1] < 0.05 else 1
    
    # =====================================================================
    # Determine p from PACF (first significant lag)
    # =====================================================================
    p = 1
    for lag in range(1, min(6, len(significant_pacf) + 1)):
        if lag in significant_pacf:
            p = lag
        else:
            break
    
    # =====================================================================
    # Determine q from ACF (first significant lag)
    # =====================================================================
    q = 1
    for lag in range(1, min(6, len(significant_acf) + 1)):
        if lag in significant_acf:
            q = lag
        else:
            break
    
    # =====================================================================
    # Detect seasonal parameters - CHECK FOR EXACT LAGS (22, 23 DAY CYCLES)
    # =====================================================================
    s_candidates = {}
    
    # Check for seasonal spikes at specific lags: 7, 22, 23, 30
    for season_lag in range(1,31):
        if season_lag > nlags:
            continue
        
        # Check if this exact lag has a significant spike
        if season_lag in significant_acf or season_lag in significant_pacf:
            s_candidates[season_lag] = 1
        
        # Also check for multiples of this lag (e.g., 46, 69 for 23-day cycle)
        for multiple in range(2, 4):
            multiple_lag = season_lag * multiple
            if multiple_lag > nlags:
                break
            if multiple_lag in significant_acf or multiple_lag in significant_pacf:
                s_candidates[season_lag] = s_candidates.get(season_lag, 0) + 1
    
    # Use strongest seasonal period
    if s_candidates:
        s = max(s_candidates, key=s_candidates.get)
        P, D, Q = 1, 0, 1
    else:
        s = 1
        P, D, Q = 0, 0, 0
    
    # =====================================================================
    # Return recommended models
    # =====================================================================
    print("\n" + "="*70)
    print("RECOMMENDED SARIMA MODELS TO TEST")
    print("="*70)
    
    models = [
        ((p, d, q), (0, 0, 0, 1)),
        ((p, d, q), (P, D, Q, s)),
        ((p+1, d, q), (P, D, Q, s)),
        ((p, d, q+1), (P, D, Q, s)),
        ((p+1, d, q+1), (P, D, Q, s)),
    ]
    
    print(f"\nTop candidates (in order to try):\n")
    for i, (order, seasonal_order) in enumerate(models, 1):
        print(f"{i}. SARIMA{order}{seasonal_order}")
    
    print(f"\nSeasonal period detected: {s} days")
    print(f"Non-seasonal: p={p}, d={d}, q={q}")
    if s != 1:
        print(f"Seasonal: P={P}, D={D}, Q={Q}, s={s}")
    
    print("="*70 + "\n")
    
    return models


def extract_timeseries(historic_data, aqs_data, city, pollutant, combine_sources=True):
    """
    Extract and prepare a time series for modeling using EPA AQS data only.
    
    Parameters:
    -----------
    historic_data : pd.DataFrame
        (Deprecated - kept for backward compatibility but not used)
    aqs_data : pd.DataFrame
        EPA AQS data (columns: datetime, city, parameter, aqi)
        Primary and only data source.
    city : str
        City name
    pollutant : str
        Pollutant name (e.g., 'PM2.5', 'Ozone', 'NO2', 'SO2', 'CO', 'PM10')
    combine_sources : bool
        (Deprecated - no longer used, EPA data is always the only source)
    
    Returns:
    --------
    pd.Series : Time series sorted by date with NaNs removed
    """
    
    # Pollutant name mapping for EPA data (with standardized clean names)
    pollutant_filter_map = {
        'PM2.5': ['PM2.5', 'PM25', 'PM2.5 - Local Conditions'],
        'Ozone': ['Ozone', 'O3'],
        'NO2': ['NO2', 'Nitrogen Dioxide'],
        'SO2': ['SO2', 'Sulfur Dioxide'],
        'CO': ['CO', 'Carbon Monoxide'],
        'PM10': ['PM10', 'PM10 Total 0-10um STP'],
    }
    
    # Extract from EPA AQS data (only source)
    if aqs_data.empty:
        print(f" No EPA data available")
        return pd.Series(dtype=float)
    
    # Filter data for the specified city
    aqs_city = aqs_data[aqs_data['city'] == city].copy()
    
    if aqs_city.empty:
        print(f" No EPA data found for city: {city}")
        print(f"  Available cities: {aqs_data['city'].unique().tolist()}")
        return pd.Series(dtype=float)
    
    # Filter by pollutant using standardized names
    if pollutant not in pollutant_filter_map:
        print(f" Pollutant '{pollutant}' not recognized")
        print(f"  Available: {list(pollutant_filter_map.keys())}")
        return pd.Series(dtype=float)
    
    # Filter to the specific pollutant
    pollutant_variations = pollutant_filter_map[pollutant]
    aqs_filtered = aqs_city[aqs_city['parameter'].str.contains('|'.join(pollutant_variations), case=False, na=False)]
    
    if aqs_filtered.empty:
        print(f" {pollutant} not found in EPA data for {city}")
        print(f"  Available pollutants: {aqs_city['parameter'].unique().tolist()}")
        return pd.Series(dtype=float)
    
    # Find and convert date column
    date_col = None
    for col in ['datetime', 'DateTime', 'date', 'Date', 'date_local']:
        if col in aqs_filtered.columns:
            date_col = col
            break
    
    if date_col is None:
        print(f" Could not find date column in EPA data")
        print(f"  Available columns: {aqs_filtered.columns.tolist()}")
        return pd.Series(dtype=float)
    
    # Convert to datetime and aggregate to daily averages
    aqs_filtered[date_col] = pd.to_datetime(aqs_filtered[date_col])
    aqs_filtered['date_only'] = aqs_filtered[date_col].dt.date
    
    # Group by date and calculate daily mean AQI
    daily_aqi = aqs_filtered.groupby('date_only')['aqi'].mean()
    
    # Remove NaN values and sort
    ts_data = daily_aqi.dropna().sort_index()
    
    if len(ts_data) == 0:
        print(f" No valid data after aggregation for {city} - {pollutant}")
        return pd.Series(dtype=float)
    
    print(f"EPA data extracted: {len(ts_data)} daily observations for {pollutant} in {city}")
    print(f"  Date range: {ts_data.index.min()} to {ts_data.index.max()}")
    print(f"  AQI range: {ts_data.min():.1f} to {ts_data.max():.1f}")
    
    return ts_data


# =================================================================
# LOAD NEW KAGGLE DATASET: pollution_us_2000_2016.csv
# =================================================================

def load_pollution_us_kaggle(file_path, year=2016, city_filter=None, state_filter=None):
    """
    Load US pollution data from Kaggle (2000-2016 dataset).
    
    File has multiple measurements per date per site.
    This function aggregates to one value per date per city.
    
    Parameters:
    -----------
    file_path : str
        Path to pollution_us_2000_2016.csv
    year : int, optional
        Year to load (default 2016). Set to None to load all years.
    city_filter : str or list, optional
        Filter to specific city/cities. If None, loads all cities.
    state_filter : str or list, optional
        Filter to specific state/states. If None, loads all states.
        
    Returns:
    --------
    pd.DataFrame : Standardized data with columns:
        - datetime: Date
        - city: City name
        - state: State name
        - NO2_AQI: NO2 AQI value
        - O3_AQI: Ozone AQI value
        - SO2_AQI: SO2 AQI value
        - CO_AQI: CO AQI value
    """
    try:
        print("="*80)
        print(f"LOADING KAGGLE POLLUTION DATA ({year if year else '2000-2016'})")
        print("="*80)
        
        # Load raw data
        print(f"\nReading {file_path}...")
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} raw records")
        
        # Parse date
        df['datetime'] = pd.to_datetime(df['Date Local'])
        print(f"  Original date range: {df['datetime'].min().date()} to {df['datetime'].max().date()}")
        
        # Filter by year if specified
        if year is not None:
            df['year'] = df['datetime'].dt.year
            df = df[df['year'] == year]
            print(f"  Filtered to year: {year}")
        
        # Apply city filter
        if city_filter is not None:
            if isinstance(city_filter, str):
                city_filter = [city_filter]
            df = df[df['City'].isin(city_filter)]
            print(f"  Filtered to cities: {city_filter}")
        
        # Apply state filter
        if state_filter is not None:
            if isinstance(state_filter, str):
                state_filter = [state_filter]
            df = df[df['State'].isin(state_filter)]
            print(f"  Filtered to states: {state_filter}")
        
        print(f"After filtering: {len(df)} records")
        
        # Aggregate to one value per date per city
        # Multiple measurements per date per site → average them
        print(f"\nAggregating multiple measurements per date...")
        
        agg_dict = {
            'NO2_AQI': ('NO2 AQI', 'mean'),
            'O3_AQI': ('O3 AQI', 'mean'),
            'SO2_AQI': ('SO2 AQI', 'mean'),
            'CO_AQI': ('CO AQI', 'mean'),
        }
        
        cleaned = df.groupby(['datetime', 'City', 'State']).agg(**agg_dict).reset_index()
        cleaned.columns = ['datetime', 'city', 'state', 'NO2_AQI', 'O3_AQI', 'SO2_AQI', 'CO_AQI']
        
        # Sort by date
        cleaned = cleaned.sort_values('datetime').reset_index(drop=True)
        
        print(f"✓ Aggregated to {len(cleaned)} unique (date, city) records")
        print(f"\nData Summary:")
        print(f"  Unique cities: {cleaned['city'].nunique()}")
        print(f"  Unique states: {cleaned['state'].nunique()}")
        print(f"  Date range: {cleaned['datetime'].min().date()} to {cleaned['datetime'].max().date()}")
        print(f"  Total records: {len(cleaned)}")
        
        print(f"\nAQI Statistics:")
        for col in ['NO2_AQI', 'O3_AQI', 'SO2_AQI', 'CO_AQI']:
            vals = cleaned[col].dropna()
            if len(vals) > 0:
                print(f"  {col}: Min={vals.min():.1f}, Mean={vals.mean():.1f}, Max={vals.max():.1f} (n={len(vals)})")
        
        print(f"\n✓ Data loaded successfully!")
        return cleaned
        
    except FileNotFoundError:
        print(f"[ERROR] File not found: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"[ERROR] Error loading file: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def get_kaggle_city_data(kaggle_df, city_name):
    """
    Extract time series for a specific city from Kaggle data.
    
    Parameters:
    -----------
    kaggle_df : pd.DataFrame
        DataFrame from load_pollution_us_kaggle()
    city_name : str
        Name of city to extract
        
    Returns:
    --------
    pd.DataFrame : Data for specified city, sorted by date
    """
    city_data = kaggle_df[kaggle_df['city'] == city_name].copy()
    return city_data.sort_values('datetime').reset_index(drop=True)