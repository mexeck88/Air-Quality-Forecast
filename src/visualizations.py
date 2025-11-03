# All visualization functions: EDA plots, model performance charts, recommendation displays
# Includes interactive dashboards and data exploration tools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta


def visualize_aqi_comparison(historic_data, aqs_data, city_focus, pollutant_focus):
    """
    Create a side-by-side comparison of AQI data from Kaggle and EPA AQS for a specific city and pollutant.
    
    Parameters:
    -----------
    historic_data : pd.DataFrame
        DataFrame containing Kaggle historic air quality data
    aqs_data : pd.DataFrame
        DataFrame containing EPA AQS data
    city_focus : str
        The city name to visualize (e.g., 'Los Angeles')
    pollutant_focus : str
        The pollutant name to visualize (e.g., 'PM2.5', 'O3')
    """
    
    print("=" * 70)
    print(f"DATA SOURCE COMPARISON: {pollutant_focus} in {city_focus}")
    print("=" * 70)

    # Create a figure with subplots for each source (2 graphs: Kaggle and EPA AQS)
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    axes_list = axes if isinstance(axes, np.ndarray) else [axes]

    import matplotlib.dates as mdates

    # ============================================================================
    # SOURCE 1: KAGGLE HISTORIC DATA (Wide format)
    # ============================================================================
    print("\n[1/2] KAGGLE HISTORIC DATA")
    print("-" * 70)
    kaggle_plotted = False
    if not historic_data.empty:
        try:
            # Find city column
            city_col_kaggle = None
            for col in ['City', 'city', 'CITY']:
                if col in historic_data.columns:
                    city_col_kaggle = col
                    break
            
            # Find date column
            date_col_kaggle = None
            for col in ['datetime', 'DateTime', 'date', 'Date', 'date_local']:
                if col in historic_data.columns:
                    date_col_kaggle = col
                    break
            
            if city_col_kaggle and date_col_kaggle and pollutant_focus in historic_data.columns:
                # Filter for focus city
                kaggle_city_data = historic_data[historic_data[city_col_kaggle] == city_focus].copy()
                
                # Get most recent month
                kaggle_city_data[date_col_kaggle] = pd.to_datetime(kaggle_city_data[date_col_kaggle])
                max_date = kaggle_city_data[date_col_kaggle].max()
                one_month_ago = max_date - timedelta(days=30)
                kaggle_recent = kaggle_city_data[
                    (kaggle_city_data[date_col_kaggle] >= one_month_ago) &
                    (kaggle_city_data[date_col_kaggle] <= max_date)
                ].copy()
                
                if not kaggle_recent.empty:
                    kaggle_recent = kaggle_recent.sort_values(date_col_kaggle)
                    
                    # Aggregate to one point per date (handle multiple measurements per day)
                    kaggle_recent['date_only'] = kaggle_recent[date_col_kaggle].dt.date
                    kaggle_daily = kaggle_recent.groupby('date_only')[pollutant_focus].mean().reset_index()
                    kaggle_daily.columns = ['date_only', 'aqi_value']
                    kaggle_daily['datetime'] = pd.to_datetime(kaggle_daily['date_only'])
                    
                    # Plot AQI values only
                    aqi_data = kaggle_daily['aqi_value'].dropna()
                    
                    if len(aqi_data) > 0:
                        ax = axes_list[0]
                        ax.plot(kaggle_daily['datetime'], aqi_data.values,
                               color='#1f77b4', marker='o', linestyle='-', linewidth=2.5,
                               markersize=6, alpha=0.8, label='Kaggle Historic (Daily Avg)')
                        
                        # Statistics
                        mean_val = aqi_data.mean()
                        std_val = aqi_data.std()
                        ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean: {mean_val:.1f}')
                        
                        ax.set_title(f'{pollutant_focus} AQI - Most Recent Month | Kaggle Historic Data (Daily Avg) | {city_focus}',
                                    fontsize=12, fontweight='bold')
                        ax.set_xlabel('Date', fontsize=11)
                        ax.set_ylabel(f'AQI', fontsize=11)
                        ax.legend(loc='best', fontsize=10)
                        ax.grid(True, alpha=0.3)
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                        
                        # Stats box
                        stats_text = f"Min: {aqi_data.min():.1f}\nMax: {aqi_data.max():.1f}\nMean: {mean_val:.1f}\nStd: {std_val:.1f}\nN: {len(aqi_data)}"
                        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                        
                        kaggle_plotted = True
                        print(f"✓ Kaggle AQI data: {len(kaggle_recent)} total measurements aggregated to {len(kaggle_daily)} daily points")
                        print(f"  Date range: {kaggle_daily['date_only'].min()} to {kaggle_daily['date_only'].max()}")
                        print(f"  Daily average AQI data points: {len(aqi_data)}")
        except Exception as e:
            print(f"✗ Error processing Kaggle data: {e}")
            import traceback
            traceback.print_exc()

    if not kaggle_plotted:
        axes_list[0].text(0.5, 0.5, 'No Kaggle AQI data available', ha='center', va='center', fontsize=12, color='red')
        axes_list[0].set_title(f'{pollutant_focus} AQI - Kaggle Historic Data', fontsize=12)

    # ============================================================================
    # SOURCE 2: EPA AQS DATA (Convert raw to AQI)
    # ============================================================================
    print("\n[2/2] EPA AQS DATA (RAW CONVERTED TO AQI)")
    print("-" * 70)
    aqs_plotted = False
    
    try:
        if aqs_data is not None and not aqs_data.empty:
            # Find location/city column
            location_cols = ['location_name', 'city', 'City', 'location']
            city_col_aqs = None
            for col in location_cols:
                if col in aqs_data.columns:
                    city_col_aqs = col
                    break
            
            if city_col_aqs:
                # Filter for focus city
                aqs_city_data = aqs_data[aqs_data[city_col_aqs] == city_focus].copy()
                
                if not aqs_city_data.empty:
                    # Get most recent month
                    aqs_city_data['datetime'] = pd.to_datetime(aqs_city_data['datetime'])
                    max_date = aqs_city_data['datetime'].max()
                    one_month_ago = max_date - timedelta(days=30)
                    aqs_recent = aqs_city_data[
                        (aqs_city_data['datetime'] >= one_month_ago) &
                        (aqs_city_data['datetime'] <= max_date)
                    ].copy()
                    
                    if not aqs_recent.empty:
                        aqs_recent = aqs_recent.sort_values('datetime')
                        
                        # Get AQI values (should already be converted)
                        value_col_name = None
                        if 'aqi' in aqs_recent.columns and aqs_recent['aqi'].notna().sum() > 0:
                            value_col_name = 'aqi'
                        elif 'value' in aqs_recent.columns and aqs_recent['value'].notna().sum() > 0:
                            value_col_name = 'value'
                        elif 'arithmetic_mean' in aqs_recent.columns and aqs_recent['arithmetic_mean'].notna().sum() > 0:
                            value_col_name = 'arithmetic_mean'
                        
                        if value_col_name is not None:
                            # Extract just the date (without time) and aggregate to one point per date
                            aqs_recent['date_only'] = aqs_recent['datetime'].dt.date
                            aqs_daily = aqs_recent.groupby('date_only')[value_col_name].mean().reset_index()
                            aqs_daily.columns = ['date_only', 'aqi_value']
                            aqs_daily['datetime'] = pd.to_datetime(aqs_daily['date_only'])
                            
                            if len(aqs_daily) > 0:
                                ax = axes_list[1]
                                aqi_vals = aqs_daily['aqi_value']
                                daily_dates = aqs_daily['datetime']
                                
                                ax.plot(daily_dates, aqi_vals.values,
                                       color='#ff7f0e', marker='s', linewidth=2.5,
                                       markersize=6, alpha=0.8, label='EPA AQS (Raw→AQI)')
                                
                                # Statistics
                                mean_val = aqi_vals.mean()
                                std_val = aqi_vals.std()
                                ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean: {mean_val:.1f}')
                                
                                ax.set_title(f'{pollutant_focus} AQI - Most Recent Month | EPA AQS Data (Daily Avg) | {city_focus}',
                                            fontsize=12, fontweight='bold')
                                ax.set_xlabel('Date', fontsize=11)
                                ax.set_ylabel(f'AQI', fontsize=11)
                                ax.legend(loc='best', fontsize=10)
                                ax.grid(True, alpha=0.3)
                                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                                
                                # Stats box
                                stats_text = f"Min: {aqi_vals.min():.1f}\nMax: {aqi_vals.max():.1f}\nMean: {mean_val:.1f}\nStd: {std_val:.1f}\nN: {len(aqi_vals)}"
                                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9, verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                                
                                aqs_plotted = True
                                print(f"✓ EPA AQS AQI data: {len(aqs_recent)} total measurements aggregated to {len(aqs_daily)} daily points")
                                print(f"  Date range: {aqs_daily['date_only'].min()} to {aqs_daily['date_only'].max()}")
                                print(f"  Daily average AQI data points: {len(aqs_daily)}")
                        else:
                            print(f"✗ No valid AQI column found in EPA data")
                    else:
                        print(f"✗ No EPA data found for {city_focus} in the recent month")
                else:
                    print(f"✗ No EPA data found for city: {city_focus}")
                    print(f"   Available cities in EPA data: {aqs_data[city_col_aqs].unique()[:10].tolist()}")
            else:
                print(f"✗ Could not find city column in EPA data")
                print(f"   Available columns: {aqs_data.columns.tolist()}")
        else:
            print(f"✗ EPA data (aqs_data) is not available or empty")
    except Exception as e:
        print(f"✗ Error processing EPA AQS data: {e}")
        import traceback
        traceback.print_exc()

    if not aqs_plotted:
        axes_list[1].text(0.5, 0.5, 'No EPA AQS AQI data available', ha='center', va='center', fontsize=12, color='red')
        axes_list[1].set_title(f'{pollutant_focus} AQI - EPA AQS Data', fontsize=12)

    plt.tight_layout()
    plt.show()


# Model comparison visualizations can be added here in the future