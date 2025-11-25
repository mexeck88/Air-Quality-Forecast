# All visualization functions: EDA plots, model performance charts, recommendation displays
# Includes interactive dashboards and data exploration tools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from math import sqrt



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
                            print(f" No valid AQI column found in EPA data")
                    else:
                        print(f" No EPA data found for {city_focus} in the recent month")
                else:
                    print(f"No EPA data found for city: {city_focus}")
                    print(f"   Available cities in EPA data: {aqs_data[city_col_aqs].unique()[:10].tolist()}")
            else:
                print(f" Could not find city column in EPA data")
                print(f"   Available columns: {aqs_data.columns.tolist()}")
        else:
            print(f" EPA data (aqs_data) is not available or empty")
    except Exception as e:
        print(f" Error processing EPA AQS data: {e}")
        import traceback
        traceback.print_exc()

    if not aqs_plotted:
        axes_list[1].text(0.5, 0.5, 'No EPA AQS AQI data available', ha='center', va='center', fontsize=12, color='red')
        axes_list[1].set_title(f'{pollutant_focus} AQI - EPA AQS Data', fontsize=12)

    plt.tight_layout()
    plt.show()


# =============================================================================
# PREDICTION VISUALIZATIONS
# =============================================================================

def plot_3day_forecast(live_value, predictions_3day, pollutant="AQI", 
                       title="3-Day Forecast from Live Observation", figsize=(12, 6)):
    """
    Plot single live data point with 3-day dotted forecast.
    
    Perfect for real-time user-facing displays. Shows today's observation
    as a solid blue point, followed by 3-day forecast as dotted line.
    
    Features:
    - Automatic trend analysis (WORSENING/IMPROVING/STABLE)
    - Color-coded day labels
    - Shaded forecast zone
    - Value annotations
    
    Parameters:
    -----------
    live_value : float
        Current/today's AQI value
    predictions_3day : array-like
        3 predictions for days +1, +2, +3
    pollutant : str
        Pollutant name for labels
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    predictions = np.array(predictions_3day)
    if len(predictions) != 3:
        raise ValueError(f"Expected 3 predictions, got {len(predictions)}")
    
    time_points = np.array([0, 1, 2, 3])
    values = np.array([live_value, predictions[0], predictions[1], predictions[2]])
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Today (solid line)
    ax.plot(0, live_value, 'o', markersize=14, color='#1f77b4', 
           label='Today (Live Observation)', zorder=5, markeredgecolor='navy', markeredgewidth=2)
    
    # Forecast (dotted line)
    ax.plot([0, 1, 2, 3], values, 's--', linewidth=3, markersize=10, 
           color='#ff7f0e', label='3-Day Forecast', alpha=0.8, 
           markeredgecolor='darkorange', markeredgewidth=1.5)
    ax.plot([1, 2, 3], predictions, 's--', linewidth=3, markersize=10,
           color='#ff7f0e', alpha=0.8)
    
    # Shaded forecast zone
    ax.axvspan(0.5, 3.5, alpha=0.1, color='#ff7f0e', label='Forecast Period')
    ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=2, alpha=0.5)
    
    # Annotations
    ax.text(0, live_value + 2, f'TODAY\n{live_value:.1f}', 
           ha='center', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#1f77b4', alpha=0.8, edgecolor='navy'),
           color='white')
    
    day_colors = ['#2ecc71', '#3498db', '#e74c3c']
    for i, (pred, color) in enumerate(zip(predictions, day_colors)):
        day_num = i + 1
        ax.text(day_num, pred - 2, f'Day +{day_num}\n{pred:.1f}', 
               ha='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.8, edgecolor='black'),
               color='white')
    
    # Trend analysis
    trend = predictions[0] - live_value
    if abs(trend) < 0.5:
        trend_text = "STABLE"
        trend_color = '#95a5a6'
    elif trend > 0:
        trend_text = "WORSENING"
        trend_color = '#e74c3c'
    else:
        trend_text = "IMPROVING"
        trend_color = '#2ecc71'
    
    ax.text(1.5, ax.get_ylim()[1] * 0.95, 
           f'{trend_text} (Δ {trend:+.1f})', 
           ha='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.7', facecolor=trend_color, alpha=0.9, edgecolor='black'),
           color='white')
    
    # Formatting
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['Today\n(Now)', 'Tomorrow\n(+1 day)', 'In 2 Days\n(+2 days)', 'In 3 Days\n(+3 days)'],
                       fontsize=10, fontweight='bold')
    ax.set_xlabel('Forecast Timeline', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{pollutant} Index', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='y')
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
    ax.set_ylim(min(values) - 8, max(values) + 12)
    ax.set_facecolor('#f8f9fa')
    
    # Use larger margins to prevent annotation cutoff (top/bottom)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.85, bottom=0.15)
    
    return fig, ax

def plot_predictions_comparison(y_actual, y_pred, title_prefix, pollutant, figsize=(16, 12)):
    """
    Create comprehensive 5-panel visualization for model evaluation.
    
    Panels:
    1. Time series overlay (actual vs predicted)
    2. Per-day scatter plot
    3. Error boxplot by day
    4. Residuals plot
    5. Error histogram
    
    Parameters:
    -----------
    y_actual : np.ndarray
        Actual values (can be 2D for multi-day forecasts)
    y_pred : np.ndarray
        Predicted values (same shape as y_actual)
    title_prefix : str
        Prefix for plot titles
    pollutant : str
        Pollutant name
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig : matplotlib figure object
    """
    # Ensure inputs are numpy arrays
    y_actual = np.asarray(y_actual)
    y_pred = np.asarray(y_pred)
    
    # Flatten if multi-day predictions
    if y_actual.ndim > 1:
        actual_flat = y_actual.flatten()
        pred_flat = y_pred.flatten()
    else:
        actual_flat = y_actual
        pred_flat = y_pred
    
    # Calculate metrics
    mae = mean_absolute_error(actual_flat, pred_flat)
    rmse = sqrt(mean_squared_error(actual_flat, pred_flat))
    r2 = 1 - (np.sum((actual_flat - pred_flat)**2) / 
              np.sum((actual_flat - np.mean(actual_flat))**2))
    residuals = actual_flat - pred_flat
    
    fig = plt.figure(figsize=figsize)
    
    # Panel 1: Time series
    ax1 = plt.subplot(2, 3, 1)
    time_idx = np.arange(len(actual_flat))
    ax1.plot(time_idx, actual_flat, 'o-', label='Actual', color='#1f77b4', linewidth=2)
    ax1.plot(time_idx, pred_flat, 's--', label='Predicted', color='#ff7f0e', linewidth=2)
    ax1.fill_between(time_idx, actual_flat, pred_flat, alpha=0.2, color='#ff7f0e')
    ax1.set_title(f'{title_prefix}: Time Series Overlay', fontweight='bold')
    ax1.set_ylabel(f'{pollutant} AQI', fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Scatter
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(actual_flat, pred_flat, alpha=0.6, s=40, color='#2ca02c')
    lims = [np.min([actual_flat.min(), pred_flat.min()]),
            np.max([actual_flat.max(), pred_flat.max()])]
    ax2.plot(lims, lims, 'r--', lw=2, label='Perfect Prediction')
    ax2.set_xlabel('Actual', fontweight='bold')
    ax2.set_ylabel('Predicted', fontweight='bold')
    ax2.set_title('Actual vs Predicted', fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Residuals
    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(actual_flat, residuals, alpha=0.6, s=40, color='#d62728')
    ax3.axhline(y=0, color='k', linestyle='--', lw=2)
    ax3.set_xlabel('Actual', fontweight='bold')
    ax3.set_ylabel('Residuals', fontweight='bold')
    ax3.set_title('Residuals Plot', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Error distribution
    ax4 = plt.subplot(2, 3, 4)
    errors = np.abs(residuals)
    ax4.hist(errors, bins=30, color='#9467bd', alpha=0.7, edgecolor='black')
    ax4.axvline(np.mean(errors), color='r', linestyle='--', linewidth=2, label=f'Mean={np.mean(errors):.2f}')
    ax4.set_xlabel('Absolute Error', fontweight='bold')
    ax4.set_ylabel('Frequency', fontweight='bold')
    ax4.set_title('Error Distribution', fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Panel 5: Metrics
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    metrics_text = f"""
    MODEL EVALUATION METRICS
    ─────────────────────────
    MAE:   {mae:.4f}
    RMSE:  {rmse:.4f}
    R²:    {r2:.4f}
    
    RESIDUAL STATISTICS
    ─────────────────────────
    Mean:  {np.mean(residuals):+.4f}
    Std:   {np.std(residuals):.4f}
    Min:   {np.min(residuals):+.4f}
    Max:   {np.max(residuals):+.4f}
    """
    ax5.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle(f'{title_prefix} - {pollutant} AQI Forecasting', 
                fontsize=14, fontweight='bold', y=0.995)
    
    fig.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.1)
    plt.tight_layout(pad=1.5)
    
    return fig


def model_results_bar_chart(all_results, pollutant_focus, city_focus, figsize=(14, 8)):
    """
    Create a bar chart comparing multiple models across key metrics.
    
    Parameters:
    -----------
    all_results : list
        List of dictionaries containing model results with metrics
    pollutant_focus : str
        Pollutant name for title
    city_focus : str
        City name for title
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    --------
    fig : matplotlib figure object
    """
    if not all_results:
        # Create empty figure if no results
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No model results available', 
               ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title(f"Model Comparison: {pollutant_focus} in {city_focus}", fontsize=14, fontweight='bold')
        fig.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.1)
        return fig
    
    # Flatten the nested structure: extract model_name and metrics
    flattened_data = []
    for result in all_results:
        if result and isinstance(result, dict):
            row = {'Model': result.get('model_name', 'Unknown')}
            # Extract metrics from nested metrics dict
            if 'metrics' in result and isinstance(result['metrics'], dict):
                row.update(result['metrics'])
            flattened_data.append(row)
    
    if not flattened_data:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No model results available', 
               ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title(f"Model Comparison: {pollutant_focus} in {city_focus}", fontsize=14, fontweight='bold')
        fig.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.1)
        return fig
    
    results_df = pd.DataFrame(flattened_data)
    results_df = results_df.set_index('Model')
    
    # Filter only metric columns that exist
    available_metrics = [m for m in ['RMSE', 'MAE', 'MAPE', 'sMAPE'] if m in results_df.columns]
    
    if not available_metrics:
        # If no metrics, show error
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No metric data available', 
               ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title(f"Model Comparison: {pollutant_focus} in {city_focus}", fontsize=14, fontweight='bold')
        fig.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.1)
        return fig
    
    # Sort by best metric: use MAPE if RMSE values are very similar, otherwise use RMSE
    sort_metric = 'RMSE'
    ranking_description = 'RMSE'
    
    if 'RMSE' in results_df.columns and 'MAPE' in results_df.columns:
        # Calculate coefficient of variation for RMSE
        rmse_mean = results_df['RMSE'].mean()
        rmse_std = results_df['RMSE'].std()
        rmse_cv = rmse_std / rmse_mean if rmse_mean > 0 else float('inf')
        
        # If RMSE values are very similar (CV < 0.05 or 5% variation), use MAPE instead
        if rmse_cv < 0.05:
            sort_metric = 'MAPE'
            ranking_description = 'MAPE (RMSE too similar across models)'
    elif 'MAPE' in results_df.columns:
        sort_metric = 'MAPE'
        ranking_description = 'MAPE'
    
    if sort_metric in results_df.columns:
        results_df = results_df.sort_values(sort_metric)

    fig, ax = plt.subplots(figsize=figsize)
    results_df[available_metrics].plot(kind='bar', ax=ax)
    ax.set_title(f"Model Comparison: {pollutant_focus} in {city_focus}\n(Models Ranked Left-to-Right: Best → Worst by {ranking_description})", 
                fontsize=14, fontweight='bold')
    ax.set_ylabel('Error Metric Value', fontsize=12)
    ax.set_xlabel('Model (Best ← → Worst)', fontsize=12)
    ax.legend(title='Metrics', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.15)
    plt.tight_layout(pad=1.5)
    
    return fig
