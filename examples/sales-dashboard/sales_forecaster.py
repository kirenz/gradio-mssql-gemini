import sys
from pathlib import Path

src_path = Path(__file__).resolve().parents[2] / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import Optional, Dict, Union, List, Any
from datetime import datetime
import logging
from sqlalchemy.sql import text
import altair as alt
import os
from contextlib import contextmanager

from quarto_mssql_gemini import create_engine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseConnection:
    def __init__(self):
        """Wrap shared SQLAlchemy engine for forecasting module reuse"""
        self.engine = create_engine()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        connection = self.engine.connect()
        try:
            yield connection
            logger.info("Database connection successful")
        except Exception as e:
            logger.error(f"Database connection error: {str(e)}")
            raise
        finally:
            connection.close()
            logger.info("Database connection closed")

class SalesForecaster:
    def __init__(self):
        """Initialize the forecaster with database connection"""
        self.db = DatabaseConnection()
        self._validate_connection()

    def _validate_connection(self):
        """Validate database connection"""
        try:
            with self.db.get_connection() as conn:
                df = pd.read_sql('SELECT @@version', conn)
                logger.info(f"Connected to SQL Server version: {df.iloc[0,0]}")
        except Exception as e:
            logger.error(f"Failed to validate database connection: {str(e)}")
            raise

    def _build_query(self,
                    sales_org: Optional[str] = None,
                    country: Optional[str] = None,
                    region: Optional[str] = None,
                    city: Optional[str] = None,
                    state: Optional[str] = None,
                    product_line: Optional[str] = None,
                    product_category: Optional[str] = None) -> tuple[str, dict]:
        """
        Build SQL query based on filters
        
        Returns:
            Tuple of (query string, parameters dictionary)
        """
        query = """
        SELECT 
            [Calendar DueDate],
            [Revenue EUR],
            [Sales Amount],
            [Sales Organisation],
            [Sales Country],
            [Sales Region],
            [Sales City],
            [Sales State],
            [Product Line],
            [Product Category]
        FROM [DataSet_Monthly_Sales_and_Quota]
        WHERE 1=1
        """
        
        params = {}
        if sales_org:
            query += " AND [Sales Organisation] = :sales_org"
            params['sales_org'] = sales_org
        if country:
            query += " AND [Sales Country] = :country"
            params['country'] = country
        if region:
            query += " AND [Sales Region] = :region"
            params['region'] = region
        if city:
            query += " AND [Sales City] = :city"
            params['city'] = city
        if state:
            query += " AND [Sales State] = :state"
            params['state'] = state
        if product_line:
            query += " AND [Product Line] = :product_line"
            params['product_line'] = product_line
        if product_category:
            query += " AND [Product Category] = :product_category"
            params['product_category'] = product_category
            
        query += " ORDER BY [Calendar DueDate]"
        
        return query, params

    def get_filtered_data(self,
                         sales_org: Optional[str] = None,
                         country: Optional[str] = None,
                         region: Optional[str] = None,
                         city: Optional[str] = None,
                         state: Optional[str] = None,
                         product_line: Optional[str] = None,
                         product_category: Optional[str] = None) -> pd.DataFrame:
        """
        Get filtered data from database with validation
        """
        query, params = self._build_query(
            sales_org, country, region, city, state, product_line, product_category
        )
        
        try:
            with self.db.get_connection() as conn:
                # Use text() to properly handle parameter binding
                sql = text(query)
                df = pd.read_sql_query(sql, conn, params=params)
                df['Calendar DueDate'] = pd.to_datetime(df['Calendar DueDate'])
                
                # Check minimum data requirements
                if len(df) < 24:  # Mindestens 2 Jahre an Daten
                    filters_used = [f"{k}: {v}" for k, v in {
                        'Sales Organization': sales_org,
                        'Country': country,
                        'Region': region,
                        'City': city,
                        'State': state,
                        'Product Line': product_line,
                        'Product Category': product_category
                    }.items() if v is not None]
                    
                    raise ValueError(
                        f"Insufficient data for forecast. Found only {len(df)} data points with filters:\n"
                        f"{chr(10).join('- ' + f for f in filters_used)}\n"
                        f"Need at least 24 monthly data points for reliable forecasting."
                    )
                
                logger.info(f"Retrieved {len(df)} records from database")
                return df
                
        except Exception as e:
            logger.error(f"Error retrieving data: {str(e)}")
            raise

    def prepare_time_series(self, filtered_df: pd.DataFrame) -> pd.Series:
        """
        Prepare time series data for forecasting
        """
        # Group by date and sum revenue
        ts = filtered_df.groupby('Calendar DueDate')['Revenue EUR'].sum()
        ts.index = pd.DatetimeIndex(ts.index)
        ts = ts.sort_index()
        
        # Check for missing dates and fill them
        date_range = pd.date_range(start=ts.index.min(), end=ts.index.max(), freq='ME')
        ts = ts.reindex(date_range, fill_value=0)
        
        return ts

    def _save_chart(self, chart: alt.Chart, filename: str, save_path: str) -> None:
        """Helper method to save Altair charts"""
        os.makedirs(save_path, exist_ok=True)
        filepath = os.path.join(save_path, filename)
        try:
            chart.save(filepath)
            logger.info(f"Chart saved successfully to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save chart: {str(e)}")
            raise

    def plot_historical_data(self, filtered_data: pd.DataFrame, save_path: str = './', filename: str = None) -> alt.Chart:
        """
        Create an Altair visualization of historical sales data
        """
        # Prepare data
        df_melted = pd.melt(
            filtered_data.groupby('Calendar DueDate').agg({
                'Revenue EUR': 'sum',
                'Sales Amount': 'sum'
            }).reset_index(),
            id_vars=['Calendar DueDate'],
            var_name='Metric',
            value_name='Value'
        )

        # Create base chart
        base = alt.Chart(df_melted).encode(
            x=alt.X('Calendar DueDate:T', title='Date'),
            color=alt.Color('Metric:N', legend=alt.Legend(title=None))
        )

        # Create the visualization
        chart = alt.vconcat(
            # Revenue chart
            base.transform_filter(
                alt.datum.Metric == 'Revenue EUR'
            ).mark_line().encode(
                y=alt.Y('Value:Q', title='Revenue (EUR)'),
            ).properties(
                title='Monthly Revenue',
                height=300
            ),
            
            # Sales Amount chart
            base.transform_filter(
                alt.datum.Metric == 'Sales Amount'
            ).mark_line(color='#2ca02c').encode(
                y=alt.Y('Value:Q', title='Sales Amount'),
            ).properties(
                title='Monthly Sales Amount',
                height=300
            )
        ).properties(
            title='Historical Sales Performance'
        )

        # Save the chart with explicit filename
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'historical_data_{timestamp}.png'
        
        self._save_chart(chart, filename, save_path)
        return chart

    def plot_forecast(self, forecast_results: Dict[str, Any], title: str = None, save_path: str = './', filename: str = None) -> alt.Chart:
        """
        Create an Altair visualization of forecast results
        """
        # Prepare data
        historical = pd.DataFrame({
            'Date': forecast_results['historical_data'].index,
            'Value': forecast_results['historical_data'].values,
            'Type': 'Historical'
        })
        
        forecast = pd.DataFrame({
            'Date': forecast_results['forecast'].index,
            'Value': forecast_results['forecast'].values,
            'Type': 'Forecast'
        })
        
        ci_df = forecast_results['confidence_intervals'].copy()
        ci_df.columns = ['Lower Bound', 'Upper Bound']
        ci_df['Date'] = ci_df.index
        
        # Combine historical and forecast data
        df = pd.concat([historical, forecast])
        
        # Create the main line chart
        line_chart = alt.Chart(df).mark_line().encode(
            x=alt.X('Date:T', title='Date'),
            y=alt.Y('Value:Q', title='Revenue (EUR)'),
            color=alt.Color('Type:N', legend=alt.Legend(title=None))
        )
        
        # Create the confidence interval area
        ci_area = alt.Chart(ci_df).mark_area(opacity=0.3).encode(
            x='Date:T',
            y='Lower Bound:Q',
            y2='Upper Bound:Q',
            color=alt.value('gray')
        )
        
        # Combine the charts
        chart = (ci_area + line_chart).properties(
            width=800,
            height=400,
            title=title or 'Sales Forecast with Confidence Intervals'
        )

        # Save the chart with explicit filename
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'forecast_{timestamp}.png'
            
        self._save_chart(chart, filename, save_path)
        return chart

    def plot_seasonal_patterns(self, filtered_data: pd.DataFrame, save_path: str = './', filename: str = None) -> alt.Chart:
        """
        Create Altair visualization of seasonal patterns
        """
        # Prepare data
        df = filtered_data.copy()
        df['Year'] = df['Calendar DueDate'].dt.year
        df['Month'] = df['Calendar DueDate'].dt.month
        monthly_avg = df.groupby(['Year', 'Month'])['Revenue EUR'].mean().reset_index()
        
        # Create the chart
        chart = alt.Chart(monthly_avg).mark_line().encode(
            x=alt.X('Month:O', 
                   title='Month',
                   axis=alt.Axis(labelAngle=0)),
            y=alt.Y('Revenue EUR:Q', 
                   title='Average Revenue (EUR)'),
            color=alt.Color('Year:N', 
                          legend=alt.Legend(title='Year')),
            tooltip=['Year:N', 'Month:O', 'Revenue EUR:Q']
        ).properties(
            width=800,
            height=400,
            title='Seasonal Revenue Patterns by Year'
        )

        # Save the chart with explicit filename
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'seasonal_patterns_{timestamp}.png'
            
        self._save_chart(chart, filename, save_path)
        return chart

    def export_results(self, 
                      filtered_data: pd.DataFrame, 
                      forecast_results: Dict[str, Any],
                      save_path: str = './', 
                      filename: str = None) -> None:
        """
        Export results to Excel file with multiple sheets
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'forecast_results_{timestamp}.xlsx'
            
        filepath = os.path.join(save_path, filename)
        
        # Create a Pandas Excel writer
        with pd.ExcelWriter(filepath) as writer:
            # Export historical data
            historical_df = filtered_data.groupby('Calendar DueDate').agg({
                'Revenue EUR': 'sum',
                'Sales Amount': 'sum'
            }).reset_index()
            historical_df.to_excel(writer, sheet_name='Historical Data', index=False)
            
            # Export forecast results
            forecast_df = pd.DataFrame({
                'Date': forecast_results['forecast'].index,
                'Forecast': forecast_results['forecast'].values,
                'Lower Bound': forecast_results['confidence_intervals'].iloc[:, 0],
                'Upper Bound': forecast_results['confidence_intervals'].iloc[:, 1]
            })
            forecast_df.to_excel(writer, sheet_name='Forecast', index=False)
            
            # Export model metrics
            metrics_df = pd.DataFrame({
                'Metric': ['MAPE'],
                'Value': [forecast_results['mape']]
            })
            metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
            
        logger.info(f"Results exported to {filename}")

    def create_forecast(self,
                       filtered_data: pd.DataFrame,
                       forecast_periods: int = 12,
                       confidence_interval: float = 0.95) -> Dict[str, Union[pd.Series, pd.DataFrame, Any]]:
        """
        Create forecast using SARIMAX model
        """
        try:
            # Prepare time series
            ts = self.prepare_time_series(filtered_data)
            
            # Fit SARIMAX model
            model = SARIMAX(ts,
                          order=(1, 1, 1),
                          seasonal_order=(1, 1, 1, 12),
                          enforce_stationarity=False)
            results = model.fit()
            
            # Generate forecast
            forecast = results.get_forecast(steps=forecast_periods)
            forecast_mean = forecast.predicted_mean
            forecast_ci = forecast.conf_int(alpha=1-confidence_interval)
            
            # Calculate error metrics
            mape = np.mean(np.abs(results.resid / ts)) * 100
            
            return {
                'historical_data': ts,
                'forecast': forecast_mean,
                'confidence_intervals': forecast_ci,
                'mape': mape,
                'model_summary': results.summary()
            }
            
        except Exception as e:
            logger.error(f"Error creating forecast: {str(e)}")
            raise

    def get_available_filters(self) -> Dict[str, List[str]]:
        """
        Get all available filter values from database
        """
        try:
            with self.db.get_connection() as conn:
                filters = {}
                
                # Get unique values for each filter
                queries = {
                    'sales_organisations': 'SELECT DISTINCT [Sales Organisation] FROM [DataSet_Monthly_Sales_and_Quota]',
                    'countries': 'SELECT DISTINCT [Sales Country] FROM [DataSet_Monthly_Sales_and_Quota]',
                    'regions': 'SELECT DISTINCT [Sales Region] FROM [DataSet_Monthly_Sales_and_Quota]',
                    'cities': 'SELECT DISTINCT [Sales City] FROM [DataSet_Monthly_Sales_and_Quota]',
                    'states': 'SELECT DISTINCT [Sales State] FROM [DataSet_Monthly_Sales_and_Quota] WHERE [Sales State] IS NOT NULL',
                    'product_lines': 'SELECT DISTINCT [Product Line] FROM [DataSet_Monthly_Sales_and_Quota]',
                    'product_categories': 'SELECT DISTINCT [Product Category] FROM [DataSet_Monthly_Sales_and_Quota]'
                }
                
                for key, query in queries.items():
                    df = pd.read_sql_query(query, conn)
                    filters[key] = df.iloc[:, 0].tolist()
                
                return filters
                
        except Exception as e:
            logger.error(f"Error retrieving filters: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    try:
        # Initialize forecaster
        forecaster = SalesForecaster()
        
        # Get available filters
        filters = forecaster.get_available_filters()
        print("Available filters:", filters)
        
        # Create output directory for plots
        output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        # Get filtered data
        filtered_data = forecaster.get_filtered_data(
            country='Netherlands',
            product_category='City Bikes'
        )
        
        # Create and save historical data plot
        print("\nCreating historical data plot...")
        forecaster.plot_historical_data(filtered_data, save_path=output_dir)
        
        # Create forecast
        forecast_results = forecaster.create_forecast(filtered_data)
        
        # Create and save forecast plot
        print("Creating forecast plot...")
        forecaster.plot_forecast(
            forecast_results,
            title='Sales Forecast for Netherlands - City Bikes',
            save_path=output_dir
        )
        
        # Create and save seasonal patterns plot
        print("Creating seasonal patterns plot...")
        forecaster.plot_seasonal_patterns(filtered_data, save_path=output_dir)
        
        # Export results
        forecaster.export_results(filtered_data, forecast_results, save_path=output_dir)
        
        # Print results
        print(f"\nForecast MAPE: {forecast_results['mape']:.2f}%")
        print("\nForecast values:")
        print(forecast_results['forecast'])
        print("\nConfidence intervals:")
        print(forecast_results['confidence_intervals'])
        
        print(f"\nPlots have been saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
