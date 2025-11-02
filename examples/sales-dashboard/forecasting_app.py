import gradio as gr
import pandas as pd
import os
import logging
from datetime import datetime
import time

try:
    from sales_forecaster import SalesForecaster
except ModuleNotFoundError:
    from .sales_forecaster import SalesForecaster

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SalesForecastApp:
    def __init__(self):
        self.forecaster = SalesForecaster()
        # Create output directory with parents if it doesn't exist
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Output directory initialized at: {self.output_dir}")
        
        # Load and cache the complete dataset for filter dependencies
        with self.forecaster.db.get_connection() as conn:
            self.df = pd.read_sql("""
                SELECT DISTINCT 
                    [Sales Organisation],
                    [Sales Country],
                    [Sales Region],
                    [Sales State],
                    [Sales City],
                    [Product Line],
                    [Product Category]
                FROM [DataSet_Monthly_Sales_and_Quota]
            """, conn)

    def get_dependent_values(self, 
                           sales_org=None, 
                           country=None, 
                           region=None, 
                           state=None):
        """Get filtered values based on previous selections"""
        df = self.df.copy()
        
        if sales_org and sales_org != "All":
            df = df[df['Sales Organisation'] == sales_org]
        if country and country != "All":
            df = df[df['Sales Country'] == country]
        if region and region != "All":
            df = df[df['Sales Region'] == region]
        if state and state != "All":
            df = df[df['Sales State'] == state]
            
        return {
            'countries': ["All"] + sorted(df['Sales Country'].unique().tolist()),
            'regions': ["All"] + sorted(df['Sales Region'].unique().tolist()),
            'states': ["All"] + sorted(df[df['Sales State'].notna()]['Sales State'].unique().tolist()),
            'cities': ["All"] + sorted(df['Sales City'].unique().tolist()),
            'product_lines': ["All"] + sorted(df['Product Line'].unique().tolist()),
            'product_categories': ["All"] + sorted(df['Product Category'].unique().tolist())
        }

    def update_countries(self, sales_org):
        values = self.get_dependent_values(sales_org=sales_org)
        return gr.Dropdown(choices=values['countries'], value="All")

    def update_regions(self, sales_org, country):
        values = self.get_dependent_values(sales_org=sales_org, country=country)
        return gr.Dropdown(choices=values['regions'], value="All")

    def update_states(self, sales_org, country, region):
        values = self.get_dependent_values(sales_org=sales_org, country=country, region=region)
        return gr.Dropdown(choices=values['states'], value="All")

    def update_cities(self, sales_org, country, region, state):
        """Update cities dropdown and check data availability"""
        values = self.get_dependent_values(sales_org=sales_org, country=country, region=region, state=state)
        
        # Check data availability for each city
        available_cities = ["All"]
        df = self.df.copy()
        
        if sales_org and sales_org != "All":
            df = df[df['Sales Organisation'] == sales_org]
        if country and country != "All":
            df = df[df['Sales Country'] == country]
        if region and region != "All":
            df = df[df['Sales Region'] == region]
        if state and state != "All":
            df = df[df['Sales State'] == state]
            
        for city in sorted(df['Sales City'].unique()):
            # Get data for this city
            try:
                city_data = self.forecaster.get_filtered_data(
                    sales_org=sales_org if sales_org != "All" else None,
                    country=country if country != "All" else None,
                    region=region if region != "All" else None,
                    state=state if state != "All" else None,
                    city=city
                )
                if len(city_data) >= 24:  # Mindestens 2 Jahre Daten
                    available_cities.append(city)
            except:
                continue
        
        return gr.Dropdown(
            choices=available_cities,
            value="All",
            label="City (Only showing cities with sufficient data)"
        )

    def update_product_categories(self, product_line):
        df = self.df.copy()
        if product_line and product_line != "All":
            df = df[df['Product Line'] == product_line]
        return gr.Dropdown(choices=["All"] + sorted(df['Product Category'].unique().tolist()), value="All")

    def create_forecast_analysis(
        self,
        sales_org,
        country,
        region,
        state,
        city,
        product_line,
        product_category,
        forecast_periods,
        confidence_interval
    ):
        try:
            # Ensure output directory exists before generating files
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Get filtered data
            filtered_data = self.forecaster.get_filtered_data(
                sales_org=sales_org if sales_org != "All" else None,
                country=country if country != "All" else None,
                region=region if region != "All" else None,
                state=state if state != "All" else None,
                city=city if city != "All" else None,
                product_line=product_line if product_line != "All" else None,
                product_category=product_category if product_category != "All" else None
            )

            # Create timestamp ONCE and pass it to all methods
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Define file paths
            historical_plot_path = os.path.join(self.output_dir, f'historical_data_{timestamp}.png')
            forecast_plot_path = os.path.join(self.output_dir, f'forecast_{timestamp}.png')
            seasonal_plot_path = os.path.join(self.output_dir, f'seasonal_patterns_{timestamp}.png')
            
            # Generate plots with explicit filenames
            self.forecaster.plot_historical_data(
                filtered_data, 
                save_path=self.output_dir,
                filename=f'historical_data_{timestamp}.png'
            )
            
            forecast_results = self.forecaster.create_forecast(filtered_data, forecast_periods, confidence_interval)
            
            self.forecaster.plot_forecast(
                forecast_results,
                title='Sales Forecast for Selected Filters',
                save_path=self.output_dir,
                filename=f'forecast_{timestamp}.png'
            )
            
            self.forecaster.plot_seasonal_patterns(
                filtered_data,
                save_path=self.output_dir,
                filename=f'seasonal_patterns_{timestamp}.png'
            )

            # Create forecast DataFrame for display
            forecast_df = pd.DataFrame({
                'Date': forecast_results['forecast'].index,
                'Forecast': forecast_results['forecast'].values,
                'Lower Bound': forecast_results['confidence_intervals'].iloc[:, 0],
                'Upper Bound': forecast_results['confidence_intervals'].iloc[:, 1]
            })
            forecast_csv_path = os.path.join(self.output_dir, f'forecast_results_{timestamp}.csv')
            forecast_df.to_csv(forecast_csv_path, index=False)
            
            # Save Excel file in background
            self.forecaster.export_results(
                filtered_data, 
                forecast_results, 
                save_path=self.output_dir,
                filename=f'forecast_results_{timestamp}.xlsx'
            )
            
            # Wait for files with proper error handling
            max_retries = 5
            retry_delay = 2  # seconds
            
            for _ in range(max_retries):
                missing_files = [f for f in [
                    historical_plot_path,
                    forecast_plot_path,
                    seasonal_plot_path
                ] if not os.path.exists(f) or os.path.getsize(f) == 0]
                
                if not missing_files:  # All files exist and are non-empty
                    break
                    
                logger.info(f"Waiting for files... Attempt {_ + 1}/{max_retries}")
                time.sleep(retry_delay)
            else:
                raise FileNotFoundError(
                    f"Timeout waiting for files to be created. Missing or empty files: {missing_files}"
                )

            # Prepare summary text
            filters_used = [
                f"Sales Organization: {sales_org}",
                f"Country: {country}",
                f"Region: {region}",
                f"State: {state}",
                f"City: {city}",
                f"Product Line: {product_line}",
                f"Product Category: {product_category}"
            ]
            
            summary = f"""
            Forecast Analysis Summary:
            -------------------------
            Applied Filters:
            {chr(10).join('- ' + f for f in filters_used)}
            
            Data Points Analyzed: {len(filtered_data)}
            Forecast Periods: {forecast_periods}
            Confidence Interval: {confidence_interval * 100}%
            MAPE: {forecast_results['mape']:.2f}%
            
            Latest Historical Value: {forecast_results['historical_data'].iloc[-1]:,.2f} EUR
            Latest Forecast Value: {forecast_results['forecast'].iloc[-1]:,.2f} EUR
            
            Files saved in: {self.output_dir}
            """
            
            return summary, historical_plot_path, forecast_plot_path, seasonal_plot_path, forecast_df, forecast_csv_path
            
        except FileNotFoundError as e:
            logger.error(f"File creation error: {str(e)}")
            return f"Error: Could not create output files in {self.output_dir}. Please check directory permissions.", None, None, None, None, None
            
        except ValueError as e:
            # Benutzerfreundliche Fehlermeldung für unzureichende Daten
            error_message = f"""
            ⚠️ Warnung: {str(e)}

            Empfehlungen:
            - Wählen Sie einen größeren geografischen Bereich
            - Reduzieren Sie die Anzahl der Filter
            - Wählen Sie eine gröbere Granularität (z.B. Region statt Stadt)
            """
            return error_message, None, None, None, None, None
            
        except Exception as e:
            logger.error(f"Error in forecast analysis: {str(e)}")
            return f"Ein Fehler ist aufgetreten: {str(e)}", None, None, None, None, None

    def launch(self):
        with gr.Blocks(title="Sales Forecaster", css="footer {visibility: hidden}") as interface:
            gr.Markdown("# Sales Forecaster")
            gr.Markdown("Select filters and parameters to generate sales forecasts")
            
            with gr.Row():
                with gr.Column():
                    # Filter inputs with dependencies
                    sales_org = gr.Dropdown(
                        ["All"] + sorted(self.df['Sales Organisation'].unique().tolist()),
                        label="Sales Organization",
                        value="All"
                    )
                    country = gr.Dropdown(
                        ["All"],
                        label="Country",
                        value="All"
                    )
                    region = gr.Dropdown(
                        ["All"],
                        label="Region",
                        value="All"
                    )
                    state = gr.Dropdown(
                        ["All"],
                        label="State",
                        value="All"
                    )
                    city = gr.Dropdown(
                        ["All"],
                        label="City",
                        value="All"
                    )
                
                with gr.Column():
                    product_line = gr.Dropdown(
                        ["All"] + sorted(self.df['Product Line'].unique().tolist()),
                        label="Product Line",
                        value="All"
                    )
                    product_category = gr.Dropdown(
                        ["All"],
                        label="Product Category",
                        value="All"
                    )
            
            with gr.Row():
                forecast_periods = gr.Slider(
                    minimum=1,
                    maximum=24,
                    value=12,
                    step=1,
                    label="Forecast Periods (Months)"
                )
                confidence_interval = gr.Slider(
                    minimum=0.8,
                    maximum=0.99,
                    value=0.95,
                    step=0.01,
                    label="Confidence Interval"
                )
            
            # Submit button
            submit_btn = gr.Button("Generate Forecast", variant="primary")
            
            # Outputs
            summary_output = gr.Textbox(
                label="Analysis Summary",
                lines=10
            )
            
            with gr.Row():
                historical_plot = gr.Image(
                    label="Historical Sales",
                    type="filepath"
                )
                forecast_plot = gr.Image(
                    label="Sales Forecast",
                    type="filepath"
                )
            
            seasonal_plot = gr.Image(
                label="Seasonal Patterns",
                type="filepath"
            )
            
            with gr.Column():
                forecast_table = gr.Dataframe(
                    label="Forecast Results"
                )
                forecast_download = gr.DownloadButton(
                    label="Download Forecast CSV",
                    value=None,
                    variant="secondary"
                )
            
            # Set up dependent dropdown updates
            sales_org.change(
                fn=self.update_countries,
                inputs=[sales_org],
                outputs=[country]
            )
            
            country.change(
                fn=self.update_regions,
                inputs=[sales_org, country],
                outputs=[region]
            )
            
            region.change(
                fn=self.update_states,
                inputs=[sales_org, country, region],
                outputs=[state]
            )
            
            state.change(
                fn=self.update_cities,
                inputs=[sales_org, country, region, state],
                outputs=[city]
            )
            
            product_line.change(
                fn=self.update_product_categories,
                inputs=[product_line],
                outputs=[product_category]
            )
            
            # Set up forecast generation
            submit_btn.click(
                fn=self.create_forecast_analysis,
                inputs=[
                    sales_org,
                    country,
                    region,
                    state,
                    city,
                    product_line,
                    product_category,
                    forecast_periods,
                    confidence_interval
                ],
                outputs=[
                    summary_output,
                    historical_plot,
                    forecast_plot,
                    seasonal_plot,
                    forecast_table,
                    forecast_download
                ]
            )
        
        # Launch the interface
        interface.launch(
            share=True,
            server_name="0.0.0.0",
            server_port=7860,
            show_api=False
        )

if __name__ == "__main__":
    app = SalesForecastApp()
    app.launch()
