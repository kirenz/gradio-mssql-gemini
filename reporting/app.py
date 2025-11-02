import gradio as gr
import pandas as pd
import subprocess
import os
import json

class ReportApp:
    def __init__(self):
        """Initialize app with valid combinations from CSV"""
        try:
            # Load combinations relative to this file so CLI launch paths don't break filtering
            data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'valid_combinations.csv')
            self.valid_combinations = pd.read_csv(data_path)
            # Initialize with all unique values for each field
            self.all_choices = {
                'Sales Organisation': ["All"] + sorted(self.valid_combinations['Sales Organisation'].unique().tolist()),
                'Sales Country': ["All"] + sorted(self.valid_combinations['Sales Country'].unique().tolist()),
                'Sales Region': ["All"] + sorted(self.valid_combinations['Sales Region'].unique().tolist()),
                'Sales City': ["All"] + sorted(self.valid_combinations['Sales City'].unique().tolist()),
                'Product Line': ["All"] + sorted(self.valid_combinations['Product Line'].unique().tolist()),
                'Product Category': ["All"] + sorted(self.valid_combinations['Product Category'].unique().tolist())
            }
        except Exception as e:
            print(f"Error loading combinations: {e}")
            self.valid_combinations = pd.DataFrame()
            self.all_choices = {}

    def get_filtered_choices(self, current_filters):
        """Get valid choices based on current filters"""
        df = self.valid_combinations.copy()
        
        # Apply existing filters
        for field, value in current_filters.items():
            if value and value != "All":
                if isinstance(value, list):
                    df = df[df[field].isin(value)]
                else:
                    df = df[df[field] == value]
        
        # Get unique values for each field
        choices = {
            'Sales Organisation': sorted(df['Sales Organisation'].unique()),
            'Sales Country': sorted(df['Sales Country'].unique()),
            'Sales Region': sorted(df['Sales Region'].unique()),
            'Sales City': sorted(df['Sales City'].unique()),
            'Product Line': sorted(df['Product Line'].unique()),
            'Product Category': sorted(df['Product Category'].unique())
        }
        
        return {k: ["All"] + v for k, v in choices.items()}

    def update_choices(self, sales_org, country, region, city, product_line, product_category):
        """Update dropdowns based on current selections"""
        # Convert single values to lists if needed
        def ensure_list(value):
            if not value:
                return ["All"]
            return value if isinstance(value, list) else [value]

        # Update current filters with lists
        current_filters = {
            'Sales Organisation': ensure_list(sales_org),
            'Sales Country': ensure_list(country),
            'Sales Region': ensure_list(region),
            'Sales City': ensure_list(city),
            'Product Line': ensure_list(product_line),
            'Product Category': ensure_list(product_category)
        }

        # Filter combinations based on current selections
        df = self.valid_combinations.copy()
        for field, values in current_filters.items():
            if values and "All" not in values:
                df = df[df[field].isin(values)]

        # Get new valid choices for each field
        new_choices = {
            'Sales Organisation': ["All"] + sorted(df['Sales Organisation'].unique().tolist()),
            'Sales Country': ["All"] + sorted(df['Sales Country'].unique().tolist()),
            'Sales Region': ["All"] + sorted(df['Sales Region'].unique().tolist()),
            'Sales City': ["All"] + sorted(df['Sales City'].unique().tolist()),
            'Product Line': ["All"] + sorted(df['Product Line'].unique().tolist()),
            'Product Category': ["All"] + sorted(df['Product Category'].unique().tolist())
        }

        # Return updated dropdowns with current values preserved
        return [
            gr.Dropdown(choices=new_choices['Sales Organisation'], value=current_filters['Sales Organisation']),
            gr.Dropdown(choices=new_choices['Sales Country'], value=current_filters['Sales Country']),
            gr.Dropdown(choices=new_choices['Sales Region'], value=current_filters['Sales Region']),
            gr.Dropdown(choices=new_choices['Sales City'], value=current_filters['Sales City']),
            gr.Dropdown(choices=new_choices['Product Line'], value=current_filters['Product Line']),
            gr.Dropdown(choices=new_choices['Product Category'], value=current_filters['Product Category'])
        ]

    def generate_pdf_report(self, 
                          sales_org, country, region, city,
                          product_line, product_category):
        """Generate PDF report using quarto with parameters"""
        try:
            # Process filters
            def process_filter(values):
                if not values or "All" in values:
                    return "All"
                return values if isinstance(values, list) else [values]
            
            filters = {
                'sales_org': process_filter(sales_org),
                'country': process_filter(country),
                'region': process_filter(region),
                'city': process_filter(city),
                'product_line': process_filter(product_line),
                'product_category': process_filter(product_category)
            }
            
            # Save filters
            report_dir = os.path.dirname(os.path.abspath(__file__))
            filters_file = os.path.join(report_dir, 'current_filters.json')
            
            with open(filters_file, 'w') as f:
                json.dump(filters, f)
            
            # Generate report
            cmd = ["quarto", "render", "sales_pdf.qmd", "--execute"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=report_dir)
            
            if result.returncode == 0:
                pdf_path = os.path.join(report_dir, "sales_pdf.pdf")
                return f"Report generated successfully: {pdf_path}", pdf_path
            else:
                return f"Error generating report: {result.stderr}", None
                
        except Exception as e:
            return f"Error: {str(e)}", None

    def launch(self):
        """Launch the Gradio interface"""
        with gr.Blocks(title="Sales Report Generator") as interface:
            gr.Markdown("# Adventure Bikes Sales Report Generator")
            
            with gr.Row():
                with gr.Column():
                    sales_org = gr.Dropdown(
                        choices=self.all_choices['Sales Organisation'],
                        label="Sales Organization",
                        value=["All"],
                        multiselect=True
                    )
                    country = gr.Dropdown(
                        choices=self.all_choices['Sales Country'],
                        label="Country",
                        value=["All"],
                        multiselect=True
                    )
                    region = gr.Dropdown(
                        choices=self.all_choices['Sales Region'],
                        label="Region",
                        value=["All"],
                        multiselect=True
                    )
                    city = gr.Dropdown(
                        choices=self.all_choices['Sales City'],
                        label="City",
                        value=["All"],
                        multiselect=True
                    )
                
                with gr.Column():
                    product_line = gr.Dropdown(
                        choices=self.all_choices['Product Line'],
                        label="Product Line",
                        value=["All"],
                        multiselect=True
                    )
                    product_category = gr.Dropdown(
                        choices=self.all_choices['Product Category'],
                        label="Product Category",
                        value=["All"],
                        multiselect=True
                    )
            
            # Outputs
            generate_pdf_btn = gr.Button("Generate PDF Report", variant="primary")
            report_status = gr.Textbox(label="Report Status", lines=2)
            pdf_output = gr.File(label="Generated PDF Report")
            
            # Update all dropdowns when any selection changes
            all_dropdowns = [sales_org, country, region, city, product_line, product_category]
            for dropdown in all_dropdowns:
                dropdown.change(
                    fn=self.update_choices,
                    inputs=all_dropdowns,
                    outputs=all_dropdowns
                )
            
            # Generate report
            generate_pdf_btn.click(
                fn=self.generate_pdf_report,
                inputs=all_dropdowns,
                outputs=[report_status, pdf_output]
            )
        
        # Try ports in range 7860-7870
        for port in range(7860, 7871):
            try:
                interface.launch(
                    share=False,
                    server_name="0.0.0.0",
                    server_port=port
                )
                break  # Exit loop if launch successful
            except OSError:
                if port == 7870:  # If we've tried all ports
                    print("Could not find an available port. Please try again later.")
                    return
                continue  # Try next port

if __name__ == "__main__":
    app = ReportApp()
    app.launch()
