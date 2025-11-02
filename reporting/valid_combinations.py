import pandas as pd

def load_valid_combinations(filepath='valid_combinations.csv'):
    """Load valid combinations from CSV file"""
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading combinations: {e}")
        return pd.DataFrame()

def get_filtered_choices(df_combinations, current_filters):
    """Get valid choices for each filter based on current selections"""
    df = df_combinations.copy()
    
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

def is_valid_combination(df_combinations, filters):
    """Check if a combination of filters is valid"""
    query = ' & '.join([
        f"`{k}` == @filters['{k}']" 
        for k, v in filters.items() 
        if v != "All"
    ])
    
    if not query:
        return True
        
    matching = df_combinations.query(query)
    return len(matching) > 0
