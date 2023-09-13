# This modeule contains functions for general data preparation for the rest of the analysis


import pandas as pd

def convert_datatypes(df, columns_datatypes):
    """
    Convert the data types of specified columns in a dataframe.
    
    Parameters:
    - df: The input dataframe
    - columns_datatypes: A dictionary with column names as keys and desired data types as values
    
    Returns:
    - The dataframe with updated column data types
    """
    for column, dtype in columns_datatypes.items():
        if column in df.columns:
            df[column] = df[column].astype(dtype)
        else:
            print(f"Column {column} not found in the dataframe.")

    df.rename({'InvoiceDate':'InvoiceDatetime'}, axis=1, inplace=True)
    
    return df



def feature_engineering(df):
    """
    This function adds new features to the provided dataframe.
    
    Parameters:
    - df: The original dataframe.
    
    Returns:
    - df: The dataframe with new features.
    """
    
    # Calculate total spending per invoice
    df['TotalSpending'] = df['Quantity'] * df['Price']

    # Categorize purchases
    df['PriceCategory'] = pd.cut(df['Price'], 
                                 bins=[0, 10, 50, 100, 500, 1000], 
                                 labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

    # Extract year-month and individual date from 'InvoiceDatetime'
    df['YearMonth'] = df['InvoiceDatetime'].dt.to_period('M')
    df['InvoiceDate'] = df['InvoiceDatetime'].dt.to_period('D')
    df['Year'] = df['InvoiceDatetime'].dt.year
    df['Month'] = df['InvoiceDatetime'].dt.month
    
    # Extract the weekday
    df['Weekday'] = df['InvoiceDatetime'].apply(lambda x: x.strftime('%A'))
    
    return df


def data_cleaning(df):
    """
    This function processes the provided dataframe by:
    - Dropping rows with missing 'Customer ID'
    - Resetting the index
    - Removing rows with 'Customer ID' = 12346.0
    
    Parameters:
    - df: The original dataframe.
    
    Returns:
    - df: The processed dataframe.
    """
    
    # Drop rows with missing 'Customer ID'
    df.dropna(subset=['Customer ID'], inplace=True)

    # Reset the index
    df.reset_index(drop=True, inplace=True)

    # Remove rows with 'Customer ID' = 12346.0
    df = df[df['Customer ID'] != 12346.0]

    df.reset_index(drop=True, inplace=True)
    
    return df

def create_integer_indices(df, column_dict):
    """
    Enrich a DataFrame with integer indices for dict of columns.
    This is useful for feeding obeservations into neural net.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing the actual customer-product pairs.
    column_dict : dict
        Dictionary of  pairs of columns to convert to indices and the new name for the index

    Returns:
    -------
    pd.DataFrame
        A new DataFrame containing both the original and the indices

    """
    for col in column_dict:
        # Get unique values from column
        unique_col_vals = df[col].unique()
        # Create mappings for columns to integers
        col_to_index = {val1: index for index, val1 in enumerate(unique_col_vals)}
        # Create new columns in the data with the integer mappings
        df[column_dict[col]] = df[col].map(col_to_index)
    return df

