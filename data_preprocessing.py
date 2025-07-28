# This is a ultility library for preprocessing data
    # inplace=True is deprecated and being phased out
    # This means we assign the dataframe directly instead of using it
    # ✅ df = df.drop(columns=['index'])
    # ❌ df.drop(columns=['index'], inplace=True)

# Dependencies
import pandas as pd
from pandas.testing import assert_frame_equal
import regex as re
import sqlite3
import os

# Defining Warnings
def validate_dataframe(dataframe):
    if dataframe is None:
        raise ValueError("Input dataframe is None.")
def validate_column_exists(dataframe, column_name):
    validate_dataframe(dataframe)
    if column_name not in dataframe.columns:
        raise ValueError(f"Column '{column_name}' not found in dataframe.")

# Might migrate to pathlib over os considering this is only used in higher level programing
# Data Loading
def data_load(path, db_name, table_name=None):
    """
    Loads one or more tables from a SQLite database into pandas DataFrames.
    Best done as such:
        df_raw = dp.data_load('data', 'database.db')

    Parameters:
    - path (str): Directory path to the database.
    - db_name (str): SQLite database file name.
    - table_name (str, optional): Specific table to load. If None, auto-detects all tables.

    Returns:
    - DataFrame: The loaded table (even if multiple exist, returns the first by default).
    """
    db_path = os.path.join(path, db_name)
    conn = sqlite3.connect(db_path)

    if table_name is None:
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
        table_list = tables['name'].tolist()

        if not table_list:
            conn.close()
            raise ValueError("No tables found in the database.")

        if len(table_list) > 1:
            print(f"[Info] Multiple tables found: {table_list}. Defaulting to '{table_list[0]}'.")

        table_name = table_list[0]  # Default to the first table

    dataframe = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return dataframe

# Basic cleaning
def clean_string_columns(dataframe):
    for col in dataframe.select_dtypes(include=['object', 'string']):
        dataframe[col] = (
            dataframe[col]
            .str.lower()                            # Converts text to lowercase
            .str.strip()                            # Strips whitespace
            .str.replace(r'\s+', ' ', regex=True)   # Replace multiple spaces with one
            .str.replace('.', '', regex=False)      # Remove all periods
        )
    return dataframe

# Binary Converter
def convert_yes_no_to_binary(dataframe):
    dataframe = dataframe.replace({'yes': 1, 'no': 0}) # Standardise; yes = 1
    return dataframe

# Remove all non-numeric text and convert to numeric
def clean_numeric_column(dataframe, column_name):
    validate_column_exists(dataframe, column_name)

    def extract_number(text):
        # Remove characters except [digits, a single decimal point, optional minus sign]
        match = re.search(r'-?\d+(\.\d+)?', str(text))
        return match.group(0) if match else None
    
    dataframe[column_name] = dataframe[column_name].apply(extract_number)
    dataframe[column_name] = pd.to_numeric(dataframe[column_name], errors='coerce') # Converts invalids to NaN
    return dataframe

# Convert to Absolute values
def absolute_values(dataframe, column_name):
    validate_column_exists(dataframe, column_name)
    dataframe[column_name] = pd.to_numeric(dataframe[column_name], errors='coerce').abs()
    return dataframe

# Replace Values
def replace_values(dataframe, column_name, to_replace, value=None):
    validate_column_exists(dataframe, column_name)
    """
    Replace values in a dataframe column.
    # Single value replacement
    replace_values(df, 'col', to_replace='old_val', value='new_val')

    # Multiple replacements with dict
    replace_values(df, 'col', to_replace={'old1': 'new1', 'old2': 'new2'})

    # Multiple values to same replacement
    replace_values(df, 'col', to_replace=['old1', 'old2'], value='new_val')
    """
    if isinstance(to_replace, dict):
        dataframe[column_name] = dataframe[column_name].replace(to_replace)
    else:
        if value is None:
            raise ValueError("`value` must be provided if `to_replace` is not a dict.")
        dataframe[column_name] = dataframe[column_name].replace(to_replace, value)
    return dataframe

# Drop Values
def drop_values(dataframe, column_name, values=None, condition=None):
    validate_column_exists(dataframe, column_name)
    """
    - values: A single value or list/tuple/set of values to drop (equality match).
    eg. drop_values(df, "status", values=["Unknown", "N/A", "Pending"])
    - condition: A callable (e.g., lambda x: x > 10) to define custom condition.
    eg. drop_values(df, "balance", condition=lambda x: x < 0)
    """

    # Drop by values
    if values is not None:
        if isinstance(values, (list, tuple, set)):
            dataframe = dataframe[~dataframe[column_name].isin(values)]
        else:
            dataframe = dataframe[dataframe[column_name] != values]

    # Drop by condition
    if condition is not None:
        dataframe = dataframe[~dataframe[column_name].apply(condition)]
    return dataframe

# Check Duplicates
def duplicate_analysis(dataframe, column_name, preview=False):
    # Get all duplicated rows based on the column
    df_duplicate = dataframe[dataframe[column_name].duplicated(keep=False)].sort_values(by=column_name)

    total_duplicates = df_duplicate.shape[0]
    unique_duplicated_values = df_duplicate[column_name].nunique()

    print(f'Total duplicated rows = {total_duplicates}')
    print(f'Unique duplicated {column_name} values = {unique_duplicated_values}')
    
    # Reset index for easy pair comparison
    df_duplicate = df_duplicate.reset_index(drop=True)

    # Split into first and second appearances
    df1 = df_duplicate.iloc[0::2].reset_index(drop=True)
    df2 = df_duplicate.iloc[1::2].reset_index(drop=True)

    # Compare the features
    differences = (df1 != df2).sum()

    print("\nDifferences between first and second appearances (column-wise):")
    print(differences[differences > 0])

    # Optionally preview data
    if preview:
        print("\nFirst appearances:")
        display(df1.head(5))
        print("\nSecond appearances:")
        display(df2.head(5))

    return df_duplicate

# Unit Test for verification
def Preprocessing_Verification(dataframe):
    """
    This is for checking whether preprocessing in the notebook is the same in the pipeline
    May have to make this more robust if im doing preprocessing for different models and need a unit test that can adapt to all models
    Run the command below in ipynb prior to execution to export the cleaned data
    {target dataframe}.to_csv('data/verification.csv', index=False)
    Ideally we run this at the end of the preprocessing pipeline
    """
    csv_path = 'data/verification.csv'
    ipynb_df = pd.read_csv(csv_path)

    try:
        assert_frame_equal(dataframe, ipynb_df, check_dtype=False)
        print("✅ Verified: DataFrames match")
    except AssertionError as e:
        print("❌ DataFrames do not match:")
        print(e)
        raise

# Binning (Discretization)
# Unsupervised Discretization
# - Equal-wdith binning
# - Equal-Frequency binning
# - Clustering-based (Kmeans)
# - Fixed/Custom bins

# Supervised Discretization
# - Decision Tree binning
# - Chimerge / Chi2-based 
# - MDLP (minimum description length principle

# Alternative methods

# Handling outliers
# - Logartithmic Transformations
# - Winsorization
# - Exclusion

# Explore how the libraries handle it
# scikit-learn
# numpy
# optbin
# discretiztion (R)

#this isnt great, replace it soon
def bin_values(dataframe, column_name, bins, labels=None, new_column_name=None, right=True):
    """
    Bin values in a DataFrame column.

    Parameters:
    - dataframe: The input DataFrame.
    - column_name: The name of the column to bin.
    - bins: List of bin edges or an integer (number of equal-width bins).
    - labels: Optional list of labels for the bins.
    - new_column_name: Optional name for the new column. If None, overwrites the original.
    - right: Indicates whether bins include the right edge.

    Returns:
    - Modified DataFrame with binned column.
    """
    binned = pd.cut(dataframe[column_name], bins=bins, labels=labels, right=right)

    if new_column_name:
        dataframe[new_column_name] = binned
    else:
        dataframe[column_name] = binned

    return dataframe