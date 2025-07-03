# Library to preprocess data
import pandas as pd
import regex as re

# Defining Warnings
def validate_dataframe(dataframe):
    if dataframe is None:
        raise ValueError("Input dataframe is None.")
def validate_column_exists(dataframe, column_name):
    validate_dataframe(dataframe)
    if column_name not in dataframe.columns:
        raise ValueError(f"Column '{column_name}' not found in dataframe.")

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
    dataframe.replace({'yes': 1, 'no': 0}, inplace=True) # Standardise; yes = 1
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
    # - values: A single value or list/tuple/set of values to drop (equality match).
    # - condition: A callable (e.g., lambda x: x > 10) to define custom condition.
    if values is not None:
        if isinstance(values, (list, tuple, set)):
            dataframe = dataframe[~dataframe[column_name].isin(values)]
        else:
            dataframe = dataframe[dataframe[column_name] != values]

    # Drop by condition
    if condition is not None:
        dataframe = dataframe[~dataframe[column_name].apply(condition)]

    return dataframe















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

# Explore how the libraries in
# scikit-learn
# numpy
# optbin
# discretiztion (R)

#this isnt great, replace it
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