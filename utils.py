"""
Utility functions for the Data Cleaning Assistant.
This module handles file I/O operations and data validation.
"""

import pandas as pd
import io
from typing import Optional, Tuple

def load_csv(file_obj) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load a CSV file into a pandas DataFrame.
    
    This function tries to intelligently read CSV files by attempting
    different encodings if the default fails. This is important because
    CSV files can come from different sources with different encodings.
    
    Args:
        file_obj: A file-like object (from Streamlit's file_uploader)
    
    Returns:
        tuple: (DataFrame, error_message)
               If successful, returns (df, None)
               If failed, returns (None, error_string)
    """
    try:
        # Try reading with default UTF-8 encoding first
        df = pd.read_csv(file_obj)
        return df, None
    except UnicodeDecodeError:
        # If UTF-8 fails, try with latin-1 encoding (common alternative)
        try:
            file_obj.seek(0)  # Reset file pointer to beginning
            df = pd.read_csv(file_obj, encoding='latin-1')
            return df, None
        except Exception as e:
            return None, f"Encoding error: {str(e)}"
    except Exception as e:
        # Catch any other errors (malformed CSV, etc.)
        return None, f"Error loading CSV: {str(e)}"


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Validate that the DataFrame is suitable for analysis.
    
    We perform basic sanity checks to ensure the data can be
    processed. This prevents crashes later in the pipeline.
    
    Args:
        df: pandas DataFrame to validate
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df is None:
        return False, "DataFrame is None"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df.columns) == 0:
        return False, "DataFrame has no columns"
    
    # Check if DataFrame is too large (prevent memory issues)
    max_rows = 100000
    if len(df) > max_rows:
        return False, f"DataFrame too large. Maximum {max_rows} rows allowed."
    
    return True, None


def get_sample_data(df: pd.DataFrame, n_rows: int = 5) -> pd.DataFrame:
    """
    Get a sample of the DataFrame for preview or LLM analysis.
    
    We limit the amount of data sent to the LLM to:
    1. Reduce API costs
    2. Stay within token limits
    3. Speed up processing
    
    Args:
        df: Full DataFrame
        n_rows: Number of rows to sample
    
    Returns:
        Sampled DataFrame
    """
    if len(df) <= n_rows:
        return df
    return df.head(n_rows)


def dataframe_to_csv_string(df: pd.DataFrame) -> str:
    """
    Convert DataFrame to CSV string for download.
    
    This creates an in-memory CSV string that can be offered
    as a download to the user without writing to disk.
    
    Args:
        df: DataFrame to convert
    
    Returns:
        CSV string
    """
    return df.to_csv(index=False)


def format_column_info(df: pd.DataFrame) -> str:
    """
    Create a formatted string with column information.
    
    This provides a quick overview of the DataFrame structure
    that's human-readable and useful for the LLM context.
    
    Args:
        df: DataFrame to describe
    
    Returns:
        Formatted string with column details
    """
    info_lines = []
    info_lines.append(f"Total rows: {len(df)}")
    info_lines.append(f"Total columns: {len(df.columns)}\n")
    info_lines.append("Column Details:")
    
    for col in df.columns:
        dtype = df[col].dtype
        non_null = df[col].count()
        null_count = df[col].isna().sum()
        info_lines.append(
            f"  - {col}: {dtype} ({non_null} non-null, {null_count} null)"
        )
    
    return "\n".join(info_lines)