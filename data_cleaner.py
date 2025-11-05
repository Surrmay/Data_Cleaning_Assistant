import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import re
from datetime import datetime


class DataCleaner:
    """Comprehensive data cleaning with intelligent type detection and handling"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        self.cleaning_report = []
        
    def clean(self, 
              handle_missing: str = 'smart',  # 'smart', 'drop', 'fill', 'none'
              handle_duplicates: bool = True,
              handle_outliers: bool = True,
              outlier_method: str = 'iqr',  # 'iqr', 'zscore', 'none'
              standardize_text: bool = True,
              convert_types: bool = True,
              remove_special_chars: bool = False) -> pd.DataFrame:
        """
        One-shot comprehensive data cleaning
        
        Returns: Cleaned DataFrame
        """
        
        self._log("Starting data cleaning...")
        self._log(f"Original shape: {self.original_shape}")
        
        # Step 1: Remove completely empty rows/columns
        self._remove_empty_rows_cols()
        
        # Step 2: Standardize column names
        self._standardize_columns()
        
        # Step 3: Convert data types intelligently
        if convert_types:
            self._auto_convert_types()
        
        # Step 4: Handle text data
        if standardize_text:
            self._standardize_text(remove_special_chars)
        
        # Step 5: Handle missing values
        if handle_missing != 'none':
            self._handle_missing_values(handle_missing)
        
        # Step 6: Remove duplicates
        if handle_duplicates:
            self._remove_duplicates()
        
        # Step 7: Handle outliers in numeric columns
        if handle_outliers and outlier_method != 'none':
            self._handle_outliers(outlier_method)
        
        # Step 8: Final validation
        self._validate_data()
        
        self._log(f"Final shape: {self.df.shape}")
        self._print_report()
        
        return self.df
    
    def _remove_empty_rows_cols(self):
        """Remove completely empty rows and columns"""
        # Remove columns with all NaN
        empty_cols = self.df.columns[self.df.isna().all()].tolist()
        if empty_cols:
            self.df = self.df.drop(columns=empty_cols)
            self._log(f"Removed {len(empty_cols)} empty columns: {empty_cols}")
        
        # Remove rows with all NaN
        before = len(self.df)
        self.df = self.df.dropna(how='all')
        removed = before - len(self.df)
        if removed > 0:
            self._log(f"Removed {removed} completely empty rows")
    
    def _standardize_columns(self):
        """Standardize column names"""
        new_cols = []
        for col in self.df.columns:
            # Convert to lowercase, replace spaces/special chars with underscore
            clean_col = re.sub(r'[^\w\s]', '', str(col))
            clean_col = re.sub(r'\s+', '_', clean_col.strip().lower())
            new_cols.append(clean_col)
        
        self.df.columns = new_cols
        self._log(f"Standardized {len(new_cols)} column names")
    
    def _auto_convert_types(self):
        """Intelligently detect and convert data types"""
        for col in self.df.columns:
            original_dtype = self.df[col].dtype
            
            # Skip if already numeric
            if pd.api.types.is_numeric_dtype(self.df[col]):
                continue
            
            # Try to convert to numeric
            if self._is_numeric_string(col):
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                self._log(f"Converted '{col}' to numeric")
            
            # Try to convert to datetime
            elif self._is_datetime_string(col):
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                self._log(f"Converted '{col}' to datetime")
            
            # Try to convert to boolean
            elif self._is_boolean_string(col):
                bool_map = {
                    'true': True, 'false': False, 
                    'yes': True, 'no': False,
                    '1': True, '0': False,
                    't': True, 'f': False,
                    'y': True, 'n': False
                }
                self.df[col] = self.df[col].str.lower().map(bool_map)
                self._log(f"Converted '{col}' to boolean")
    
    def _is_numeric_string(self, col: str) -> bool:
        """Check if string column contains numeric values"""
        sample = self.df[col].dropna().astype(str).head(100)
        if len(sample) == 0:
            return False
        
        # Remove common numeric separators
        cleaned = sample.str.replace(',', '').str.replace('$', '').str.strip()
        numeric_count = pd.to_numeric(cleaned, errors='coerce').notna().sum()
        
        return numeric_count / len(sample) > 0.8
    
    def _is_datetime_string(self, col: str) -> bool:
        """Check if string column contains datetime values"""
        sample = self.df[col].dropna().astype(str).head(100)
        if len(sample) == 0:
            return False
        
        datetime_count = pd.to_datetime(sample, errors='coerce').notna().sum()
        return datetime_count / len(sample) > 0.8
    
    def _is_boolean_string(self, col: str) -> bool:
        """Check if string column contains boolean values"""
        sample = self.df[col].dropna().astype(str).str.lower().head(100)
        if len(sample) == 0:
            return False
        
        bool_values = {'true', 'false', 'yes', 'no', '1', '0', 't', 'f', 'y', 'n'}
        unique_values = set(sample.unique())
        
        return unique_values.issubset(bool_values)
    
    def _standardize_text(self, remove_special_chars: bool):
        """Standardize text columns"""
        text_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in text_cols:
            # Strip whitespace
            self.df[col] = self.df[col].astype(str).str.strip()
            
            # Remove multiple spaces
            self.df[col] = self.df[col].str.replace(r'\s+', ' ', regex=True)
            
            # Remove special characters if requested
            if remove_special_chars:
                self.df[col] = self.df[col].str.replace(r'[^\w\s]', '', regex=True)
            
            # Replace 'nan' strings with actual NaN
            self.df[col] = self.df[col].replace(['nan', 'NaN', 'None', 'null', ''], np.nan)
        
        if len(text_cols) > 0:
            self._log(f"Standardized {len(text_cols)} text columns")
    
    def _handle_missing_values(self, method: str):
        """Handle missing values with smart strategy"""
        missing_before = self.df.isna().sum().sum()
        
        if method == 'drop':
            # Drop rows with any missing values
            self.df = self.df.dropna()
            self._log(f"Dropped rows with missing values")
        
        elif method == 'fill':
            # Fill numeric with median, categorical with mode
            for col in self.df.columns:
                if self.df[col].isna().sum() > 0:
                    if pd.api.types.is_numeric_dtype(self.df[col]):
                        self.df[col].fillna(self.df[col].median(), inplace=True)
                    else:
                        mode_val = self.df[col].mode()
                        if len(mode_val) > 0:
                            self.df[col].fillna(mode_val[0], inplace=True)
        
        elif method == 'smart':
            # Smart handling based on missing percentage
            for col in self.df.columns:
                missing_pct = self.df[col].isna().sum() / len(self.df)
                
                if missing_pct > 0.5:
                    # Drop column if >50% missing
                    self.df = self.df.drop(columns=[col])
                    self._log(f"Dropped column '{col}' ({missing_pct:.1%} missing)")
                
                elif missing_pct > 0:
                    # Fill based on type
                    if pd.api.types.is_numeric_dtype(self.df[col]):
                        self.df[col].fillna(self.df[col].median(), inplace=True)
                    elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                        # Forward fill for datetime
                        self.df[col].fillna(method='ffill', inplace=True)
                    else:
                        # Mode for categorical
                        mode_val = self.df[col].mode()
                        if len(mode_val) > 0:
                            self.df[col].fillna(mode_val[0], inplace=True)
        
        missing_after = self.df.isna().sum().sum()
        if missing_before > 0:
            self._log(f"Handled {missing_before - missing_after} missing values")
    
    def _remove_duplicates(self):
        """Remove duplicate rows"""
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        removed = before - len(self.df)
        
        if removed > 0:
            self._log(f"Removed {removed} duplicate rows")
    
    def _handle_outliers(self, method: str):
        """Handle outliers in numeric columns"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            before = len(self.df)
            
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing
                self.df[col] = self.df[col].clip(lower=lower, upper=upper)
            
            elif method == 'zscore':
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                self.df = self.df[z_scores < 3]
            
            after = len(self.df)
            if before - after > 0:
                self._log(f"Handled outliers in '{col}': {before - after} rows affected")
    
    def _validate_data(self):
        """Final validation checks"""
        # Check for remaining issues
        issues = []
        
        # Check for infinite values
        if self.df.select_dtypes(include=[np.number]).isin([np.inf, -np.inf]).any().any():
            # Replace inf with NaN
            self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
            issues.append("Replaced infinite values with NaN")
        
        # Check for very long strings (potential data quality issue)
        for col in self.df.select_dtypes(include=['object']).columns:
            max_len = self.df[col].astype(str).str.len().max()
            if max_len > 1000:
                issues.append(f"Column '{col}' has very long strings (max: {max_len})")
        
        if issues:
            self._log("Validation warnings: " + "; ".join(issues))
    
    def _log(self, message: str):
        """Log cleaning actions"""
        self.cleaning_report.append(message)
    
    def _print_report(self):
        """Print cleaning report"""
        print("\n" + "="*60)
        print("DATA CLEANING REPORT")
        print("="*60)
        for item in self.cleaning_report:
            print(f"✓ {item}")
        print("="*60 + "\n")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get data summary after cleaning"""
        return {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isna().sum().to_dict(),
            'memory_usage': f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        }


# Easy-to-use function
def clean_data(df: pd.DataFrame, 
               aggressive: bool = False,
               **kwargs) -> pd.DataFrame:
    """
    One-line data cleaning
    
    Args:
        df: Input DataFrame
        aggressive: If True, uses more aggressive cleaning (drops more data)
        **kwargs: Additional cleaning options
    
    Returns:
        Cleaned DataFrame
    """
    cleaner = DataCleaner(df)
    
    if aggressive:
        return cleaner.clean(
            handle_missing='drop',
            handle_duplicates=True,
            handle_outliers=True,
            outlier_method='zscore',
            standardize_text=True,
            convert_types=True,
            **kwargs
        )
    else:
        return cleaner.clean(
            handle_missing='smart',
            handle_duplicates=True,
            handle_outliers=True,
            outlier_method='iqr',
            standardize_text=True,
            convert_types=True,
            **kwargs
        )