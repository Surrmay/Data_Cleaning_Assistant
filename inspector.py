"""
Data Inspector Module - The Detective of Data Quality

This module performs comprehensive analysis of CSV data to detect
various quality issues. Think of it as a health checkup for your dataset.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class InspectionReport:
    """
    A structured container for all inspection findings.
    
    Using a dataclass makes our code cleaner and provides
    automatic initialization and type hints.
    """
    missing_values: Dict[str, int]
    data_types: Dict[str, str]
    duplicates: int
    outliers: Dict[str, List[Any]]
    inconsistent_categories: Dict[str, List[str]]
    summary_stats: Dict[str, Dict[str, float]]
    issues_found: List[str]


class DataInspector:
    """
    The main inspection engine that analyzes DataFrames for quality issues.
    
    This class follows the Single Responsibility Principle - it focuses
    solely on detecting problems, not fixing them. The fixing part comes
    later through the LLM-generated code.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the inspector with a DataFrame to analyze.
        
        Args:
            df: The pandas DataFrame to inspect
        """
        self.df = df
        self.report = None
    
    def inspect(self) -> InspectionReport:
        """
        Run all inspection checks and compile a comprehensive report.
        
        This is the main entry point that orchestrates all individual checks.
        The order matters - we want to detect simpler issues before complex ones.
        
        Returns:
            InspectionReport with all findings
        """
        issues = []
        
        # Check 1: Missing values (most common issue)
        missing = self._check_missing_values()
        if missing:
            issues.append(f"Found missing values in {len(missing)} columns")
        
        # Check 2: Data type problems
        dtypes = self._check_data_types()
        
        # Check 3: Duplicate rows
        dup_count = self._check_duplicates()
        if dup_count > 0:
            issues.append(f"Found {dup_count} duplicate rows")
        
        # Check 4: Outliers in numeric columns
        outliers = self._detect_outliers()
        if outliers:
            issues.append(f"Detected outliers in {len(outliers)} columns")
        
        # Check 5: Inconsistent categorical values
        inconsistent = self._check_categorical_consistency()
        if inconsistent:
            issues.append(f"Found inconsistent categories in {len(inconsistent)} columns")
        
        # Check 6: Basic statistics for numeric columns
        stats = self._compute_summary_stats()
        
        self.report = InspectionReport(
            missing_values=missing,
            data_types=dtypes,
            duplicates=dup_count,
            outliers=outliers,
            inconsistent_categories=inconsistent,
            summary_stats=stats,
            issues_found=issues if issues else ["No major issues detected"]
        )
        
        return self.report
    
    def _check_missing_values(self) -> Dict[str, int]:
        """
        Detect columns with missing (NaN, None, empty) values.
        
        Missing data is the #1 problem in real-world datasets.
        We count how many values are missing per column.
        
        Returns:
            Dictionary mapping column names to count of missing values
        """
        missing = {}
        for col in self.df.columns:
            null_count = self.df[col].isna().sum()
            if null_count > 0:
                missing[col] = int(null_count)
        return missing
    
    def _check_data_types(self) -> Dict[str, str]:
        """
        Get the data type of each column.
        
        Understanding data types is crucial because:
        - Numeric data stored as strings won't calculate correctly
        - Dates stored as strings can't be sorted properly
        - We need types to suggest appropriate cleaning methods
        
        Returns:
            Dictionary mapping column names to their data types
        """
        return {col: str(dtype) for col, dtype in self.df.dtypes.items()}
    
    def _check_duplicates(self) -> int:
        """
        Count duplicate rows in the DataFrame.
        
        Duplicates can occur from:
        - Data entry errors
        - Multiple system exports
        - Join operations gone wrong
        
        Returns:
            Number of duplicate rows found
        """
        return int(self.df.duplicated().sum())
    
    def _detect_outliers(self) -> Dict[str, List[Any]]:
        """
        Detect outliers in numeric columns using the IQR method.
        
        The IQR (Interquartile Range) method is a robust statistical
        technique. It defines outliers as values that fall below Q1-1.5*IQR
        or above Q3+1.5*IQR. This catches extreme values that might
        indicate data errors or anomalies.
        
        Why IQR? Unlike standard deviation methods, IQR is not affected
        by the outliers themselves, making it more reliable.
        
        Returns:
            Dictionary mapping column names to lists of outlier values
        """
        outliers = {}
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Calculate quartiles
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier boundaries
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Find outliers
            column_outliers = self.df[
                (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            ][col].tolist()
            
            # Only report if we found outliers (limit to first 10 for brevity)
            if column_outliers:
                outliers[col] = column_outliers[:10]
        
        return outliers
    
    def _check_categorical_consistency(self) -> Dict[str, List[str]]:
        """
        Detect inconsistent categorical values (e.g., "USA" vs "usa" vs "U.S.A").
        
        This is a subtle but common problem. Humans enter data inconsistently:
        - Different capitalizations ("Male" vs "male")
        - Abbreviations vs full forms ("CA" vs "California")
        - Whitespace variations ("New York" vs "New York ")
        
        We flag columns where similar-looking values might be duplicates.
        
        Returns:
            Dictionary mapping column names to lists of potentially inconsistent values
        """
        inconsistent = {}
        
        # Focus on object (string) columns with reasonable cardinality
        object_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in object_cols:
            unique_values = self.df[col].dropna().unique()
            
            # Only check columns with 2-50 unique values (likely categorical)
            if 2 <= len(unique_values) <= 50:
                # Look for values that differ only in case or whitespace
                normalized = {}
                for val in unique_values:
                    # Normalize: lowercase and strip whitespace
                    norm_val = str(val).lower().strip()
                    if norm_val in normalized:
                        normalized[norm_val].append(val)
                    else:
                        normalized[norm_val] = [val]
                
                # Flag groups with multiple variations
                variations = [
                    vals for vals in normalized.values() if len(vals) > 1
                ]
                
                if variations:
                    # Flatten the list of variations
                    all_inconsistent = [v for group in variations for v in group]
                    inconsistent[col] = all_inconsistent[:20]  # Limit output
        
        return inconsistent
    
    def _compute_summary_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate descriptive statistics for numeric columns.
        
        These statistics give us a quick snapshot of the data distribution.
        They help the LLM understand the scale and range of the data when
        suggesting cleaning strategies.
        
        Returns:
            Dictionary with statistics for each numeric column
        """
        stats = {}
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            stats[col] = {
                'mean': float(self.df[col].mean()),
                'median': float(self.df[col].median()),
                'std': float(self.df[col].std()),
                'min': float(self.df[col].min()),
                'max': float(self.df[col].max())
            }
        
        return stats