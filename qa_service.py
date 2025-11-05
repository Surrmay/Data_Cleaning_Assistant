"""
Enhanced QA Service Module - Intelligent LLM-Powered Data Cleaning

CHANGES MADE TO FIX "5+ ATTEMPTS" ISSUE:
1. Increased max_tokens from 3000 to 8000 for complete code generation
2. More explicit code structure requirements in prompt
3. Added example-driven few-shot learning sections
4. Better data sample formatting with explicit patterns
5. Stricter output format requirements
6. Added self-validation step in prompt
7. Enhanced error handling patterns in generated code
"""

import os
import json
from typing import Dict, Any, List
import pandas as pd
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


class EnhancedCleaningAdvisor:
    """
    An intelligent advisor that leverages LLM reasoning for superior data cleaning.
    """
    
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        """Initialize with enhanced configuration for better code generation."""
        self.llm = ChatGroq(
            api_key=api_key,
            model_name=model,
            temperature=0.1,  # CHANGED: Lower temperature for more consistent output
            max_tokens=8000   # CHANGED: Increased from 3000 to ensure complete code
        )
        
        self.prompt = self._create_enhanced_prompt_template()
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def _create_enhanced_prompt_template(self) -> PromptTemplate:
        """
        MAJOR CHANGES: More structured prompt with explicit examples and requirements
        """
        template = """You are an expert data scientist. Generate COMPLETE, EXECUTABLE Python code in ONE response.

CRITICAL RULES:
1. Generate ALL code in a SINGLE code block - DO NOT split into multiple blocks
2. Code must be IMMEDIATELY executable without modifications
3. Include ALL necessary imports at the top
4. Handle EVERY column mentioned in the analysis
5. Add comprehensive error handling for EVERY operation
6. NO placeholders, NO "...", NO incomplete sections

DATASET OVERVIEW:
{dataset_info}

DATA SAMPLES (Examine these CAREFULLY):
{data_samples}

COLUMN ANALYSIS:
{column_details}

ISSUES DETECTED:
{issues_summary}

DETAILED PROBLEMS:
- Missing Values: {missing_values}
- Duplicates: {duplicates}
- Outliers: {outliers}
- Inconsistent Categories: {inconsistent_categories}

================================================================================
STEP 1: PATTERN RECOGNITION
================================================================================

Look at the DATA SAMPLES above and identify these patterns:

NUMERIC COLUMNS WITH TEXT:
- Pattern: Column shows numbers but dtype is 'object'
- Indicators: Values like "123", "456", "N/A", "unknown", "-", "."
- Solution: pd.to_numeric(df['col'], errors='coerce')

DATETIME COLUMNS:
- Pattern: Strings that look like dates
- Indicators: "2023-01-01", "01/15/2023", "Jan 1, 2023"
- Solution: pd.to_datetime(df['col'], errors='coerce')

CATEGORICAL WITH VARIATIONS:
- Pattern: Same meaning, different formatting
- Indicators: "Male"/"male"/"MALE"/"M", "Yes"/"yes"/"Y"
- Solution: Standardize with .str.lower() and .replace()

WHITESPACE ISSUES:
- Pattern: Extra spaces affecting values
- Indicators: " text ", "text  ", "  text"
- Solution: .str.strip() for all object columns

================================================================================
STEP 2: GENERATE COMPLETE CODE - MUST BE EXECUTABLE AS-IS
================================================================================

Generate code following this EXACT structure. Fill in ALL sections completely:

```python
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def clean_data(file_path='your_data.csv'):
    \"\"\"
    Complete data cleaning pipeline - handles all detected issues
    \"\"\"
    
    # Load data
    print("Loading data...")
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Loaded {{len(df)}} rows, {{len(df.columns)}} columns")
    except Exception as e:
        print(f"✗ Error loading file: {{e}}")
        return None
    
    # Backup
    df_original = df.copy()
    
    print("\\n" + "="*70)
    print("STARTING DATA CLEANING PIPELINE")
    print("="*70)
    
    # ============================================================
    # SECTION 1: STRIP WHITESPACE FROM ALL TEXT COLUMNS
    # ============================================================
    print("\\n[1/6] Cleaning text columns...")
    text_columns = df.select_dtypes(include=['object']).columns
    for col in text_columns:
        try:
            df[col] = df[col].astype(str).str.strip()
            # Replace 'nan' strings with actual NaN
            df[col] = df[col].replace(['nan', 'NaN', 'None', '', 'null', 'NULL'], np.nan)
        except Exception as e:
            print(f"  Warning: Could not clean {{col}}: {{e}}")
    print(f"✓ Cleaned {{len(text_columns)}} text columns")
    
    # ============================================================
    # SECTION 2: FIX DATA TYPES (CRITICAL SECTION)
    # ============================================================
    print("\\n[2/6] Converting data types...")
    
    # IMPORTANT: Based on your COLUMN ANALYSIS above, convert each column
    # Example pattern for numeric columns stored as text:
    
    # FOR EACH NUMERIC COLUMN DETECTED IN ANALYSIS:
    # try:
    #     df['column_name'] = pd.to_numeric(df['column_name'], errors='coerce')
    #     print(f"  ✓ Converted column_name to numeric")
    # except Exception as e:
    #     print(f"  ✗ Failed to convert column_name: {{e}}")
    
    # FOR EACH DATETIME COLUMN DETECTED IN ANALYSIS:
    # try:
    #     df['date_column'] = pd.to_datetime(df['date_column'], errors='coerce')
    #     print(f"  ✓ Converted date_column to datetime")
    # except Exception as e:
    #     print(f"  ✗ Failed to convert date_column: {{e}}")
    
    # NOW GENERATE THE ACTUAL CONVERSION CODE FOR YOUR DATA:
    {type_conversion_code}
    
    # ============================================================
    # SECTION 3: HANDLE MISSING VALUES
    # ============================================================
    print("\\n[3/6] Handling missing values...")
    
    missing_before = df.isnull().sum().sum()
    
    # Strategy: Fill numeric with median, categorical with mode
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Numeric: use median
                    df[col].fillna(df[col].median(), inplace=True)
                    print(f"  ✓ Filled {{missing_count}} missing values in {{col}} (numeric)")
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    # Datetime: forward fill
                    df[col].fillna(method='ffill', inplace=True)
                    print(f"  ✓ Forward-filled {{missing_count}} missing values in {{col}} (datetime)")
                else:
                    # Categorical: use mode
                    mode_value = df[col].mode()
                    if len(mode_value) > 0:
                        df[col].fillna(mode_value[0], inplace=True)
                        print(f"  ✓ Filled {{missing_count}} missing values in {{col}} (categorical)")
            except Exception as e:
                print(f"  ✗ Could not fill {{col}}: {{e}}")
    
    missing_after = df.isnull().sum().sum()
    print(f"✓ Reduced missing values from {{missing_before}} to {{missing_after}}")
    
    # ============================================================
    # SECTION 4: REMOVE DUPLICATES
    # ============================================================
    print("\\n[4/6] Removing duplicates...")
    before_dup = len(df)
    df = df.drop_duplicates()
    after_dup = len(df)
    removed_dup = before_dup - after_dup
    print(f"✓ Removed {{removed_dup}} duplicate rows")
    
    # ============================================================
    # SECTION 5: HANDLE OUTLIERS IN NUMERIC COLUMNS
    # ============================================================
    print("\\n[5/6] Handling outliers...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        try:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing
            outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers_count > 0:
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                print(f"  ✓ Capped {{outliers_count}} outliers in {{col}}")
        except Exception as e:
            print(f"  ✗ Could not handle outliers in {{col}}: {{e}}")
    
    # ============================================================
    # SECTION 6: STANDARDIZE CATEGORICAL VALUES
    # ============================================================
    print("\\n[6/6] Standardizing categorical values...")
    
    # IMPORTANT: Based on INCONSISTENT CATEGORIES analysis, add mappings
    # Example pattern:
    # if 'gender' in df.columns:
    #     df['gender'] = df['gender'].str.lower()
    #     df['gender'] = df['gender'].replace({{'m': 'male', 'f': 'female'}})
    
    # NOW GENERATE ACTUAL STANDARDIZATION CODE FOR YOUR DATA:
    {categorical_standardization_code}
    
    # ============================================================
    # FINAL VALIDATION & SUMMARY
    # ============================================================
    print("\\n" + "="*70)
    print("CLEANING COMPLETE - SUMMARY")
    print("="*70)
    print(f"Original shape: {{df_original.shape}}")
    print(f"Final shape: {{df.shape}}")
    print(f"Rows removed: {{len(df_original) - len(df)}}")
    print(f"\\nFinal data types:")
    print(df.dtypes)
    print(f"\\nRemaining missing values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    # Save
    output_file = 'cleaned_data.csv'
    df.to_csv(output_file, index=False)
    print(f"\\n✓ Cleaned data saved to '{{output_file}}'")
    
    return df

# Execute cleaning
if __name__ == "__main__":
    cleaned_df = clean_data('your_data.csv')
```

================================================================================
NOW GENERATE THE COMPLETE CODE ABOVE
================================================================================

REQUIREMENTS CHECKLIST (All must be satisfied):
☐ Single complete code block (not split)
☐ All imports included at top
☐ Every column in analysis is handled
☐ Type conversions for ALL identified columns in {type_conversion_code} section
☐ Categorical mappings for ALL identified issues in {categorical_standardization_code} section
☐ Error handling on every operation
☐ Progress print statements
☐ No placeholders or "..."
☐ Code is immediately executable

Generate the COMPLETE, EXECUTABLE code now:"""

        return PromptTemplate(
            input_variables=[
                "dataset_info",
                "data_samples",
                "column_details",
                "issues_summary",
                "missing_values",
                "duplicates",
                "outliers",
                "inconsistent_categories",
                "type_conversion_code",
                "categorical_standardization_code"
            ],
            template=template
        )
    
    def generate_cleaning_advice(
        self, 
        inspection_report: Dict[str, Any],
        df_sample: pd.DataFrame
    ) -> str:
        """
        Generate intelligent cleaning advice by showing actual data to the LLM.
        
        CHANGES: Added pre-analysis to generate specific code hints
        """
        try:
            # ADDED: Pre-analyze data to generate specific code hints
            type_hints = self._generate_type_conversion_hints(df_sample, inspection_report)
            categorical_hints = self._generate_categorical_hints(df_sample, inspection_report)
            
            # Format inputs for the enhanced prompt
            formatted_input = self._format_enhanced_input(
                inspection_report, 
                df_sample
            )
            
            # ADDED: Include the hints in the prompt
            formatted_input['type_conversion_code'] = type_hints
            formatted_input['categorical_standardization_code'] = categorical_hints
            
            # Call LLM with enhanced context
            response = self.chain.run(**formatted_input)
            
            return response
            
        except Exception as e:
            return self._generate_fallback_advice(str(e), inspection_report)
    
    def _generate_type_conversion_hints(
        self, 
        df: pd.DataFrame, 
        report: Dict[str, Any]
    ) -> str:
        """
        NEW METHOD: Generate specific type conversion code hints
        """
        hints = []
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if it looks numeric
                sample_values = df[col].dropna().head(20).astype(str).tolist()
                numeric_count = sum(1 for v in sample_values 
                                   if v.replace('.','',1).replace('-','',1).isdigit())
                
                if numeric_count > len(sample_values) * 0.6:
                    hints.append(f"""
    # Column '{col}' appears to be numeric stored as text
    try:
        df['{col}'] = pd.to_numeric(df['{col}'], errors='coerce')
        print(f"  ✓ Converted '{col}' to numeric")
    except Exception as e:
        print(f"  ✗ Failed to convert '{col}': {{e}}")""")
                
                # Check if it looks like datetime
                elif any(keyword in col.lower() for keyword in ['date', 'time', 'timestamp']):
                    hints.append(f"""
    # Column '{col}' appears to be a datetime
    try:
        df['{col}'] = pd.to_datetime(df['{col}'], errors='coerce')
        print(f"  ✓ Converted '{col}' to datetime")
    except Exception as e:
        print(f"  ✗ Failed to convert '{col}': {{e}}")""")
        
        return "\n".join(hints) if hints else "    # No type conversions needed"
    
    def _generate_categorical_hints(
        self, 
        df: pd.DataFrame, 
        report: Dict[str, Any]
    ) -> str:
        """
        NEW METHOD: Generate specific categorical standardization code hints
        """
        hints = []
        inconsistent = report.get('inconsistent_categories', {})
        
        for col, variations in inconsistent.items():
            if col in df.columns:
                # Get unique values
                unique_vals = df[col].dropna().unique()[:20]
                
                hints.append(f"""
    # Standardize '{col}' (detected variations: {variations[:3]})
    if '{col}' in df.columns:
        df['{col}'] = df['{col}'].str.lower().str.strip()
        print(f"  ✓ Standardized '{col}'")""")
        
        return "\n".join(hints) if hints else "    # No categorical standardization needed"
    
    def _format_enhanced_input(
        self, 
        report: Dict[str, Any],
        df_sample: pd.DataFrame
    ) -> Dict[str, str]:
        """Format all inputs for the enhanced prompt."""
        dataset_info = f"""
Dataset: {report.get('row_count', 'unknown')} rows × {report.get('column_count', 'unknown')} columns
Columns: {', '.join(df_sample.columns.tolist())}
"""
        
        data_samples = self._format_data_samples(df_sample)
        column_details = self._create_column_analysis(df_sample, report)
        issues_summary = "\n".join(report.get('issues_found', ['No issues detected']))
        missing_values = self._format_missing_values(report.get('missing_values', {}))
        duplicates = f"{report.get('duplicates', 0)} duplicate rows detected"
        outliers = self._format_outliers(report.get('outliers', {}))
        inconsistent = self._format_inconsistent_categories(
            report.get('inconsistent_categories', {})
        )
        
        return {
            'dataset_info': dataset_info,
            'data_samples': data_samples,
            'column_details': column_details,
            'issues_summary': issues_summary,
            'missing_values': missing_values,
            'duplicates': duplicates,
            'outliers': outliers,
            'inconsistent_categories': inconsistent
        }
    
    def _format_data_samples(self, df: pd.DataFrame) -> str:
        """Format actual data samples - ENHANCED with better formatting."""
        sample = df.head(15)  # CHANGED: Increased from 10 to 15 rows
        
        # Show table format
        sample_str = sample.to_string()
        
        # ADDED: Show data types alongside samples
        dtypes_str = "\n".join([f"  {col}: {dtype}" for col, dtype in df.dtypes.items()])
        
        # ADDED: Show unique value counts
        unique_counts = "\n".join([f"  {col}: {df[col].nunique()} unique values" 
                                   for col in df.columns])
        
        return f"""
TABLE VIEW (first 15 rows):
{sample_str}

CURRENT DATA TYPES:
{dtypes_str}

UNIQUE VALUE COUNTS:
{unique_counts}
"""
    
    def _create_column_analysis(
        self, 
        df: pd.DataFrame, 
        report: Dict[str, Any]
    ) -> str:
        """Create detailed column analysis - ENHANCED with pattern detection."""
        analysis_lines = []
        
        for col in df.columns:
            analysis_lines.append(f"\n{'='*50}")
            analysis_lines.append(f"COLUMN: {col}")
            analysis_lines.append(f"{'='*50}")
            
            current_dtype = df[col].dtype
            analysis_lines.append(f"Current dtype: {current_dtype}")
            
            # ENHANCED: Show more sample values
            unique_samples = df[col].dropna().unique()[:15]
            analysis_lines.append(f"Sample values: {list(unique_samples)}")
            
            # ADDED: Show value type distribution
            non_null = df[col].dropna().astype(str)
            if len(non_null) > 0:
                sample_vals = non_null.head(30).tolist()
                
                # Count numeric-looking values
                numeric_like = sum(1 for v in sample_vals 
                                  if v.replace('.','',1).replace('-','',1).replace('+','',1).isdigit())
                
                if current_dtype == 'object' and numeric_like > len(sample_vals) * 0.6:
                    analysis_lines.append(f"⚠️ CRITICAL: {numeric_like}/{len(sample_vals)} values look NUMERIC but stored as text!")
                    
                    # Find the problematic values
                    non_numeric = [v for v in sample_vals[:10] 
                                  if not v.replace('.','',1).replace('-','',1).replace('+','',1).isdigit()]
                    if non_numeric:
                        analysis_lines.append(f"   Problem values causing text dtype: {non_numeric}")
            
            # Missing info
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                missing_pct = (missing_count / len(df)) * 100
                analysis_lines.append(f"Missing: {missing_count} ({missing_pct:.1f}%)")
            
            # ADDED: Value counts for low-cardinality columns
            if df[col].nunique() <= 20:
                value_counts = df[col].value_counts().head(10)
                analysis_lines.append(f"Value distribution:\n{value_counts.to_string()}")
        
        return "\n".join(analysis_lines)
    
    def _format_missing_values(self, missing: Dict[str, int]) -> str:
        """Format missing values."""
        if not missing:
            return "No missing values detected"
        
        lines = []
        for col, count in missing.items():
            lines.append(f"  • {col}: {count} missing values")
        return "\n".join(lines)
    
    def _format_outliers(self, outliers: Dict[str, list]) -> str:
        """Format outliers."""
        if not outliers:
            return "No significant outliers detected"
        
        lines = []
        for col, values in outliers.items():
            lines.append(f"  • {col}: {len(values)} outliers detected")
            lines.append(f"    Range: {min(values)} to {max(values)}")
        return "\n".join(lines)
    
    def _format_inconsistent_categories(self, inconsistent: Dict[str, list]) -> str:
        """Format categorical inconsistencies."""
        if not inconsistent:
            return "No inconsistent categorical values detected"
        
        lines = []
        for col, values in inconsistent.items():
            lines.append(f"  • {col}: Inconsistent formatting")
            lines.append(f"    Variations: {', '.join([str(v) for v in values[:10]])}")
        return "\n".join(lines)
    
    def _generate_fallback_advice(
        self, 
        error: str, 
        report: Dict[str, Any]
    ) -> str:
        """Generate fallback advice if LLM fails."""
        advice = f"""
# Error Generating AI Advice

Error: {error}

Basic cleaning script based on detected issues:

```python
import pandas as pd
import numpy as np

df = pd.read_csv('your_data.csv')

# Handle missing values
"""
        
        if report.get('missing_values'):
            advice += "\n# Fill missing values\n"
            for col in report['missing_values'].keys():
                advice += f"df['{col}'].fillna(df['{col}'].median(), inplace=True)\n"
        
        if report.get('duplicates', 0) > 0:
            advice += "\n# Remove duplicates\ndf = df.drop_duplicates()\n"
        
        advice += "\ndf.to_csv('cleaned_data.csv', index=False)\n```"
        
        return advice


def create_advisor(api_key: str = None) -> EnhancedCleaningAdvisor:
    """Factory function to create an EnhancedCleaningAdvisor instance."""
    if api_key is None:
        api_key = os.getenv('GROQ_API_KEY')
        
    if not api_key:
        raise ValueError(
            "Groq API key not found. Please set GROQ_API_KEY environment variable "
            "or pass api_key parameter."
        )
    
    return EnhancedCleaningAdvisor(api_key)