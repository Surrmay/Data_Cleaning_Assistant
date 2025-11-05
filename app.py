"""
Data Cleaning Assistant - Main Streamlit Application

This is the user-facing interface where all the magic comes together.
We orchestrate the flow from file upload to cleaned data download.
"""

import streamlit as st
import pandas as pd
import os
from io import StringIO

# Import our custom modules
import utils
from inspector import DataInspector
from qa_service import create_advisor

# Configure the Streamlit page - this must be the first Streamlit command
st.set_page_config(
    page_title="AI Data Cleaning Assistant",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_session_state():
    """
    Initialize Streamlit session state variables.
    
    Session state persists data across reruns (Streamlit reruns the entire
    script on every interaction). This prevents losing our analysis results
    when the user clicks a button.
    """
    if 'df_original' not in st.session_state:
        st.session_state.df_original = None
    if 'df_cleaned' not in st.session_state:
        st.session_state.df_cleaned = None
    if 'inspection_report' not in st.session_state:
        st.session_state.inspection_report = None
    if 'cleaning_advice' not in st.session_state:
        st.session_state.cleaning_advice = None


def show_header():
    """Display the application header with branding and description."""
    st.title("🧹 AI Data Cleaning Assistant")
    st.markdown("""
    **Stop wasting time on messy data!** Upload your CSV file and let AI identify 
    quality issues, explain them in plain English, and generate ready-to-run 
    cleaning code.
    
    ---
    """)


def show_sidebar():
    """
    Display the sidebar with app information and settings.
    
    The sidebar is perfect for meta-information that doesn't clutter
    the main workflow.
    """
    with st.sidebar:
        st.header("📊 About")
        st.info("""
        This assistant helps you:
        - Detect missing values
        - Identify data type issues
        - Find duplicates and outliers
        - Spot inconsistent categories
        - Generate cleaning code automatically
        """)
        
        st.header("🔐 Privacy")
        st.success("""
        Your data is processed locally and sent only to the Groq API 
        for analysis. No data is stored on our servers.
        """)
        
        st.header("⚙️ Settings")
        
        # Allow users to input API key if not in environment
        api_key = os.getenv('GROQ_API_KEY', '')
        api_key_input = st.text_input(
            "Groq API Key",
            value=api_key if api_key else "",
            type="password",
            help="Enter your Groq API key. Get one at console.groq.com"
        )
        
        # Store API key in session state
        if api_key_input:
            st.session_state.api_key = api_key_input
        
        st.markdown("---")
        st.caption("Built with Streamlit, LangChain, and Groq")


def handle_file_upload():
    """
    Handle the file upload process and initial data loading.
    
    This function manages the critical first step of getting data into
    the application. We validate the file before proceeding.
    
    Returns:
        tuple: (DataFrame, success_flag)
    """
    st.header("📤 Step 1: Upload Your CSV")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file to analyze. Maximum 100,000 rows."
    )
    
    if uploaded_file is not None:
        with st.spinner("Loading your data..."):
            # Use our utility function to load the CSV
            df, error = utils.load_csv(uploaded_file)
            
            if error:
                st.error(f"❌ {error}")
                return None, False
            
            # Validate the DataFrame
            is_valid, validation_error = utils.validate_dataframe(df)
            if not is_valid:
                st.error(f"❌ {validation_error}")
                return None, False
            
            # Show success and preview
            st.success(f"✅ Successfully loaded {len(df)} rows and {len(df.columns)} columns")
            
            with st.expander("👀 Preview your data (first 10 rows)"):
                st.dataframe(df.head(10))
            
            return df, True
    
    return None, False


def perform_inspection(df: pd.DataFrame):
    """
    Run the data inspection and display results.
    
    This is where our DataInspector class shines, systematically
    checking for various data quality issues.
    
    Args:
        df: The DataFrame to inspect
    """
    st.header("🔍 Step 2: Data Quality Analysis")
    
    with st.spinner("Analyzing your data for quality issues..."):
        # Create inspector and run analysis
        inspector = DataInspector(df)
        report = inspector.inspect()
        
        # Store in session state
        st.session_state.inspection_report = report
        
    # Display results in organized sections
    st.subheader("Analysis Results")
    
    # Overview metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        missing_cols = len(report.missing_values)
        st.metric("Columns with Missing Data", missing_cols)
    with col4:
        st.metric("Duplicate Rows", report.duplicates)
    
    # Issues summary
    st.subheader("Issues Detected")
    for issue in report.issues_found:
        st.warning(f"⚠️ {issue}")
    
    # Detailed findings in expandable sections
    if report.missing_values:
        with st.expander("📋 Missing Values Details"):
            for col, count in report.missing_values.items():
                percentage = (count / len(df)) * 100
                st.write(f"**{col}**: {count} missing ({percentage:.1f}%)")
    
    if report.outliers:
        with st.expander("📊 Outliers Detected"):
            for col, values in report.outliers.items():
                st.write(f"**{col}**: {len(values)} outliers")
                st.write(f"Examples: {values[:5]}")
    
    if report.inconsistent_categories:
        with st.expander("🏷️ Inconsistent Categories"):
            for col, values in report.inconsistent_categories.items():
                st.write(f"**{col}**: Variations found")
                st.write(f"Examples: {', '.join([str(v) for v in values[:5]])}")
    
    with st.expander("📈 Summary Statistics"):
        if report.summary_stats:
            stats_df = pd.DataFrame(report.summary_stats).T
            st.dataframe(stats_df)
        else:
            st.info("No numeric columns found for statistics")


def generate_ai_advice():
    """
    Call the LLM to generate cleaning recommendations and code.
    
    This is the AI magic moment - we transform structured findings
    into actionable human-readable advice and executable code.
    
    ENHANCED: Now passes actual data samples to the LLM for better reasoning.
    """
    st.header("🤖 Step 3: AI-Generated Cleaning Recommendations")
    
    # Check if we have an API key
    api_key = st.session_state.get('api_key') or os.getenv('GROQ_API_KEY')
    if not api_key:
        st.error("❌ Groq API key not found. Please enter it in the sidebar.")
        st.info("💡 Get a free API key at: https://console.groq.com")
        return
    
    if st.button("✨ Generate Cleaning Recommendations", type="primary"):
        with st.spinner("AI is analyzing your data deeply and generating intelligent recommendations..."):
            try:
                # Create the enhanced advisor
                advisor = create_advisor(api_key)
                
                # Prepare the report for the LLM
                report_dict = {
                    'row_count': len(st.session_state.df_original),
                    'column_count': len(st.session_state.df_original.columns),
                    'missing_values': st.session_state.inspection_report.missing_values,
                    'data_types': st.session_state.inspection_report.data_types,
                    'duplicates': st.session_state.inspection_report.duplicates,
                    'outliers': st.session_state.inspection_report.outliers,
                    'inconsistent_categories': st.session_state.inspection_report.inconsistent_categories,
                    'issues_found': st.session_state.inspection_report.issues_found
                }
                
                # CRITICAL IMPROVEMENT: Pass actual data sample to LLM
                # This allows the LLM to see patterns and generate context-aware code
                df_sample = st.session_state.df_original.head(15)
                
                # Generate advice with enhanced context
                advice = advisor.generate_cleaning_advice(report_dict, df_sample)
                st.session_state.cleaning_advice = advice
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Tip: Check your API key and internet connection")
                return
    
    # Display the advice if available
    if st.session_state.cleaning_advice:
        st.success("✅ Recommendations generated successfully!")
        
        # Display the advice in a nice formatted way
        st.markdown(st.session_state.cleaning_advice)
        
        # Offer to download the advice
        st.download_button(
            label="📥 Download Recommendations",
            data=st.session_state.cleaning_advice,
            file_name="cleaning_recommendations.txt",
            mime="text/plain"
        )


def main():
    """
    Main application orchestrator.
    
    This function coordinates the entire user flow from start to finish.
    Think of it as the director of a play, making sure each scene
    happens in the right order.
    """
    # Initialize
    initialize_session_state()
    show_header()
    show_sidebar()
    
    # Step 1: File Upload
    df, success = handle_file_upload()
    
    if success and df is not None:
        # Store the original DataFrame
        st.session_state.df_original = df
        
        # Step 2: Inspect the data
        perform_inspection(df)
        
        # Step 3: Generate AI recommendations
        if st.session_state.inspection_report is not None:
            generate_ai_advice()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Made with ❤️ for data analysts | Powered by Groq & LangChain</p>
    </div>
    """, unsafe_allow_html=True)


# Entry point
if __name__ == "__main__":
    main()