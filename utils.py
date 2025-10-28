"""
Shared utility functions for the African Imports Analytics Platform
"""

import streamlit as st
import pandas as pd


def format_currency(value):
    """Smart formatting function for currency values"""
    if value >= 1e9:
        return f"${value/1e9:.2f}B"
    elif value >= 1e6:
        return f"${value/1e6:.2f}M"
    elif value >= 1e3:
        return f"${value/1e3:.2f}K"
    else:
        return f"${value:.2f}"


def validate_dataframe(df, min_rows=1):
    """
    Validate dataframe has sufficient data
    
    Args:
        df: DataFrame to validate
        min_rows: Minimum required rows
        
    Returns:
        bool: True if valid, False otherwise (stops app)
    """
    if df is None:
        st.error("❌ No data available. Please check data source.")
        st.stop()
        return False
    
    if len(df) == 0:
        st.error("❌ Dataset is empty. Please check filters or data source.")
        st.stop()
        return False
    
    if len(df) < min_rows:
        st.warning(f"⚠️ Only {len(df)} records available. Need at least {min_rows} for meaningful analysis.")
        st.info("💡 Try adjusting your filters to include more data.")
        st.stop()
        return False
    
    return True


def safe_operation(operation_name="operation"):
    """
    Decorator for safe execution with error handling
    
    Usage:
        @safe_operation("model training")
        def train_model():
            # your code
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                with st.spinner(f"⏳ Running {operation_name}..."):
                    result = func(*args, **kwargs)
                return result
            except Exception as e:
                st.error(f"❌ Error during {operation_name}: {str(e)}")
                st.info("💡 Try refreshing the page or adjusting your parameters.")
                with st.expander("🔍 Technical Details"):
                    st.code(str(e))
                return None
        return wrapper
    return decorator
