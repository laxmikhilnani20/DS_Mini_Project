#!/bin/bash

# African Import Analysis - Streamlit App Launcher
# Run this script to start the dashboard

echo "🌍 African Import Analysis - Starting Streamlit Dashboard..."
echo ""
echo "📦 Checking dependencies..."

# Check if streamlit is installed
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "⚠️  Streamlit not found. Installing dependencies..."
    pip3 install -r requirements.txt
else
    echo "✅ All dependencies found"
fi

echo ""
echo "🚀 Launching dashboard..."
echo ""
echo "📊 Dashboard will open in your browser at: http://localhost:8501"
echo ""
echo "Features:"
echo "  • 32 ML Models (Regression + Classification + Clustering + Deep Learning)"
echo "  • 10 Years of Data (2015-2025)"
echo "  • 139,566 Transactions"
echo "  • Interactive Visualizations"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "-------------------------------------------------------------------"
echo ""

# Run the Streamlit app
python3 -m streamlit run streamlit_app.py
