# ✅ ERROR HANDLING TEMPLATE
## Copy This Pattern to All Dashboards

---

## 📋 **What We Just Did:**

1. ✅ Added validation utilities to `utils.py`
2. ✅ Added error handling to `anomaly_detection.py` (example)
3. ⏳ **NOW:** Apply same pattern to 4 remaining dashboards

---

## 🔄 **COPY-PASTE THIS PATTERN:**

### **Step 1: Update Imports (Line 1-15)**

```python
# In EVERY dashboard file (data_overview.py, eda_explorer.py, ml_models.py, risk_optimization.py)

# OLD:
from utils import format_currency

# NEW:
from utils import format_currency, validate_dataframe
```

---

### **Step 2: Add Validation at Start of render() Function**

```python
def render(df):
    """Render dashboard"""
    
    # ADD THIS AT THE VERY START:
    # ============================
    # Validate data
    if not validate_dataframe(df, min_rows=10):
        return
    # ============================
    
    # ... rest of your code ...
```

**Why `min_rows=10`?**
- Most statistical operations need at least 10 data points
- For ML models, use `min_rows=100` instead
- Adjust based on what the dashboard does

---

### **Step 3: Wrap Expensive Operations in try-except**

```python
# BEFORE (risky):
df_filtered = df[complex_conditions]
result = expensive_computation(df_filtered)

# AFTER (safe):
try:
    with st.spinner("⏳ Computing..."):
        df_filtered = df[complex_conditions]
        
        # Check if filtering resulted in empty data
        if len(df_filtered) == 0:
            st.warning("⚠️ No data matches your filters")
            st.info("💡 Try adjusting your selection")
            return
        
        result = expensive_computation(df_filtered)
        
except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.info("💡 Try refreshing or adjusting your filters")
    with st.expander("🔍 Technical Details"):
        st.code(str(e))
    return  # Stop execution
```

---

## 📝 **DASHBOARD-BY-DASHBOARD CHECKLIST:**

### ✅ **1. data_overview.py**

**Add validation for:**
- [ ] Data filtering (lines ~100-120)
- [ ] CSV download generation
- [ ] Empty filter results

**Pattern:**
```python
def render(df):
    # Add at start
    if not validate_dataframe(df, min_rows=1):
        return
    
    # ... existing code ...
    
    # Around filtering section (~line 115):
    if len(filtered_df) == 0:
        st.warning("⚠️ No records match your filters")
        st.info("💡 Try adjusting date range or country selection")
        st.stop()
```

---

### ✅ **2. eda_explorer.py**

**Add validation for:**
- [ ] Country/commodity selection (empty results)
- [ ] Time series calculations (insufficient data points)
- [ ] Correlation matrices (need 2+ variables)

**Pattern:**
```python
def render(df):
    # Add at start
    if not validate_dataframe(df, min_rows=30):  # EDA needs more data
        return
    
    # After filtering by country/commodity:
    eda_filtered = df[
        (df['country_name'] == selected_country) &
        (df['commodity'] == selected_commodity)
    ]
    
    if len(eda_filtered) == 0:
        st.error("❌ No data for this country-commodity combination")
        st.info("💡 Try selecting a different combination")
        st.stop()
    
    if len(eda_filtered) < 10:
        st.warning("⚠️ Very limited data (<10 records)")
        st.info("Results may not be meaningful")
```

---

### ✅ **3. ml_models.py** (MOST IMPORTANT)

**Add validation for:**
- [ ] Minimum data for training (100+ rows recommended)
- [ ] Model training failures
- [ ] Prediction errors

**Pattern:**
```python
def render(df):
    # Add at start
    if not validate_dataframe(df, min_rows=100):
        st.warning("⚠️ ML models need at least 100 records for reliable training")
        return
    
    # Around model training (~line 150-200):
    try:
        with st.spinner("⏳ Training model... This may take a moment"):
            # Prepare data
            X_train, X_test, y_train, y_test = train_test_split(...)
            
            # Check train set size
            if len(X_train) < 50:
                st.error("❌ Insufficient training data after split")
                st.info("💡 Need at least 50 training samples")
                return
            
            # Train model
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            # Check for NaN in predictions
            if pd.isna(predictions).any():
                st.warning("⚠️ Model produced invalid predictions")
                return
                
    except Exception as e:
        st.error(f"❌ Model training failed: {str(e)}")
        st.info("💡 This may be due to:")
        st.markdown("""
        - Insufficient data variety
        - All target values are identical
        - Memory limitations with large datasets
        
        **Suggested actions:**
        - Try a different date range
        - Select fewer commodities
        - Use a simpler model type
        """)
        with st.expander("🔍 Technical Details"):
            st.code(str(e))
        return
```

---

### ✅ **4. risk_optimization.py**

**Add validation for:**
- [ ] Volatility calculations (need time series data)
- [ ] HHI calculations (need diversity)
- [ ] Seasonal analysis (need full year)

**Pattern:**
```python
def render(df):
    # Add at start
    if not validate_dataframe(df, min_rows=30):
        return
    
    # In Seasonal Timing tab:
    comm_data = df[df['commodity'] == selected_commodity]
    
    if len(comm_data) == 0:
        st.error("❌ No data for selected commodity")
        return
    
    if len(comm_data) < 12:
        st.warning("⚠️ Less than 12 months of data")
        st.info("💡 Seasonal analysis is most accurate with 1+ year of data")
    
    # Volatility calculations:
    try:
        monthly = comm_data.groupby(comm_data['date'].dt.to_period('M'))['value_dl'].sum()
        
        if len(monthly) < 3:
            st.warning("⚠️ Insufficient data points for volatility analysis")
            return
        
        cv = monthly.std() / monthly.mean() * 100
        
    except Exception as e:
        st.error(f"❌ Error calculating volatility: {str(e)}")
        return
```

---

## ⏱️ **TIME ESTIMATE:**

- ✅ `utils.py` - DONE (5 mins)
- ✅ `anomaly_detection.py` - DONE (15 mins)
- ⏳ `data_overview.py` - 10 mins
- ⏳ `eda_explorer.py` - 15 mins
- ⏳ `ml_models.py` - 20 mins (most complex)
- ⏳ `risk_optimization.py` - 10 mins

**TOTAL REMAINING: ~55 minutes**

---

## ✅ **TESTING CHECKLIST:**

After adding error handling to each dashboard, test:

1. **Empty filters:** Select filters that return 0 results
2. **Single record:** Select filters that return only 1 record
3. **Edge cases:** 
   - All values identical
   - Negative values
   - Missing required columns
4. **Large datasets:** Select "All" to test performance
5. **Invalid inputs:** Try unusual parameter combinations

---

## 🎯 **QUICK START:**

### **Do This Right Now (5 mins each):**

1. Open `pages/data_overview.py`
2. Change import:
   ```python
   from utils import format_currency, validate_dataframe
   ```
3. Add after `def render(df):`:
   ```python
   if not validate_dataframe(df, min_rows=1):
       return
   ```
4. Find any `if len(filtered_df) == 0:` and add warning
5. Test by selecting impossible filters
6. Repeat for next dashboard

---

## 🚀 **COMMIT MESSAGE TEMPLATE:**

```bash
git add pages/data_overview.py
git commit -m "Add error handling to Data Overview dashboard

- Validate dataframe at start
- Handle empty filter results gracefully
- Add user-friendly error messages"
```

---

## 💡 **PRO TIPS:**

1. **Always validate first** - Check data before processing
2. **Fail fast** - Stop execution early if problems detected
3. **Be helpful** - Tell users what to do, not just what went wrong
4. **Show details optionally** - Technical errors in expandable section
5. **Test edge cases** - Empty, single value, all identical

---

## ✨ **RESULT:**

After completing this for all dashboards:
- ✅ No more Python tracebacks visible to users
- ✅ Helpful error messages guide users
- ✅ App never crashes
- ✅ Professional user experience
- ✅ Ready for production deployment

---

**START WITH:** `data_overview.py` (easiest, 10 mins)  
**THEN:** `risk_optimization.py` (medium, 10 mins)  
**THEN:** `eda_explorer.py` (medium, 15 mins)  
**FINALLY:** `ml_models.py` (most complex, 20 mins)

**Total time investment: ~1 hour for bulletproof error handling! 🎯**
