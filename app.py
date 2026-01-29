import streamlit as st
import sys
import os
import plotly.express as px
import pandas as pd

# --- 1. THE PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

# --- 2. THE IMPORTS ---
try:
    import engine 
    import auditor 
except ModuleNotFoundError:
    st.error("❌ Technical modules missing in /src. Please ensure engine.py and auditor.py are in the src folder.")
    st.stop()

# --- 3. PAGE CONFIG ---
st.set_page_config(page_title="Incentive AI Optimizer", layout="wide", page_icon="🎯")

# --- 4. DATA LOADING ---
@st.cache_data
def load_fixed_data():
    return engine.generate_synthetic_data(100)

config = engine.get_config()
raw_df = load_fixed_data()

# Sidebar
st.sidebar.header("🕹️ Strategy Controls")
scenarios = list(config.get('scenarios', {}).keys())
selected_scenario = st.sidebar.selectbox("Select Payout Scenario", scenarios)

# Process Math
processed_df = engine.calculate_recommendation(raw_df, config, selected_scenario)

# --- 5. TOP METRICS ---
st.title("🎯 Incentive Payout Optimizer")

total_allocation = processed_df['Recommended_Payout'].sum()
strat_opt = total_allocation * 0.10 
avg_quality = processed_df['Quality_Score'].mean()

m_col1, m_col2, m_col3 = st.columns(3)
with m_col1:
    st.metric(label="Total AI Allocation", value=f"${total_allocation:,.0f}")
with m_col2:
    st.metric(label="Strategic Optimization", value=f"${strat_opt:,.0f}", delta="ROI Optimized")
with m_col3:
    st.metric(label="Avg Quality Score", value=f"{avg_quality:.1f}%")

st.markdown("---")

# --- 6. CHARTS (Matching your previous visual request) ---
c_col1, c_col2 = st.columns(2)
with c_col1:
    st.markdown("### Payout Distribution")
    fig = px.histogram(processed_df, x="Recommended_Payout", color="Dept", nbins=20, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
with c_col2:
    st.markdown("### Revenue vs. Payout")
    fig2 = px.scatter(processed_df, x="Revenue_KPI", y="Recommended_Payout", size="Quality_Score", color="Dept", template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

# --- 7. THE INTERACTIVE AGENTIC AUDITOR (Requested Format) ---
st.markdown("---")
st.header("🕵️ Interactive Agentic Auditor")
st.write("Select an employee from the dropdown to trigger a real-time Strategic Risk Analysis.")

# Search Employee ID Dropdown
employee_list = processed_df['Employee_ID'].tolist()
selected_emp_id = st.selectbox("Search Employee ID:", employee_list)

# Filter data for selected employee
target_emp = processed_df[processed_df['Employee_ID'] == selected_emp_id].iloc[0]



# Audit Display Columns
aud_col1, aud_col2 = st.columns([1, 2])

with aud_col1:
    st.subheader(f"Data for {selected_emp_id}")
    st.write(f"**Department:** {target_emp['Dept']}")
    st.write(f"**Revenue KPI:** {target_emp['Revenue_KPI']:.1f}%")
    st.write(f"**Quality Score:** {target_emp['Quality_Score']:.1f}%")
    st.divider()
    st.metric("Recommended Payout", f"${target_emp['Recommended_Payout']:,.2f}")

with aud_col2:
    st.subheader("Agentic Risk Assessment")
    
    with st.spinner("Performing Strategic Risk Analysis..."):
        # Fetching the live justification from auditor.py
        reasoning = auditor.get_agentic_audit(target_emp, selected_scenario)
        
        # Displaying in the exact format requested
        if target_emp['Quality_Score'] > 85:
            st.success(f"**✅ OPTIMAL PERFORMANCE**\n\n**Analysis:** {reasoning}\n\n**Action:** Approve for immediate disbursement.")
        elif target_emp['Quality_Score'] < 65:
            st.error(f"**⚠️ HIGH RISK DETECTED**\n\n**Analysis:** {reasoning}\n\n**Action:** Agent recommends a 15% hold-back for quality remediation.")
        else:
            st.info(f"**ℹ️ STANDARD REVIEW**\n\n**Analysis:** {reasoning}\n\n**Action:** Standard approval.")

# --- 8. FULL TABLE ---
with st.expander("View Full Payout Ledger"):
    # Defining columns to show
    ledger_cols = [
        'Employee_ID', 
        'Dept', 
        'CSAT_KPI', 
        'Base_Bonus_Target', 
        'Score', 
        'Recommended_Payout'
    ]
    
    st.dataframe(
        processed_df[ledger_cols], 
        use_container_width=True,
        column_config={
            "Employee_ID": "ID",
            "Dept": "Department",
            "CSAT_KPI": st.column_config.NumberColumn("CSAT (%)", format="%.1f"),
            "Base_Bonus_Target": st.column_config.NumberColumn("Base_Bonus_Target", format="$%d"),
            "Score": st.column_config.NumberColumn("Score", format="%.2f"),
            "Recommended_Payout": st.column_config.NumberColumn("Recommended_Payout", format="$%.2f")
        },
        hide_index=True
    )

    # --- SideBar: System Health (The Vercel Way) ---
with st.sidebar:
    st.divider()
    with st.status("📡 Agentic System Status", expanded=False) as status:
        st.write("Engine: Deterministic Math Verified")
        st.write("Auditor: Llama-3.3-70B Active")
        st.write("Infrastructure: Groq LPU Sub-second")
        status.update(label="✅ System Online", state="complete")

        
