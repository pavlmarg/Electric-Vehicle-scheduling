import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Page Configuration
st.set_page_config(page_title="EV Optimization Dashboard", layout="wide")
st.title("EV Charging Optimal Benchmark Dashboard")

# 2. Load the Data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("history_optimal.txt")
        # Convert Station_ID to string so it plots as categories, not a continuous line
        df['Station_ID'] = df['Station_ID'].astype(str)
        return df
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.error("Could not find 'history_optimal.txt'. Please run main_optimal.py first!")
else:
    # --- 3. TOP LEVEL KPIs ---
    total_evs = len(df)
    survivors = len(df[df['Status'] == 'Survived'])
    survival_rate = (survivors / total_evs) * 100 if total_evs > 0 else 0
    total_energy = df['Energy_kWh'].sum()
    total_revenue = df['Cost_Euro'].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Traffic (EVs)", total_evs)
    col2.metric("Survival Rate", f"{survival_rate:.1f}%")
    col3.metric("Total Energy Delivered", f"{total_energy:.1f} kWh")
    col4.metric("Total Grid Revenue", f"€ {total_revenue:.2f}")
    
    st.markdown("---")
    
    # --- 4. ROW 1: PIE CHARTS (Status & Charger Types) ---
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("System Reliability (EV Status)")
        # Custom colors for clarity
        color_map_status = {
            'Survived': '#28a745',          # Green
            'Crashed (No Battery)': '#dc3545', # Red
            'Failed (Undercharged)': '#ffc107' # Yellow
        }
        fig_status = px.pie(df, names='Status', hole=0.4, color='Status', color_discrete_map=color_map_status)
        st.plotly_chart(fig_status, use_container_width=True)
        
    with col_b:
        st.subheader("Hardware Utilization (Fast vs Slow)")
        color_map_charger = {
            'Fast': '#17a2b8',  
            'Slow': '#6c757d', 
            'N/A': '#343a40'    
        }
        fig_charger = px.pie(df, names='Charger_Type', hole=0.4, color='Charger_Type', color_discrete_map=color_map_charger)
        st.plotly_chart(fig_charger, use_container_width=True)

    # --- 5. ROW 2: STATION ANALYTICS (Bar Charts) ---
    st.subheader("Energy Demand per Station")
    # Group the data by station to see which one works the hardest
    station_energy = df.groupby('Station_ID')['Energy_kWh'].sum().reset_index()
    
    fig_station = px.bar(
        station_energy, 
        x='Station_ID', 
        y='Energy_kWh', 
        color='Energy_kWh', 
        color_continuous_scale='Viridis',
        labels={'Station_ID': 'Station ID', 'Energy_kWh': 'Total Energy Delivered (kWh)'}
    )
    st.plotly_chart(fig_station, use_container_width=True)

    # --- 6. RAW DATA TABLE ---
    with st.expander("🔍 View Raw History Data"):
        st.dataframe(df, use_container_width=True)