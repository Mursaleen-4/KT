import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import folium
from streamlit_folium import folium_static
import requests
from datetime import datetime
import joblib
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="SpaceX Launch Analytics",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .title-text {
        font-size: 2.5rem;
        color: #00FF00;
        text-align: center;
        margin-bottom: 2rem;
    }
    .subtitle-text {
        font-size: 1.5rem;
        color: #FFFFFF;
        margin-bottom: 1rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1.2rem;
        color: #FFFFFF;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    launches_df = pd.read_csv('data/processed/launches.csv')
    launchpads_df = pd.read_csv('data/processed/launchpads.csv')
    rockets_df = pd.read_csv('data/processed/rockets.csv')
    return launches_df, launchpads_df, rockets_df

# Load model
@st.cache_resource
def load_model():
    return joblib.load('models/launch_predictor.joblib')

# Navigation
selected = option_menu(
    menu_title=None,
    options=["Dashboard", "Predictions", "Analysis", "About"],
    icons=["speedometer2", "rocket", "graph-up", "info-circle"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#1E1E1E"},
        "icon": {"color": "#00FF00", "font-size": "25px"},
        "nav-link": {
            "font-size": "16px",
            "text-align": "center",
            "margin": "0px",
            "--hover-color": "#2E2E2E",
        },
        "nav-link-selected": {"background-color": "#2E2E2E"},
    }
)

# Dashboard Page
if selected == "Dashboard":
    st.markdown('<h1 class="title-text">SpaceX Launch Analytics</h1>', unsafe_allow_html=True)
    
    # Load data
    launches_df, launchpads_df, rockets_df = load_data()
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Launches",
            value=len(launches_df),
            delta=f"{len(launches_df[launches_df['success'] == True])} Successful"
        )
    
    with col2:
        success_rate = (launches_df['success'].mean() * 100)
        st.metric(
            label="Success Rate",
            value=f"{success_rate:.1f}%",
            delta=f"{success_rate - 95:.1f}% vs Target"
        )
    
    with col3:
        st.metric(
            label="Active Launchpads",
            value=len(launchpads_df[launchpads_df['status'] == 'active']),
            delta="Operational"
        )
    
    with col4:
        st.metric(
            label="Rocket Types",
            value=len(rockets_df),
            delta="In Service"
        )
    
    # Main Content
    st.markdown('<h2 class="subtitle-text">Launch Performance Overview</h2>', unsafe_allow_html=True)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Launch Timeline", "Success Analysis", "Geographic Distribution"])
    
    with tab1:
        # Launch Timeline with Plotly
        timeline_data = launches_df.groupby('launch_year')['success'].agg(['count', 'mean']).reset_index()
        timeline_data['success_rate'] = timeline_data['mean'] * 100
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=timeline_data['launch_year'],
            y=timeline_data['count'],
            name='Number of Launches'
        ))
        fig.add_trace(go.Scatter(
            x=timeline_data['launch_year'],
            y=timeline_data['success_rate'],
            name='Success Rate',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Launch Success Rate Over Time',
            xaxis_title='Year',
            yaxis_title='Number of Launches',
            yaxis2=dict(
                title='Success Rate (%)',
                overlaying='y',
                side='right',
                range=[0, 100]
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Success Analysis with Plotly
        col1, col2 = st.columns(2)
        
        with col1:
            # Payload Mass vs Success
            fig = px.scatter(
                launches_df,
                x='payload_mass_kg',
                y='success',
                color='success',
                title='Payload Mass vs Launch Success',
                labels={'payload_mass_kg': 'Payload Mass (kg)', 'success': 'Success'},
                color_discrete_sequence=['#FF0000', '#00FF00']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Success Rate by Rocket
            rocket_success = launches_df.groupby('rocket_name')['success'].mean().reset_index()
            fig = px.bar(
                rocket_success,
                x='rocket_name',
                y='success',
                title='Success Rate by Rocket Type',
                labels={'rocket_name': 'Rocket', 'success': 'Success Rate'},
                color='success',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Geographic Distribution with Folium
        m = folium.Map(location=[28.5728, -80.6490], zoom_start=4)
        
        for _, row in launchpads_df.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=10,
                popup=f"{row['name']}<br>Success Rate: {row['success_rate']:.1f}%",
                color='green' if row['success_rate'] > 90 else 'red',
                fill=True
            ).add_to(m)
        
        folium_static(m, width=800, height=400)

# Predictions Page
elif selected == "Predictions":
    st.markdown('<h1 class="title-text">Launch Success Predictor</h1>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    # Create prediction form
    st.markdown('<h2 class="subtitle-text">Enter Launch Parameters</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        rocket_name = st.selectbox(
            "Rocket Type",
            options=["Falcon 9", "Falcon Heavy"]
        )
        
        payload_mass = st.number_input(
            "Payload Mass (kg)",
            min_value=0,
            max_value=50000,
            value=5000
        )
        
        launch_site = st.selectbox(
            "Launch Site",
            options=["Kennedy Space Center", "Vandenberg Air Force Base", "Cape Canaveral"]
        )
    
    with col2:
        weather_condition = st.selectbox(
            "Weather Condition",
            options=["Clear", "Cloudy", "Rainy", "Windy"]
        )
        
        wind_speed = st.number_input(
            "Wind Speed (km/h)",
            min_value=0,
            max_value=100,
            value=20
        )
        
        temperature = st.number_input(
            "Temperature (°C)",
            min_value=-20,
            max_value=50,
            value=25
        )
    
    if st.button("Predict Launch Success", type="primary"):
        # Prepare input data
        input_data = pd.DataFrame({
            'rocket_name': [rocket_name],
            'payload_mass_kg': [payload_mass],
            'launch_site': [launch_site],
            'weather_condition': [weather_condition],
            'wind_speed': [wind_speed],
            'temperature': [temperature]
        })
        
        # Make prediction
        prediction = model.predict_proba(input_data)[0]
        
        # Display results
        st.markdown("### Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Success Probability",
                value=f"{prediction[1]*100:.1f}%",
                delta=f"{prediction[1]*100 - 50:.1f}%"
            )
        
        with col2:
            st.metric(
                label="Risk Level",
                value="High" if prediction[1] < 0.7 else "Low",
                delta="Critical" if prediction[1] < 0.5 else "Acceptable"
            )
        
        # Visualize prediction
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction[1]*100,
            title={'text': "Launch Success Probability"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "red"},
                    {'range': [50, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "green"}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

# Analysis Page
elif selected == "Analysis":
    st.markdown('<h1 class="title-text">Launch Analysis</h1>', unsafe_allow_html=True)
    
    # Load data
    launches_df, launchpads_df, rockets_df = load_data()
    
    # Analysis options
    analysis_type = st.selectbox(
        "Select Analysis Type",
        options=["Launch Performance", "Rocket Efficiency", "Weather Impact"]
    )
    
    if analysis_type == "Launch Performance":
        st.markdown('<h2 class="subtitle-text">Launch Performance Analysis</h2>', unsafe_allow_html=True)
        
        # Performance metrics
        col1, col2 = st.columns(2)
        
        with col1:
            # Success rate by year
            yearly_success = launches_df.groupby('launch_year')['success'].mean().reset_index()
            fig = px.line(
                yearly_success,
                x='launch_year',
                y='success',
                title='Success Rate Trend',
                labels={'launch_year': 'Year', 'success': 'Success Rate'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Payload mass distribution
            fig = px.histogram(
                launches_df,
                x='payload_mass_kg',
                color='success',
                title='Payload Mass Distribution',
                labels={'payload_mass_kg': 'Payload Mass (kg)'},
                color_discrete_sequence=['#FF0000', '#00FF00']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Rocket Efficiency":
        st.markdown('<h2 class="subtitle-text">Rocket Efficiency Analysis</h2>', unsafe_allow_html=True)
        
        # Rocket metrics
        col1, col2 = st.columns(2)
        
        with col1:
            # Success rate by rocket
            rocket_success = launches_df.groupby('rocket_name')['success'].mean().reset_index()
            fig = px.bar(
                rocket_success,
                x='rocket_name',
                y='success',
                title='Success Rate by Rocket',
                labels={'rocket_name': 'Rocket', 'success': 'Success Rate'},
                color='success',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Payload capacity utilization
            fig = px.scatter(
                launches_df,
                x='rocket_name',
                y='payload_mass_kg',
                color='success',
                title='Payload Capacity Utilization',
                labels={'rocket_name': 'Rocket', 'payload_mass_kg': 'Payload Mass (kg)'},
                color_discrete_sequence=['#FF0000', '#00FF00']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:  # Weather Impact
        st.markdown('<h2 class="subtitle-text">Weather Impact Analysis</h2>', unsafe_allow_html=True)
        
        # Weather metrics
        col1, col2 = st.columns(2)
        
        with col1:
            # Success rate by weather condition
            weather_success = launches_df.groupby('weather_condition')['success'].mean().reset_index()
            fig = px.bar(
                weather_success,
                x='weather_condition',
                y='success',
                title='Success Rate by Weather',
                labels={'weather_condition': 'Weather', 'success': 'Success Rate'},
                color='success',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Wind speed impact
            fig = px.scatter(
                launches_df,
                x='wind_speed',
                y='success',
                color='success',
                title='Wind Speed Impact',
                labels={'wind_speed': 'Wind Speed (km/h)', 'success': 'Success'},
                color_discrete_sequence=['#FF0000', '#00FF00']
            )
            st.plotly_chart(fig, use_container_width=True)

# About Page
else:
    st.markdown('<h1 class="title-text">About the Project</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Project Overview
    This SpaceX Launch Analysis Platform provides comprehensive insights into launch operations, 
    success predictions, and performance analytics.
    
    ### Features
    - Real-time launch data analysis
    - Interactive visualizations
    - Machine learning predictions
    - Weather impact analysis
    - Performance metrics
    
    ### Technology Stack
    - Python
    - Streamlit
    - Plotly
    - Scikit-learn
    - Pandas
    
    ### Data Sources
    - SpaceX API
    - Weather API
    - Historical launch data
    
    ### Contact
    For more information, please contact the development team.
    """) 