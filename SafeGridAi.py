import os
import sys

# Add project directory and its subfolders to sys.path to resolve module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
subfolder = os.path.join(current_dir, "women_safety_route_app")
if os.path.isdir(subfolder) and subfolder not in sys.path:
    sys.path.append(subfolder)

import asyncio
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import streamlit as st
import matplotlib.cm as cm
from dotenv import load_dotenv

load_dotenv(override=True)

# Streamlit Page Config
st.set_page_config(layout="wide", page_title="SafeGrid AI - Women Safety Route Planner", page_icon="🛡️")

# Custom CSS Styles and Fonts Injection
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

/* Global Styles */
.stApp {
    background-color: #F8FAFC;
    font-family: 'Plus Jakarta Sans', 'Outfit', -apple-system, sans-serif;
}

div.block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1300px !important;
}

/* Hide Streamlit default components */
footer {visibility: hidden !important;}
#MainMenu {visibility: hidden !important;}
div[data-testid="stHeader"] {background: transparent !important;}

/* Hero Header and Banner */
.hero-container {
    background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
    border-radius: 20px;
    padding: 2.5rem;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px -5px rgba(37, 99, 235, 0.25);
    position: relative;
    overflow: hidden;
}

.hero-container::after {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 60%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 60%);
    transform: rotate(-15deg);
    pointer-events: none;
}

.hero-title {
    font-family: 'Outfit', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.03em;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.hero-subtitle {
    font-size: 1.25rem;
    font-weight: 500;
    opacity: 0.95;
    margin-top: 0.5rem;
    margin-bottom: 1rem;
    font-family: 'Outfit', sans-serif;
    letter-spacing: -0.01em;
}

.hero-description {
    font-size: 0.95rem;
    opacity: 0.85;
    max-width: 750px;
    line-height: 1.6;
}

/* Form Container / Custom Card */
div[data-testid="stForm"] {
    background-color: white !important;
    border-radius: 16px !important;
    border: 1px solid #E2E8F0 !important;
    padding: 2rem !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.02), 0 2px 4px rgba(0, 0, 0, 0.02) !important;
    margin-bottom: 2rem !important;
}

/* Custom Card Layout */
.custom-card {
    background: white;
    border-radius: 16px;
    padding: 2rem;
    border: 1px solid #E2E8F0;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.02);
    margin-bottom: 2rem;
}

/* Tabs Styling */
div[data-testid="stTabBar"] {
    background-color: white;
    border-radius: 12px;
    padding: 0.35rem;
    border: 1px solid #E2E8F0;
    margin-bottom: 2rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.01);
}

button[data-testid="stTab"] {
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    font-size: 1rem !important;
    color: #64748B !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s ease !important;
    border: none !important;
}

button[data-testid="stTab"][aria-selected="true"] {
    background-color: #2563EB !important;
    color: white !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.18) !important;
}

/* Text Inputs Styling */
div[data-testid="stTextInput"] label {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    color: #475569 !important;
    font-size: 0.95rem !important;
    margin-bottom: 0.5rem !important;
}

div[data-testid="stTextInput"] input {
    background-color: #F8FAFC !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    padding: 0.8rem 1rem !important;
    font-size: 1rem !important;
    color: #1E293B !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stTextInput"] input:focus {
    border-color: #2563EB !important;
    background-color: white !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
}

/* Buttons Styling */
div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
    color: white !important;
    border-radius: 10px !important;
    padding: 0.8rem 2rem !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    border: none !important;
    width: 100% !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
}

div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
    background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(37, 99, 235, 0.3) !important;
}

div.stButton > button:active, div[data-testid="stFormSubmitButton"] > button:active {
    transform: translateY(0px) !important;
}

/* SOS Submit Button Specific Styling */
div.sos-btn-container button {
    background: linear-gradient(135deg, #DC2626 0%, #B91C1C 100%) !important;
    box-shadow: 0 4px 12px rgba(220, 38, 38, 0.25) !important;
    animation: sosPulse 2.5s infinite;
}

div.sos-btn-container button:hover {
    background: linear-gradient(135deg, #B91C1C 0%, #991B1B 100%) !important;
    box-shadow: 0 8px 22px rgba(220, 38, 38, 0.35) !important;
}

@keyframes sosPulse {
    0% {
        box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.4);
    }
    70% {
        box-shadow: 0 0 0 10px rgba(220, 38, 38, 0);
    }
    100% {
        box-shadow: 0 0 0 0 rgba(220, 38, 38, 0);
    }
}

/* Metric Cards Layout */
.metric-container {
    display: flex;
    flex-wrap: wrap;
    gap: 1.5rem;
    margin-bottom: 2rem;
    width: 100%;
}

.metric-card {
    flex: 1;
    min-width: 250px;
    background: white;
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid #E2E8F0;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.02);
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.04);
}

.metric-card::before {
    content: '';
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 4px;
    background: #E2E8F0;
}

.metric-card.distance::before {
    background: #2563EB;
}

.metric-card.eta::before {
    background: #4F46E5;
}

.metric-card.risk-low::before {
    background: #16A34A;
}
.metric-card.risk-medium::before {
    background: #F59E0B;
}
.metric-card.risk-high::before {
    background: #DC2626;
}

.metric-label {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748B;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #0F172A;
    font-family: 'Outfit', sans-serif;
    letter-spacing: -0.02em;
}

.metric-subtitle {
    font-size: 0.8rem;
    color: #94A3B8;
}

/* Map IFrame Styling Override */
iframe[title="streamlit_folium.st_folium"] {
    border-radius: 12px !important;
    border: none !important;
    width: 100% !important;
}

/* Insight Cards Styling */
.insight-card {
    background: white;
    border-radius: 16px;
    padding: 1.75rem;
    border: 1px solid #E2E8F0;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.02);
    height: 100%;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.insight-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.04);
}

.insight-header {
    font-family: 'Outfit', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #1E293B;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    border-bottom: 1px solid #F1F5F9;
    padding-bottom: 0.75rem;
}

.insight-item {
    display: flex;
    align-items: flex-start;
    gap: 0.85rem;
    padding: 0.85rem 0;
    border-bottom: 1px solid #F8FAFC;
}

.insight-item:last-child {
    border-bottom: none;
    padding-bottom: 0;
}

.insight-icon {
    font-size: 1.3rem;
    margin-top: 0.1rem;
}

.insight-content {
    flex: 1;
}

.insight-title {
    font-weight: 700;
    color: #334155;
    font-size: 0.95rem;
    font-family: 'Outfit', sans-serif;
}

.insight-desc {
    color: #64748B;
    font-size: 0.9rem;
    line-height: 1.4;
    margin-top: 0.25rem;
}

.police-badge {
    background: #EFF6FF;
    color: #1E40AF;
    padding: 0.2rem 0.55rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 700;
    display: inline-block;
    margin-top: 0.35rem;
    border: 1px solid #DBEAFE;
}

/* SOS Output Warning Card */
.sos-output-container {
    background: #FEF2F2 !important;
    border: 1px solid #FCA5A5 !important;
    border-radius: 16px !important;
    padding: 2rem !important;
    margin-bottom: 2rem !important;
    box-shadow: 0 4px 15px rgba(220, 38, 38, 0.05) !important;
}

.sos-output-container div, .sos-output-container p, .sos-output-container li, .sos-output-container span {
    color: #7F1D1D !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.95rem !important;
}

.sos-output-container h1, .sos-output-container h2, .sos-output-container h3, .sos-output-container h4 {
    color: #991B1B !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    margin-top: 0 !important;
    margin-bottom: 1rem !important;
}

.sos-output-container ul {
    margin-bottom: 0 !important;
    padding-left: 1.25rem !important;
}

.sos-output-container li {
    margin-bottom: 0.5rem !important;
}

.sos-output-container li:last-child {
    margin-bottom: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# Custom Hero Section
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🛡️ SafeGrid AI</div>
    <div class="hero-subtitle">Smart Women Safety Navigation using AI Agents</div>
    <div class="hero-description">
        SafeGrid AI leverages Google ADK-powered multi-agent systems, historical crime datasets, and safety analysis to plan the most secure routes. Stay protected with real-time risk scores, visual route mapping, and instant emergency services locator.
    </div>
</div>
""", unsafe_allow_html=True)

# Import ADK elements
try:
    from google.adk.runners import InMemoryRunner
    from adk_agents import safety_workflow, emergency_workflow, routing_service
except Exception as e:
    st.error(f"Failed to import ADK elements or routing service: {e}")
    st.stop()
# Local Fallback Safety Solver
def local_safety_solver(start_location, destination):
    from geopy.geocoders import Nominatim
    from shapely.geometry import LineString
    import json
    
    geolocator = Nominatim(user_agent="safegrid")
    
    # Geocode start
    query_start = start_location if "delhi" in start_location.lower() else f"{start_location}, Delhi"
    geo_start = geolocator.geocode(query_start)
    if not geo_start:
        raise ValueError(f"Start location '{start_location}' not found")
        
    # Geocode destination
    query_end = destination if "delhi" in destination.lower() else f"{destination}, Delhi"
    geo_end = geolocator.geocode(query_end)
    if not geo_end:
        raise ValueError(f"Destination location '{destination}' not found")
        
    # Call routing service
    res = routing_service.find_safest_route(
        geo_start.latitude, geo_start.longitude,
        geo_end.latitude, geo_end.longitude
    )
    if not res.get("route_found", False):
        raise ValueError("Route not found between the specified locations.")
        
    cells = res["route_cells"]
    scores = [c["safety_score"] for c in cells]
    avg_safety = sum(scores) / len(scores) if scores else 1.0
    risk_score = int(round((1 - avg_safety) * 100))
    
    if risk_score < 30:
        risk_level = "LOW"
    elif risk_score < 60:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"
        
    # Hotspots
    hotspots = []
    risky_cells = [c for c in cells if c.get("safety_score", 1.0) < 0.5]
    df_m = routing_service.location_df
    for cell in risky_cells:
        lat = cell.get("lat")
        lon = cell.get("lon")
        safety_score = cell.get("safety_score", 0.0)
        if lat is not None and lon is not None:
            dist_sq = (df_m['Latitude'] - float(lat))**2 + (df_m['Longitude'] - float(lon))**2
            idx = dist_sq.idxmin()
            closest_station = df_m.loc[idx]['Station Names']
            risk_desc = "High" if safety_score < 0.3 else "Moderate"
            hotspots.append(f"{closest_station} Area — {risk_desc} Risk")
    unique_hotspots = list(dict.fromkeys(hotspots))[:5]
    
    # Police on Route
    police_on_route = []
    route_coords = [(float(c["lon"]), float(c["lat"])) for c in cells if c.get("lat") is not None and c.get("lon") is not None]
    if len(route_coords) >= 2:
        route_line = LineString(route_coords)
        gdf_route = gpd.GeoDataFrame(geometry=[route_line], crs="EPSG:4326")
        gdf_route_proj = gdf_route.to_crs(epsg=32643)
        route_geom_proj = gdf_route_proj.geometry.iloc[0]
        
        df_p = routing_service.police_df
        gdf_police = gpd.GeoDataFrame(
            df_p,
            geometry=gpd.points_from_xy(df_p["Longitude"], df_p["Latitude"]),
            crs="EPSG:4326"
        )
        gdf_police_proj = gdf_police.to_crs(epsg=32643)
        distances_m = gdf_police_proj.geometry.distance(route_geom_proj)
        
        df_p_copy = df_p.copy()
        df_p_copy["distance_m"] = distances_m
        nearby_df = df_p_copy[df_p_copy["distance_m"] <= 1000.0].copy()
        nearby_df = nearby_df.sort_values(by="distance_m").drop_duplicates(subset=["Police Station"], keep="first")
        for _, row in nearby_df.iterrows():
            police_on_route.append({
                "name": str(row["Police Station"]),
                "lat": float(row["Latitude"]),
                "lon": float(row["Longitude"]),
                "distance_m": float(row["distance_m"])
            })
            
    # Recommendations
    advice = []
    if risk_level == "HIGH":
        advice.extend([
            "Avoid travel after 10 PM along this route.",
            "Prefer using Delhi Metro lines instead of walking or two-wheeler transport.",
            "Share your live location with trusted family/friends.",
            "Keep active GPS navigation on and stick to the main illuminated roads."
        ])
    elif risk_level == "MEDIUM":
        advice.extend([
            "Stick to main streets; avoid unlit alleys.",
            "Keep emergency contacts on speed dial.",
            "Share travel details before starting your trip."
        ])
    else:
        advice.extend([
            "Route is generally safe.",
            "Stick to normal safety precautions."
        ])
        
    police_list_str = ""
    if police_on_route:
        for p in police_on_route:
            police_list_str += f"- **{p['name']}** (~{int(p['distance_m'])}m away)\n"
    else:
        police_list_str = "No police stations found within 1km of the route.\n"
        
    advice_str = "\n".join([f"- {a}" for a in advice])
    
    recommendation = f"""### Safety Assessment (Local Offline Mode)
- **Route Summary**: Route found from {start_location} to {destination}. Distance: {res['distance_km']} km. ETA: {res['eta_min']} mins.
- **Risk Level**: {risk_level}
- **Risk Score**: {risk_score}%

### Nearby Police Stations Along Route
{police_list_str}
### Safety Advice & Recommendations
{advice_str}
"""
    return {
        "route_found": True,
        "distance_km": res["distance_km"],
        "eta_min": res["eta_min"],
        "route_cells": cells,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "hotspots": unique_hotspots,
        "police_on_route": police_on_route,
        "recommendation": recommendation
    }

# Local Fallback Emergency Solver
def local_emergency_solver(user_location):
    from geopy.geocoders import Nominatim
    geolocator = Nominatim(user_agent="safegrid")
    query = user_location if "delhi" in user_location.lower() else f"{user_location}, Delhi"
    geo = geolocator.geocode(query)
    if not geo:
        raise ValueError(f"Could not locate user location '{user_location}'")
        
    user_lat, user_lon = geo.latitude, geo.longitude
    
    # 1. Find nearest police station
    df_p = routing_service.police_df
    dist_sq_p = (df_p['Latitude'] - user_lat)**2 + (df_p['Longitude'] - user_lon)**2
    idx_p = dist_sq_p.idxmin()
    closest_p = df_p.loc[idx_p]
    p_name = closest_p['Police Station']
    p_lat = float(closest_p['Latitude'])
    p_lon = float(closest_p['Longitude'])
    
    # 2. Find nearest metro station
    df_m = routing_service.location_df
    dist_sq_m = (df_m['Latitude'] - user_lat)**2 + (df_m['Longitude'] - user_lon)**2
    idx_m = dist_sq_m.idxmin()
    closest_m = df_m.loc[idx_m]
    m_name = closest_m['Station Names']
    m_line = closest_m['Metro Line']
    m_lat = float(closest_m['Latitude'])
    m_lon = float(closest_m['Longitude'])
    
    # 3. Find nearest hospital
    hospitals = [
        {"name": "Lok Nayak Hospital", "lat": 28.6366, "lon": 77.2407},
        {"name": "Dr. Ram Manohar Lohia Hospital", "lat": 28.6253, "lon": 77.2007},
        {"name": "AIIMS New Delhi", "lat": 28.5672, "lon": 77.2100},
        {"name": "Max Super Speciality Hospital, Shalimar Bagh", "lat": 28.7180, "lon": 77.1585},
        {"name": "Sir Ganga Ram Hospital", "lat": 28.6385, "lon": 77.1895}
    ]
    closest_h = min(hospitals, key=lambda h: (h['lat'] - user_lat)**2 + (h['lon'] - user_lon)**2)
    h_name = closest_h['name']
    h_lat = closest_h['lat']
    h_lon = closest_h['lon']
    
    emergency_info = f"""### 🚨 SOS Activated: Emergency Services Near {user_location}

Based on local spatial database queries:

#### 🚓 Nearest Police Station
- **Name**: {p_name} Police Station
- **Coordinates**: ({p_lat:.4f}, {p_lon:.4f})
- **Action**: Head here or call **112** (Delhi Police Emergency Helpline).

#### 🚇 Nearest Transit / Metro Station
- **Name**: {m_name} Metro Station
- **Line**: {m_line} Line
- **Coordinates**: ({m_lat:.4f}, {m_lon:.4f})
- **Action**: Metro stations are heavily monitored, illuminated, and staffed with security guards.

#### 🏥 Nearest Emergency Hospital
- **Name**: {h_name}
- **Coordinates**: ({h_lat:.4f}, {h_lon:.4f})
- **Action**: Medical help is available here.

---

### 🛡️ Quick Safety Advice for SOS Situations:
1. **Move to a Public Place**: Head immediately towards the nearest Metro Station or a highly illuminated public street.
2. **Share Live Location**: Use the "Share Location" feature on your phone to send coordinates to trusted contacts.
3. **Stay in Contact**: Call the emergency helpline (**112**) or keep a call active with someone you trust while moving.
"""
    return {
        "emergency_info": emergency_info
    }

# API Key Validation
if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    st.warning("⚠️ **API Key Missing**: Please set `GEMINI_API_KEY` or `GOOGLE_API_KEY` in a `.env` file or environment variables. System will use local offline routing engine.")

# Create tabs
tab1, tab2 = st.tabs(["🛣️ Route Safety Planner", "🚨 Emergency SOS Locator"])

with tab1:
    with st.form("route_form"):
        st.markdown('<div style="font-family:\'Outfit\', sans-serif; font-size:1.4rem; font-weight:700; color:#1E293B; margin-bottom:1.5rem;">🗺️ Route Safe-Planner</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            start_location = st.text_input("Start Location", value="IGDTUW", key="route_start")
        with col2:
            destination = st.text_input("Destination", value="Connaught Place", key="route_dest")
            
        # Form Submit Button
        btn_route_submitted = st.form_submit_button("Generate Safe Route")
        
    if btn_route_submitted:
        if not start_location or not destination:
            st.error("Please enter both Start Location and Destination.")
        else:
            with st.spinner("AI agents are negotiating and planning your route..."):
                try:
                    res = None
                    try:
                        runner = InMemoryRunner(node=safety_workflow)
                        # Run workflow asynchronously
                        events = asyncio.run(runner.run_debug(f"Source: {start_location}, Destination: {destination}"))
                        
                        # Extract final workflow output
                        output_event = next((e for e in reversed(events) if e.output is not None), None)
                        res = output_event.output if output_event else None
                    except Exception as agent_err:
                        st.info("ℹ️ **Gemini API Limit Reached**: Computing route & safety parameters locally.")
                        res = local_safety_solver(start_location, destination)
                        
                    if not res or not isinstance(res, dict):
                        st.error("No valid response from agents or local fallback solver.")
                    elif not res.get("route_found", False):
                        st.error("Route not found between the specified locations. Please try again.")
                    else:
                        if "local" in res.get("recommendation", "").lower():
                            st.success("Safe route computed successfully via local fallback engine!")
                        else:
                            st.success("Safe route computed successfully!")
                        
                        # Display Route Metrics (Custom Beautiful Cards)
                        risk_lvl = res['risk_level'].upper()
                        risk_cls = "risk-low"
                        if "MEDIUM" in risk_lvl:
                            risk_cls = "risk-medium"
                        elif "HIGH" in risk_lvl or "DANGER" in risk_lvl:
                            risk_cls = "risk-high"
                            
                        metrics_html = f"""
                        <div class="metric-container">
                            <div class="metric-card distance">
                                <div class="metric-label">
                                    <span class="metric-icon">🛣️</span> Total Distance
                                </div>
                                <div class="metric-value">{res['distance_km']} km</div>
                                <div class="metric-subtitle">Optimized safety routing distance</div>
                            </div>
                            <div class="metric-card eta">
                                <div class="metric-label">
                                    <span class="metric-icon">⏱️</span> Estimated Time (ETA)
                                </div>
                                <div class="metric-value">{res['eta_min']} mins</div>
                                <div class="metric-subtitle">Based on road networks & conditions</div>
                            </div>
                            <div class="metric-card {risk_cls}">
                                <div class="metric-label">
                                    <span class="metric-icon">🛡️</span> Safety Level / Risk
                                </div>
                                <div class="metric-value">{res['risk_level']} <span style="font-size: 1.1rem; font-weight:500; color:#64748B;">({res['risk_score']}% risk)</span></div>
                                <div class="metric-subtitle">Multi-agent evaluated route threat index</div>
                            </div>
                        </div>
                        """
                        st.markdown(metrics_html, unsafe_allow_html=True)
                        
                        # Map section: Main Hero card
                        st.markdown('<div class="custom-card" style="padding: 1.5rem 1.5rem 0.5rem 1.5rem;"><div class="insight-header">🗺️ SafeGrid Interactive Route Map</div>', unsafe_allow_html=True)
                        
                        # Render Folium Map
                        # Get start coordinate from first cell
                        cells = res["route_cells"]
                        start_lat, start_lon = cells[0]["lat"], cells[0]["lon"]
                        end_lat, end_lon = cells[-1]["lat"], cells[-1]["lon"]
                        
                        m_grid = folium.Map(location=[start_lat, start_lon], zoom_start=13)
                        
                        # Add boundary
                        folium.GeoJson(
                            routing_service.gdf_boundary.to_crs(epsg=4326),
                            style_function=lambda x: {"color": "black", "weight": 2, "fillOpacity": 0}
                        ).add_to(m_grid)
                        
                        # Add Grid colored by safety score
                        grid_gdf_4326 = routing_service.grid_gdf.to_crs(epsg=4326).copy()
                        grid_gdf_4326["color"] = grid_gdf_4326["safety_score"].apply(
                            lambda s: '#%02x%02x%02x' % tuple(int(255 * c) for c in cm.RdYlGn(s)[:3])
                        )
                        
                        folium.GeoJson(
                            grid_gdf_4326.__geo_interface__,
                            style_function=lambda feature: {
                                "fillColor": feature["properties"]["color"],
                                "color": "black",
                                "weight": 0.2,
                                "fillOpacity": 0.4
                            }
                        ).add_to(m_grid)
                        
                        # Add route PolyLine
                        route_coords = [(c["lat"], c["lon"]) for c in cells]
                        folium.PolyLine(route_coords, color="blue", weight=5, opacity=0.8, tooltip="Safest Route").add_to(m_grid)
                        
                        # Add markers
                        folium.Marker([start_lat, start_lon], popup=f"Start: {start_location}", icon=folium.Icon(color='green', icon='play')).add_to(m_grid)
                        folium.Marker([end_lat, end_lon], popup=f"End: {destination}", icon=folium.Icon(color='red', icon='stop')).add_to(m_grid)
                        
                        # Add police stations near the route
                        for ps in res.get("police_on_route", []):
                            folium.Marker(
                                [ps["lat"], ps["lon"]],
                                popup=f"Police Station: {ps['name']} ({ps['distance_m']:.0f}m from route)",
                                icon=folium.Icon(color='blue', icon='shield')
                            ).add_to(m_grid)
                        
                        st_folium(
                            m_grid,
                            width=None,
                            height=550,
                            returned_objects=[],
                            key="safety_map",
                            use_container_width=True
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Parse advice bullets
                        advice_bullets = []
                        rec_text = res.get("recommendation", "")
                        if "Safety Advice & Recommendations" in rec_text:
                            parts = rec_text.split("Safety Advice & Recommendations")
                            advice_part = parts[-1]
                            for line in advice_part.split("\n"):
                                line_s = line.strip()
                                if line_s.startswith("-") or line_s.startswith("*"):
                                    advice_bullets.append(line_s.lstrip("-* ").strip())
                        if not advice_bullets:
                            advice_bullets = [
                                "Stay on main roads and avoid deserted pathways.",
                                "Keep your phone active and emergency contacts on speed dial.",
                                "Share your real-time ETA and route tracking with family."
                            ]
                        
                        # Parse hotspots
                        hotspots = res.get('hotspots', [])
                        hotspots_text = "None detected on route" if not hotspots else "<br/>".join(hotspots)
                        
                        # Construct Assessment Html (flat strings with no leading indentation on each line)
                        assessment_html = (
                            f'<div class="insight-item">'
                            f'<div class="insight-content">'
                            f'<div class="insight-title">Safety Confidence</div>'
                            f'<div class="insight-desc">The path is evaluated to be <strong>{100 - res["risk_score"]}% Secure</strong>.</div>'
                            f'</div>'
                            f'</div>'
                            f'<div class="insight-item">'
                            f'<div class="insight-content">'
                            f'<div class="insight-title">Active Threat Hotspots</div>'
                            f'<div class="insight-desc">{hotspots_text}</div>'
                            f'</div>'
                            f'</div>'
                        )
                        
                        # Construct Police Html (flat strings with no leading indentation on each line)
                        police_html = ""
                        police_list = res.get("police_on_route", [])
                        if police_list:
                            for ps in police_list[:3]: # Show top 3
                                police_html += (
                                    f'<div class="insight-item">'
                                    f'<span class="insight-icon">🚓</span>'
                                    f'<div class="insight-content">'
                                    f'<div class="insight-title">{ps["name"]}</div>'
                                    f'<span class="police-badge">{ps["distance_m"]:.0f}m from route</span>'
                                    f'</div>'
                                    f'</div>'
                                )
                        else:
                            police_html = (
                                f'<div class="insight-item">'
                                f'<span class="insight-icon">⚠️</span>'
                                f'<div class="insight-content">'
                                f'<div class="insight-title">No Stations Detected</div>'
                                f'<div class="insight-desc">No police stations found within 1km of the route cells.</div>'
                                f'</div>'
                                f'</div>'
                            )
                            
                        # Construct Advice Html (flat strings with no leading indentation on each line)
                        recommendation_html = ""
                        icons = ["✅", "📞", "📍", "💡"]
                        for idx, advice in enumerate(advice_bullets[:3]):
                            icon = icons[idx % len(icons)]
                            recommendation_html += (
                                f'<div class="insight-item">'
                                f'<span class="insight-icon">{icon}</span>'
                                f'<div class="insight-content">'
                                f'<div class="insight-desc" style="color:#334155; font-weight:500; font-size:0.95rem; margin-top:0;">{advice}</div>'
                                f'</div>'
                                f'</div>'
                            )
                            
                        # Display Safety Insights Grid
                        st.markdown('<h3 style="font-family:\'Outfit\', sans-serif; font-weight:700; color:#1E293B; margin-top:2.5rem; margin-bottom:1rem;">🛡️ SafeGrid Route Insights</h3>', unsafe_allow_html=True)
                        ins_col1, ins_col2, ins_col3 = st.columns(3)
                        
                        with ins_col1:
                            st.markdown(
                                f'<div class="insight-card">'
                                f'<div class="insight-header">📊 Safety Assessment</div>'
                                f'<div class="insight-content">{assessment_html}</div>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                            
                        with ins_col2:
                            st.markdown(
                                f'<div class="insight-card">'
                                f'<div class="insight-header">🚓 Nearby Police Stations</div>'
                                f'<div class="insight-content">{police_html}</div>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                            
                        with ins_col3:
                            st.markdown(
                                f'<div class="insight-card">'
                                f'<div class="insight-header">💡 Safety Recommendations</div>'
                                f'<div class="insight-content">{recommendation_html}</div>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                            
                except Exception as e:
                    st.error(f"Error during route planning: {e}")

with tab2:
    with st.form("sos_form"):
        st.markdown('<div style="font-family:\'Outfit\', sans-serif; font-size:1.4rem; font-weight:700; color:#DC2626; margin-bottom:1.5rem;">🚨 Emergency Help & SOS Locator</div>', unsafe_allow_html=True)
        user_location = st.text_input("Enter Your Current Location or Nearby Landmark", value="Connaught Place", key="sos_loc")
        
        # Pulse-animated button container
        st.markdown('<div class="sos-btn-container">', unsafe_allow_html=True)
        btn_sos_submitted = st.form_submit_button("Activate SOS & Find Help")
        st.markdown('</div>', unsafe_allow_html=True)
        
    if btn_sos_submitted:
        if not user_location:
            st.error("Please enter your location.")
        else:
            with st.spinner("Emergency Agent is locating nearest help stations..."):
                try:
                    res = None
                    try:
                        runner = InMemoryRunner(node=emergency_workflow)
                        events = asyncio.run(runner.run_debug(f"User Location: {user_location}"))
                        
                        output_event = next((e for e in reversed(events) if e.output is not None), None)
                        res = output_event.output if output_event else None
                    except Exception as agent_err:
                        st.info("ℹ️ **Gemini API Limit Reached**: Fetching nearby emergency services locally.")
                        res = local_emergency_solver(user_location)
                        
                    if not res or not isinstance(res, dict):
                        st.error("Emergency Agent did not return a valid response and local fallback failed.")
                    else:
                        # Wrap emergency information in a styled container
                        st.markdown(f'<div class="sos-output-container">\n\n{res["emergency_info"]}\n\n</div>', unsafe_allow_html=True)
                        
                        # Let's geocode user location to render a map
                        from geopy.geocoders import Nominatim
                        geolocator = Nominatim(user_agent="safegrid")
                        query = user_location if "delhi" in user_location.lower() else f"{user_location}, Delhi"
                        geo = geolocator.geocode(query)
                        
                        if geo:
                            user_lat, user_lon = geo.latitude, geo.longitude
                            m_sos = folium.Map(location=[user_lat, user_lon], zoom_start=14)
                            
                            # User marker
                            folium.Marker(
                                [user_lat, user_lon], 
                                popup="YOUR LOCATION", 
                                icon=folium.Icon(color='red', icon='user')
                            ).add_to(m_sos)
                            
                            # Find and add nearest police station
                            try:
                                df_p = routing_service.police_df
                                dist_sq = (df_p['Latitude'] - user_lat)**2 + (df_p['Longitude'] - user_lon)**2
                                closest_p = df_p.loc[dist_sq.idxmin()]
                                folium.Marker(
                                    [closest_p['Latitude'], closest_p['Longitude']],
                                    popup=f"Police: {closest_p['Police Station']}",
                                    icon=folium.Icon(color='blue', icon='shield')
                                ).add_to(m_sos)
                            except:
                                pass
                                
                            # Find and add nearest metro station
                            try:
                                df_m = routing_service.location_df
                                dist_sq = (df_m['Latitude'] - user_lat)**2 + (df_m['Longitude'] - user_lon)**2
                                closest_m = df_m.loc[dist_sq.idxmin()]
                                folium.Marker(
                                    [closest_m['Latitude'], closest_m['Longitude']],
                                    popup=f"Metro: {closest_m['Station Names']}",
                                    icon=folium.Icon(color='purple', icon='subway')
                                ).add_to(m_sos)
                            except:
                                pass
                                
                            # Add hospital stub marker
                            hospitals = [
                                {"name": "Lok Nayak Hospital", "lat": 28.6366, "lon": 77.2407},
                                {"name": "Dr. Ram Manohar Lohia Hospital", "lat": 28.6253, "lon": 77.2007},
                                {"name": "AIIMS New Delhi", "lat": 28.5672, "lon": 77.2100},
                                {"name": "Max Super Speciality Hospital, Shalimar Bagh", "lat": 28.7180, "lon": 77.1585},
                                {"name": "Sir Ganga Ram Hospital", "lat": 28.6385, "lon": 77.1895}
                            ]
                            closest_h = min(hospitals, key=lambda h: (h['lat'] - user_lat)**2 + (h['lon'] - user_lon)**2)
                            folium.Marker(
                                [closest_h['lat'], closest_h['lon']],
                                popup=f"Hospital: {closest_h['name']} (Stub)",
                                icon=folium.Icon(color='green', icon='plus-sign')
                            ).add_to(m_sos)
                            
                            st.markdown('<div class="custom-card" style="padding: 1.5rem 1.5rem 0.5rem 1.5rem;"><div class="insight-header">🗺️ Nearest Services Map</div>', unsafe_allow_html=True)
                            st_folium(
                                m_sos,
                                width=None,
                                height=450,
                                returned_objects=[],
                                key="sos_map",
                                use_container_width=True
                            )
                            st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error during SOS retrieval: {e}")