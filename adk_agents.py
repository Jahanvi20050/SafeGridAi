import os
import json
import re
import asyncio
import threading
from google.adk.models.lite_llm import LiteLlm
from google.adk import Agent, Workflow, Context
from google.adk.workflow import RetryConfig
from routing_service import RoutingService

# 1. Configuration & Service Setup
MODEL = LiteLlm(
    model="groq/llama-3.1-8b-instant"
)

agent_retry_config = RetryConfig(
    max_attempts=5,
    initial_delay=2.0,
    max_delay=60.0,
    backoff_factor=2.0,
    jitter=1.0
)

# Resolve paths relative to the current file (adk_agents.py is in SafeGridAi/)
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")

CRIME_PATH = os.path.join(data_dir, "crime_dataset (1) (1).json")
METRO_PATH = os.path.join(data_dir, "Delhi metro.csv")
BOUNDARY_PATH = os.path.join(data_dir, "Delhi_Boundary.geojson")
POLICE_CSV_PATH = os.path.join(data_dir, "delhi_police_station_locs.csv")

routing_service = RoutingService(
    crime_path=CRIME_PATH,
    metro_path=METRO_PATH,
    boundary_path=BOUNDARY_PATH,
    police_csv_path=POLICE_CSV_PATH
)

# 2. Deterministic Workflow Nodes

def route_planner_node(ctx: Context, node_input: str) -> str:
    # Extract source and destination
    source = "IGDTUW"
    destination = "Connaught Place"
    match = re.search(r"Source:\s*(.*?),\s*Destination:\s*(.*)", node_input, re.IGNORECASE)
    if match:
        source = match.group(1).strip()
        destination = match.group(2).strip()
        
    from geopy.geocoders import Nominatim
    geolocator = Nominatim(user_agent="safegrid")
    
    # Geocode start
    query_start = source if "delhi" in source.lower() else f"{source}, Delhi"
    try:
        geo_start = geolocator.geocode(query_start)
        if not geo_start:
            raise ValueError(f"Source location '{source}' not found")
    except Exception as e:
        raise ValueError(f"Failed to geocode source '{source}': {str(e)}")
        
    # Geocode end
    query_end = destination if "delhi" in destination.lower() else f"{destination}, Delhi"
    try:
        geo_end = geolocator.geocode(query_end)
        if not geo_end:
            raise ValueError(f"Destination location '{destination}' not found")
    except Exception as e:
        raise ValueError(f"Failed to geocode destination '{destination}': {str(e)}")
        
    res = routing_service.find_safest_route(
        geo_start.latitude, geo_start.longitude,
        geo_end.latitude, geo_end.longitude
    )
    serializable_res = {
        "route_found": res["route_found"],
        "distance_km": res["distance_km"],
        "eta_min": res["eta_min"],
        "eta_minutes": res["eta_min"],
        "route_cells": res["route_cells"]
    }
    res_str = json.dumps(serializable_res)
    ctx.state["route_info"] = res_str
    return res_str

def crime_analyzer_node(ctx: Context) -> str:
    route_info_str = ctx.state.get("route_info", "{}")
    route_info = json.loads(route_info_str)
    cells = route_info.get("route_cells", [])
    
    # Risk analysis
    if not cells:
        risk_score = 0
        risk_level = "UNKNOWN"
    else:
        scores = [c["safety_score"] for c in cells]
        avg_safety = sum(scores) / len(scores)
        risk_score = int(round((1 - avg_safety) * 100))
        if risk_score < 30:
            risk_level = "LOW"
        elif risk_score < 60:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
            
    # Hotspot detection
    hotspots = []
    risky_cells = [c for c in cells if c.get("safety_score", 1.0) < 0.5]
    df_m = routing_service.location_df
    for cell in risky_cells:
        lat = cell.get("lat") or cell.get("latitude")
        lon = cell.get("lon") or cell.get("longitude")
        safety_score = cell.get("safety_score", 0.0)
        if lat is not None and lon is not None:
            dist_sq = (df_m['Latitude'] - float(lat))**2 + (df_m['Longitude'] - float(lon))**2
            idx = dist_sq.idxmin()
            closest_station = df_m.loc[idx]['Station Names']
            risk_desc = "High" if safety_score < 0.3 else "Moderate"
            hotspots.append(f"{closest_station} Area — {risk_desc} Risk")
    unique_hotspots = []
    for h in hotspots:
        if h not in unique_hotspots:
            unique_hotspots.append(h)
    unique_hotspots = unique_hotspots[:5]
    
    res = {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "hotspots": unique_hotspots
    }
    res_str = json.dumps(res)
    ctx.state["risk_info"] = res_str
    return res_str

def police_on_route_node(ctx: Context) -> str:
    route_info_str = ctx.state.get("route_info", "{}")
    route_info = json.loads(route_info_str)
    cells = route_info.get("route_cells", [])
    if not cells or len(cells) < 2:
        res_str = "[]"
    else:
        from shapely.geometry import Point, LineString
        import geopandas as gpd
        import pandas as pd
        
        route_coords = []
        for c in cells:
            if isinstance(c, dict):
                lat = c.get("lat") or c.get("latitude")
                lon = c.get("lon") or c.get("longitude")
                if lat is not None and lon is not None:
                    route_coords.append((float(lon), float(lat)))
                    
        if len(route_coords) < 2:
            res_str = "[]"
        else:
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
            
            df_p = df_p.copy()
            df_p["distance_m"] = distances_m
            nearby_df = df_p[df_p["distance_m"] <= 1000.0].copy()
            nearby_df = nearby_df.sort_values(by="distance_m")
            nearby_df = nearby_df.drop_duplicates(subset=["Police Station"], keep="first")
            
            nearby_stations = []
            for _, row in nearby_df.iterrows():
                nearby_stations.append({
                    "name": str(row["Police Station"]),
                    "lat": float(row["Latitude"]),
                    "lon": float(row["Longitude"]),
                    "distance_m": float(row["distance_m"])
                })
            res_str = json.dumps(nearby_stations)
            
    ctx.state["emergency_output"] = res_str
    return res_str

def geocode_node(ctx: Context, node_input: str) -> str:
    location = node_input
    match = re.search(r"User Location:\s*(.*)", node_input, re.IGNORECASE)
    if match:
        location = match.group(1).strip()
        
    from geopy.geocoders import Nominatim
    geolocator = Nominatim(user_agent="safegrid")
    query_loc = location if "delhi" in location.lower() else f"{location}, Delhi"
    try:
        geo_loc = geolocator.geocode(query_loc)
        if not geo_loc:
            lat, lon = 28.6304, 77.2177
        else:
            lat, lon = geo_loc.latitude, geo_loc.longitude
    except Exception:
        lat, lon = 28.6304, 77.2177
        
    coords = {"lat": lat, "lon": lon, "location_name": location}
    coords_str = json.dumps(coords)
    ctx.state["coordinates"] = coords_str
    return coords_str

def emergency_lookup_node(ctx: Context) -> str:
    coords_str = ctx.state.get("coordinates", "{}")
    coords = json.loads(coords_str)
    lat = coords.get("lat", 28.6304)
    lon = coords.get("lon", 77.2177)
    
    # 1. Nearest Police Station
    df_p = routing_service.police_df
    dist_sq_p = (df_p['Latitude'] - lat)**2 + (df_p['Longitude'] - lon)**2
    idx_p = dist_sq_p.idxmin()
    closest_p = df_p.loc[idx_p]
    police_data = {
        "name": closest_p['Police Station'],
        "lat": float(closest_p['Latitude']),
        "lon": float(closest_p['Longitude'])
    }
    
    # 2. Nearest Metro
    df_m = routing_service.location_df
    dist_sq_m = (df_m['Latitude'] - lat)**2 + (df_m['Longitude'] - lon)**2
    idx_m = dist_sq_m.idxmin()
    closest_m = df_m.loc[idx_m]
    metro_data = {
        "name": closest_m['Station Names'],
        "line": closest_m['Metro Line'],
        "lat": float(closest_m['Latitude']),
        "lon": float(closest_m['Longitude'])
    }
    
    # 3. Nearest Hospital
    hospitals = [
        {"name": "Lok Nayak Hospital", "lat": 28.6366, "lon": 77.2407},
        {"name": "Dr. Ram Manohar Lohia Hospital", "lat": 28.6253, "lon": 77.2007},
        {"name": "AIIMS New Delhi", "lat": 28.5672, "lon": 77.2100},
        {"name": "Max Super Speciality Hospital, Shalimar Bagh", "lat": 28.7180, "lon": 77.1585},
        {"name": "Sir Ganga Ram Hospital", "lat": 28.6385, "lon": 77.1895}
    ]
    closest_h = min(hospitals, key=lambda h: (h['lat'] - lat)**2 + (h['lon'] - lon)**2)
    
    res = {
        "nearest_police": police_data,
        "nearest_metro": metro_data,
        "nearest_hospital": closest_h,
        "location_name": coords.get("location_name", "User Location")
    }
    res_str = json.dumps(res)
    ctx.state["emergency_info_json"] = res_str
    return res_str

# 3. Agent Declarations

safety_agent = Agent(
    name="SafetyAgent",
    model=MODEL,
    description="Generates final user-facing safety suggestions and route analysis summary.",
    instruction="""
    You are the Route Safety Recommendation Agent.
    
    You will receive in your context state:
    - `route_info`: A JSON string containing the planned route details (distance_km, eta_min, etc.).
    - `risk_info`: A JSON string containing the risk_score (0-100), risk_level (LOW, MEDIUM, HIGH), and active crime hotspots.
    - `emergency_output`: A JSON string representing a list of nearby police stations along the route with their distances.

    Your job is to generate a clean, reassuring, and user-friendly summary of the route safety assessment in Markdown format.
    
    Please include:
    1. ### Safety Assessment
       - **Route Summary**: Summarize the route distance (in km) and ETA (in minutes).
       - **Risk Level**: State the risk level (LOW/MEDIUM/HIGH) with an appropriate emoji.
       - **Risk Score**: Show the safety risk score (0-100%).
    2. ### Nearby Police Stations Along Route
       - List the names of unique police stations near the route with their distances in meters.
       - If the list is empty, state: 'No police stations found within 1km of the route.'
    3. ### Active Crime Hotspots
       - List any hotspots found. If none, state: 'No major crime hotspots detected along this route.'
    4. ### Safety Advice & Recommendations
       - Give tailored, actionable safety precautions based on the risk level (e.g. share live location, prefer Metro for high-risk routes, stay on lit roads).
    """,
    output_key="final_recommendation",
    retry_config=agent_retry_config
)

emergency_agent = Agent(
    name="EmergencyAgent",
    model=MODEL,
    description="Formats final emergency SOS response.",
    instruction="""
    You are the Emergency Response formatting agent.
    
    You will receive in your context state:
    - `emergency_info_json`: A JSON string with details of the nearest emergency services (police, metro, hospital).

    Your job is to generate a clean, clear, and reassuring emergency SOS response in Markdown format.
    
    Include:
    1. ### 🚨 Emergency Safety Services near the user location:
       - **Nearest Police Station**: Name and coordinates.
       - **Nearest Metro Station**: Name and Line color (with coordinates).
       - **Nearest Hospital**: Name and coordinates.
    2. ### 🛡️ Quick Safety Advice for SOS Situations:
       - List 3 quick, actionable safety advice steps for an SOS situation (e.g., move to a public place, share live location, call 112/100).
    """,
    output_key="emergency_output",
    retry_config=agent_retry_config
)

# 4. Workflows Setup

def return_safety_output(ctx: Context) -> dict:
    try:
        route_info = json.loads(ctx.state.get("route_info", "{}"))
    except:
        route_info = {}
    try:
        risk_info = json.loads(ctx.state.get("risk_info", "{}"))
    except:
        risk_info = {}
    try:
        police_on_route = json.loads(ctx.state.get("emergency_output", "[]"))
        if not isinstance(police_on_route, list):
            police_on_route = []
    except:
        police_on_route = []
        
    return {
        "route_found": route_info.get("route_found", False),
        "distance_km": route_info.get("distance_km", 0.0),
        "eta_min": route_info.get("eta_min", 0),
        "route_cells": route_info.get("route_cells", []),
        "risk_score": risk_info.get("risk_score", 0),
        "risk_level": risk_info.get("risk_level", "UNKNOWN"),
        "hotspots": risk_info.get("hotspots", []),
        "police_on_route": police_on_route,
        "recommendation": ctx.state.get("final_recommendation", "")
    }

def return_emergency_output(ctx: Context) -> dict:
    return {
        "emergency_info": ctx.state.get("emergency_output", "No emergency services found.")
    }

safety_workflow = Workflow(
    name="SafetyWorkflow",
    edges=[
        ("START", route_planner_node),
        (route_planner_node, crime_analyzer_node),
        (crime_analyzer_node, police_on_route_node),
        (police_on_route_node, safety_agent),
        (safety_agent, return_safety_output)
    ]
)

emergency_workflow = Workflow(
    name="EmergencyWorkflow",
    edges=[
        ("START", geocode_node),
        (geocode_node, emergency_lookup_node),
        (emergency_lookup_node, emergency_agent),
        (emergency_agent, return_emergency_output)
    ]
)
