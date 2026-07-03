import os
import json
from typing import Any
from mcp.server.fastmcp import FastMCP
from routing_service import RoutingService

# 1. Configuration & Service Setup
CRIME_PATH = r"C:\Users\dell\OneDrive\Documents\JN\all syllabussss\discrete mathematics\semester 4\crime_dataset (1) (1).json"
METRO_PATH = r"C:\Users\dell\OneDrive\Documents\JN\all syllabussss\discrete mathematics\semester 4\Delhi metro.csv"
BOUNDARY_PATH = r"C:\Users\dell\OneDrive\Documents\JN\all syllabussss\discrete mathematics\semester 4\Delhi_Boundary.geojson"
POLICE_CSV_PATH = r"c:\Users\dell\OneDrive\Documents\JN\all syllabussss\discrete mathematics\semester 4\delhi_police_station_locs.csv"

routing_service = RoutingService(
    crime_path=CRIME_PATH,
    metro_path=METRO_PATH,
    boundary_path=BOUNDARY_PATH,
    police_csv_path=POLICE_CSV_PATH
)

# 2. FastMCP Instance Creation
mcp = FastMCP("SafeGrid")

# 3. Tool Registrations

@mcp.tool()
async def get_safe_route(source: str, destination: str) -> str:
    """Geocodes source and destination, calls routing service, and returns safest route coordinates, distance, and ETA."""
    from geopy.geocoders import Nominatim
    geolocator = Nominatim(user_agent="safegrid")
    
    # Geocode start
    query_start = source if "delhi" in source.lower() else f"{source}, Delhi"
    try:
        geo_start = geolocator.geocode(query_start)
        if not geo_start:
            return json.dumps({"route_found": False, "error": f"Source location '{source}' not found"})
    except Exception as e:
        return json.dumps({"route_found": False, "error": f"Failed to geocode source: {str(e)}"})
        
    # Geocode end
    query_end = destination if "delhi" in destination.lower() else f"{destination}, Delhi"
    try:
        geo_end = geolocator.geocode(query_end)
        if not geo_end:
            return json.dumps({"route_found": False, "error": f"Destination location '{destination}' not found"})
    except Exception as e:
        return json.dumps({"route_found": False, "error": f"Failed to geocode destination: {str(e)}"})
        
    # Call routing service
    try:
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
        return json.dumps(serializable_res)
    except Exception as e:
        return json.dumps({"route_found": False, "error": f"Routing failed: {str(e)}"})

def parse_route_coordinates(route_coordinates: Any) -> list:
    if not route_coordinates:
        return []
    if not isinstance(route_coordinates, str):
        return route_coordinates
        
    s = route_coordinates.strip()
    # Strip markdown code blocks
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 2:
            s = "\n".join(lines[1:-1]).strip()
            
    # Find first '[' and last ']' to extract the JSON array
    start_idx = s.find('[')
    end_idx = s.rfind(']')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        s = s[start_idx:end_idx+1]
        
    try:
        return json.loads(s)
    except Exception as e:
        try:
            val = json.loads(route_coordinates)
            if isinstance(val, dict) and "route_cells" in val:
                return val["route_cells"]
            return val
        except:
            raise e

@mcp.tool()
async def analyze_route_risk(route_coordinates: Any) -> str:
    """Analyzes the safety scores along the route cells to determine average risk score and level (LOW, MEDIUM, HIGH)."""
    try:
        cells = parse_route_coordinates(route_coordinates)
        
        if not cells:
            return json.dumps({"risk_score": 0, "risk_level": "UNKNOWN"})
            
        scores = [c["safety_score"] for c in cells]
        avg_safety = sum(scores) / len(scores)
        risk_score = int(round((1 - avg_safety) * 100))
        
        if risk_score < 30:
            risk_level = "LOW"
        elif risk_score < 60:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
            
        return json.dumps({"risk_score": risk_score, "risk_level": risk_level})
    except Exception as e:
        return json.dumps({"error": f"Failed to analyze risk: {str(e)}"})

@mcp.tool()
async def detect_hotspots(route_coordinates: Any) -> str:
    """Detects crime hotspots (cells with safety score below 0.5) along the route."""
    try:
        cells = parse_route_coordinates(route_coordinates)
            
        if not cells:
            return "[]"
            
        hotspots = []
        risky_cells = [c for c in cells if c.get("safety_score", 1.0) < 0.5]
        df_m = routing_service.location_df
        
        for cell in risky_cells:
            lat = cell.get("lat") or cell.get("latitude")
            lon = cell.get("lon") or cell.get("longitude")
            safety_score = cell.get("safety_score", 0.0)
            
            if lat is not None and lon is not None:
                # Find nearest metro station name locally
                dist_sq = (df_m['Latitude'] - float(lat))**2 + (df_m['Longitude'] - float(lon))**2
                idx = dist_sq.idxmin()
                closest_station = df_m.loc[idx]['Station Names']
                
                # Determine risk level based on safety score
                risk_desc = "High" if safety_score < 0.3 else "Moderate"
                hotspots.append(f"{closest_station} Area — {risk_desc} Risk")
                
        # Keep unique area risk labels
        unique_hotspots = []
        for h in hotspots:
            if h not in unique_hotspots:
                unique_hotspots.append(h)
                
        return json.dumps(unique_hotspots[:5])
    except Exception as e:
        return json.dumps({"error": f"Failed to detect hotspots: {str(e)}"})

@mcp.tool()
async def nearest_police_station(latitude: float, longitude: float) -> str:
    """Finds the nearest police station from coordinates."""
    try:
        df_p = routing_service.police_df
        dist_sq = (df_p['Latitude'] - latitude)**2 + (df_p['Longitude'] - longitude)**2
        idx = dist_sq.idxmin()
        closest = df_p.loc[idx]
        return json.dumps({
            "name": closest['Police Station'],
            "lat": float(closest['Latitude']),
            "lon": float(closest['Longitude'])
        })
    except Exception as e:
        return json.dumps({"error": str(e)})

@mcp.tool()
async def nearest_metro(latitude: float, longitude: float) -> str:
    """Finds the nearest Delhi metro station coordinates and line color."""
    try:
        df_m = routing_service.location_df
        dist_sq = (df_m['Latitude'] - latitude)**2 + (df_m['Longitude'] - longitude)**2
        idx = dist_sq.idxmin()
        closest = df_m.loc[idx]
        return json.dumps({
            "name": closest['Station Names'],
            "line": closest['Metro Line'],
            "lat": float(closest['Latitude']),
            "lon": float(closest['Longitude'])
        })
    except Exception as e:
        return json.dumps({"error": str(e)})

@mcp.tool()
async def nearest_hospital(latitude: float, longitude: float) -> str:
    """Finds the nearest emergency hospital from coordinates."""
    try:
        hospitals = [
            {"name": "Lok Nayak Hospital", "lat": 28.6366, "lon": 77.2407},
            {"name": "Dr. Ram Manohar Lohia Hospital", "lat": 28.6253, "lon": 77.2007},
            {"name": "AIIMS New Delhi", "lat": 28.5672, "lon": 77.2100},
            {"name": "Max Super Speciality Hospital, Shalimar Bagh", "lat": 28.7180, "lon": 77.1585},
            {"name": "Sir Ganga Ram Hospital", "lat": 28.6385, "lon": 77.1895}
        ]
        closest = min(hospitals, key=lambda h: (h['lat'] - latitude)**2 + (h['lon'] - longitude)**2)
        return json.dumps(closest)
    except Exception as e:
        return json.dumps({"error": str(e)})

@mcp.tool()
async def get_safety_recommendation(risk_score: float, risk_level: str) -> str:
    """Generates user safety recommendations based on the risk score and level."""
    try:
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
        return json.dumps(advice)
    except Exception as e:
        return json.dumps({"error": str(e)})

@mcp.tool()
async def find_police_on_route(route_coordinates: Any) -> str:
    """Finds all unique police stations located near the route (within a 1km threshold)."""
    try:
        cells = parse_route_coordinates(route_coordinates)
        if not cells or len(cells) < 2:
            return "[]"
            
        from shapely.geometry import Point, LineString
        import geopandas as gpd
        import pandas as pd
        
        # 1. Construct the LineString from the route cells in WGS84 (EPSG:4326)
        route_coords = []
        for c in cells:
            if isinstance(c, dict):
                lat = c.get("lat") or c.get("latitude")
                lon = c.get("lon") or c.get("longitude")
                if lat is not None and lon is not None:
                    route_coords.append((float(lon), float(lat)))
            elif isinstance(c, (list, tuple)) and len(c) >= 2:
                val1, val2 = float(c[0]), float(c[1])
                # Delhi bounding box: lat ~ 28, lon ~ 77
                if 28.0 <= val1 <= 29.0 and 76.0 <= val2 <= 78.0:
                    lat, lon = val1, val2
                elif 28.0 <= val2 <= 29.0 and 76.0 <= val1 <= 78.0:
                    lon, lat = val1, val2
                else:
                    lat, lon = val1, val2  # Default to [lat, lon]
                route_coords.append((lon, lat))
                
        if len(route_coords) < 2:
            return "[]"
        route_line = LineString(route_coords)
        gdf_route = gpd.GeoDataFrame(geometry=[route_line], crs="EPSG:4326")
        
        # Project route to UTM 43N (EPSG:32643) for meter-based distance calculations
        gdf_route_proj = gdf_route.to_crs(epsg=32643)
        route_geom_proj = gdf_route_proj.geometry.iloc[0]
        
        # 2. Load the police station dataset
        df_p = pd.read_csv(POLICE_CSV_PATH)
        
        # Create GeoDataFrame in WGS84
        gdf_police = gpd.GeoDataFrame(
            df_p,
            geometry=gpd.points_from_xy(df_p["Longitude"], df_p["Latitude"]),
            crs="EPSG:4326"
        )
        
        # Project police stations to UTM 43N (EPSG:32643)
        gdf_police_proj = gdf_police.to_crs(epsg=32643)
        
        # 3. Compute shortest distance from each police station point to the LineString
        distances_m = gdf_police_proj.geometry.distance(route_geom_proj)
        
        # 4. Filter stations within 1km (1000 meters)
        threshold_m = 1000.0
        df_p["distance_m"] = distances_m
        nearby_df = df_p[df_p["distance_m"] <= threshold_m].copy()
        
        # Sort by distance (closest first)
        nearby_df = nearby_df.sort_values(by="distance_m")
        
        # 5. Deduplication Guard: keep the closest unique police station name
        nearby_df = nearby_df.drop_duplicates(subset=["Police Station"], keep="first")
        
        # 6. Format output
        nearby_stations = []
        for _, row in nearby_df.iterrows():
            nearby_stations.append({
                "name": str(row["Police Station"]),
                "lat": float(row["Latitude"]),
                "lon": float(row["Longitude"]),
                "distance_m": float(row["distance_m"])
            })
            
        return json.dumps(nearby_stations)
    except Exception as e:
        return json.dumps({"error": f"Failed to find police on route: {str(e)}"})

if __name__ == "__main__":
    mcp.run()
