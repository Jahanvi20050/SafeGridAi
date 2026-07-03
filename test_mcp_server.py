import asyncio
import json
import sys
import os

# Ensure the workspace directory is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from safegrid_mcp_server import (
    get_safe_route,
    analyze_route_risk,
    detect_hotspots,
    nearest_police_station,
    nearest_metro,
    nearest_hospital,
    get_safety_recommendation
)

async def test_tools():
    print("==================================================")
    print("Testing get_safe_route...")
    route_json_str = await get_safe_route("IGDTUW", "Connaught Place")
    route_res = json.loads(route_json_str)
    print(f"  Route Found: {route_res.get('route_found')}")
    if not route_res.get('route_found'):
        print(f"  Error: {route_res.get('error')}")
        return
        
    print(f"  Distance: {route_res.get('distance_km')} km")
    print(f"  ETA: {route_res.get('eta_min')} mins")
    
    route_cells = route_res.get('route_cells')
    print(f"  Number of route cells: {len(route_cells)}")
    
    print("\nTesting analyze_route_risk...")
    risk_json_str = await analyze_route_risk(route_cells)
    risk_res = json.loads(risk_json_str)
    print(f"  Risk Score: {risk_res.get('risk_score')}%")
    print(f"  Risk Level: {risk_res.get('risk_level')}")
    
    print("\nTesting detect_hotspots...")
    hotspots_json_str = await detect_hotspots(route_cells)
    hotspots_res = json.loads(hotspots_json_str)
    print(f"  Hotspots count: {len(hotspots_res)}")
    for idx, hs in enumerate(hotspots_res):
        print(f"    - {hs}")
        
    print("\nTesting get_safety_recommendation...")
    rec_json_str = await get_safety_recommendation(risk_res.get('risk_score'), risk_res.get('risk_level'))
    rec_res = json.loads(rec_json_str)
    print(f"  Recommendations count: {len(rec_res)}")
    for idx, rec in enumerate(rec_res):
        print(f"    - {rec}")
        
    print("\nTesting nearest_police_station (Delhi CP coords)...")
    police_res = json.loads(await nearest_police_station(28.6304, 77.2177))
    print(f"  Nearest Police Station: {police_res.get('name')} at ({police_res.get('lat')}, {police_res.get('lon')})")
    
    print("\nTesting nearest_metro...")
    metro_res = json.loads(await nearest_metro(28.6304, 77.2177))
    print(f"  Nearest Metro: {metro_res.get('name')} ({metro_res.get('line')} line) at ({metro_res.get('lat')}, {metro_res.get('lon')})")

    print("\nTesting nearest_hospital...")
    hosp_res = json.loads(await nearest_hospital(28.6304, 77.2177))
    print(f"  Nearest Hospital: {hosp_res.get('name')} at ({hosp_res.get('lat')}, {hosp_res.get('lon')})")
    print("==================================================")

if __name__ == "__main__":
    asyncio.run(test_tools())
