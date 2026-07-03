import asyncio
import sys
import os
import json
import logging
from dotenv import load_dotenv

# Configure logging to see MCP details and stack traces
logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)

# Load env variables including GEMINI_API_KEY
load_dotenv(override=True)

# Ensure the workspace directory is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from google.adk.runners import InMemoryRunner
from adk_agents import safety_workflow, emergency_workflow

async def run_safety_workflow():
    print("==================================================")
    print("RUNNING SAFETY WORKFLOW (Route Agent -> Crime Agent -> Rec Agent)...")
    print("==================================================")
    runner = InMemoryRunner(node=safety_workflow)
    events = await runner.run_debug("Source: IGDTUW, Destination: Connaught Place")
    
    # Print out agent events
    for e in events:
        agent_name = getattr(e, 'agent_name', None)
        if agent_name:
            print(f"\n--- Agent: {agent_name} ---")
            if hasattr(e, 'output') and e.output:
                print(f"Output: {e.output}")
                
    output_event = next((e for e in reversed(events) if e.output is not None), None)
    res = output_event.output if output_event else None
    print("\n==================================================")
    print("FINAL WORKFLOW OUTPUT:")
    print("==================================================")
    if res:
        print(f"Route Found: {res.get('route_found')}")
        print(f"Distance: {res.get('distance_km')} km")
        print(f"ETA: {res.get('eta_min')} mins")
        print(f"Risk Score: {res.get('risk_score')}%")
        print(f"Risk Level: {res.get('risk_level')}")
        print(f"Hotspots Found: {len(res.get('hotspots', []))}")
        print(f"Recommendation Preview:\n{res.get('recommendation')[:300]}...")
    else:
        print("No response from workflow!")
    print("==================================================\n")

async def run_emergency_workflow():
    print("==================================================")
    print("RUNNING EMERGENCY WORKFLOW (Emergency Agent)...")
    print("==================================================")
    runner = InMemoryRunner(node=emergency_workflow)
    events = await runner.run_debug("User Location: Connaught Place")
    
    for e in events:
        agent_name = getattr(e, 'agent_name', None)
        if agent_name:
            print(f"\n--- Agent: {agent_name} ---")
            if hasattr(e, 'output') and e.output:
                print(f"Output:\n{e.output}")
                
    output_event = next((e for e in reversed(events) if e.output is not None), None)
    res = output_event.output if output_event else None
    print("\n==================================================")
    print("FINAL SOS OUTPUT:")
    print("==================================================")
    if res:
        print(res.get("emergency_info"))
    else:
        print("No response from workflow!")
    print("==================================================")

if __name__ == "__main__":
    # Ensure any background loop tasks are handled
    asyncio.run(run_safety_workflow())
    asyncio.run(run_emergency_workflow())
