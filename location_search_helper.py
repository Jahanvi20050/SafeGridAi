import time
import requests
import streamlit as st

def enforce_rate_limit():
    """Enforces at least 1.0 second delay between requests to Nominatim API."""
    if "last_nominatim_request_time" not in st.session_state:
        st.session_state["last_nominatim_request_time"] = 0.0
    
    now = time.time()
    elapsed = now - st.session_state["last_nominatim_request_time"]
    delay = 1.0 - elapsed
    if delay > 0:
        time.sleep(delay)
    st.session_state["last_nominatim_request_time"] = time.time()

@st.cache_data(show_spinner=False, ttl=600)
def search_nominatim(query):
    """
    Fetches up to 5 location suggestions from Nominatim Search API.
    Filters search results to Delhi, India.
    Handles rate limits, errors, and caching.
    """
    if not query or len(query.strip()) < 3:
        return []

    # Enforce rate limit on cache miss
    enforce_rate_limit()

    full_query = f"{query.strip()}, Delhi, India"
    
    headers = {
        "User-Agent": "SafeGridAi-WomenSafetyRoutePlanner/1.0 (contact: safegrid.safety.route@gmail.com)"
    }
    params = {
        "q": full_query,
        "format": "json",
        "limit": 5,
        "addressdetails": 1
    }
    
    url = "https://nominatim.openstreetmap.org/search"
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            return "RATE_LIMIT_ERROR"
        else:
            return "API_ERROR"
    except (requests.exceptions.RequestException, Exception) as e:
        # Internal logging, no traceback shown to user
        import logging
        logging.getLogger("nominatim_helper").error(f"Error fetching suggestions for '{query}': {e}")
        return "NETWORK_ERROR"
