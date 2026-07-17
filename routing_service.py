import os
import json
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Point
import networkx as nx
from scipy.spatial import KDTree
import osmnx as ox

# Global Configuration
BETA = 3.0  # Configurable safety penalty parameter

class RoutingService:
    def __init__(self, crime_path, metro_path, boundary_path, police_csv_path):
        self.crime_path = self._resolve_path(crime_path)
        self.metro_path = self._resolve_path(metro_path)
        self.boundary_path = self._resolve_path(boundary_path)
        self.police_csv_path = self._resolve_path(police_csv_path)

        # Configure OSMnx global settings (Timeout, cache)
        ox.settings.overpass_url = "https://lz4.overpass-api.de/api"
        ox.settings.timeout = 60
        ox.settings.requests_timeout = 60
        ox.settings.use_cache = True
        ox.settings.cache_folder = "./cache"

        # 1. Load Crime Dataset
        with open(self.crime_path, 'r') as file:
            self.data = json.load(file)
        self.df = pd.DataFrame(self.data)
        self.df.rename(columns={'longitude': 'Longitude', 'latitude': 'Latitude'}, inplace=True)

        # 2. Build Projected KDTree for exact meter-based distance crime queries
        gdf_crime = gpd.GeoDataFrame(
            self.df,
            geometry=gpd.points_from_xy(self.df["Longitude"], self.df["Latitude"]),
            crs="EPSG:4326"
        ).to_crs(epsg=32643) # UTM zone 43N
        crime_coords_projected = np.array([[geom.x, geom.y] for geom in gdf_crime.geometry])
        self.crime_tree = KDTree(crime_coords_projected)

        # 3. Load Metro Dataset
        self.location_df = pd.read_csv(self.metro_path)

        # 4. Load Police Stations Dataset
        self.police_df = pd.read_csv(self.police_csv_path)

        # 5. Lazy loading definitions for Delhi Boundary & Fallback Grid
        self._gdf_boundary = None
        self._grid_gdf = None
        self._G_grid = None

    @property
    def gdf_boundary(self):
        if self._gdf_boundary is None:
            self._gdf_boundary = gpd.read_file(self.boundary_path).to_crs(epsg=32643)
        return self._gdf_boundary

    @property
    def grid_gdf(self):
        if self._grid_gdf is None:
            logging.info("Constructing grid cells and computing safety scores (lazy initialization)...")
            minx, miny, maxx, maxy = self.gdf_boundary.total_bounds
            self.grid_size = 250  # meters
            grid_cells = [box(x, y, x + self.grid_size, y + self.grid_size)
                          for x in np.arange(minx, maxx, self.grid_size)
                          for y in np.arange(miny, maxy, self.grid_size)]
            grid_gdf = gpd.GeoDataFrame({'geometry': grid_cells}, crs=self.gdf_boundary.crs)
            grid_gdf = gpd.overlay(grid_gdf, self.gdf_boundary, how='intersection')

            # Map Crime Counts & Compute Safety Scores for the fallback Grid
            gdf_crime_grid = gpd.GeoDataFrame(
                self.df,
                geometry=gpd.points_from_xy(self.df["Longitude"], self.df["Latitude"]),
                crs="EPSG:4326"
            ).to_crs(grid_gdf.crs)

            joined = gpd.sjoin(gdf_crime_grid, grid_gdf, how="left", predicate="within")
            crime_counts = joined.groupby('index_right').size()
            max_crime = crime_counts.max() if len(crime_counts) > 0 else 1
            grid_gdf["safety_score"] = 1 - (crime_counts / max_crime).reindex(grid_gdf.index).fillna(0)
            self._grid_gdf = grid_gdf
        return self._grid_gdf

    @property
    def G_grid(self):
        if self._G_grid is None:
            logging.info("Building grid graph (lazy initialization)...")
            self._G_grid = nx.Graph()
            for idx, row in self.grid_gdf.iterrows():
                self._G_grid.add_node(idx, geometry=row.geometry, safety_score=row.safety_score)

            sindex = self.grid_gdf.sindex
            for idx, cell in self.grid_gdf.iterrows():
                possible_neighbors = list(sindex.intersection(cell.geometry.bounds))
                for n_idx in possible_neighbors:
                    if idx != n_idx and cell.geometry.touches(self.grid_gdf.at[n_idx, "geometry"]):
                        weight = 1 - ((cell.safety_score + self.grid_gdf.at[n_idx, "safety_score"]) / 2)
                        self._G_grid.add_edge(idx, n_idx, weight=weight)
        return self._G_grid

    def _resolve_path(self, path):
        """Helper to resolve paths either as absolute or local fallback."""
        if os.path.exists(path):
            return path
        # Try local filename fallback
        basename = os.path.basename(path)
        if os.path.exists(basename):
            return basename
        # Try parent directory file fallback
        parent_path = os.path.join("..", basename)
        if os.path.exists(parent_path):
            return parent_path
        return path

    def find_safest_route(self, start_lat, start_lon, end_lat, end_lon):
        """Finds safest path between coordinates. Falls back to grid routing if OSMnx fails."""
        try:
            # 1. Calculate bbox with 0.01 degree margin (approx 1 km padding)
            margin = 0.01
            west = min(start_lon, end_lon) - margin
            south = min(start_lat, end_lat) - margin
            east = max(start_lon, end_lon) + margin
            north = max(start_lat, end_lat) + margin

            # 2. Download localized road network (walk)
            G = ox.graph_from_bbox(bbox=(west, south, east, north), network_type='walk')
            if len(G.nodes) == 0:
                raise ValueError("Downloaded road network contains 0 nodes.")

            # 3. Project road network nodes to EPSG:32643 for meter-based queries
            node_ids = list(G.nodes)
            node_coords_wgs84 = np.array([[G.nodes[n]['x'], G.nodes[n]['y']] for n in node_ids])
            
            gdf_nodes = gpd.GeoDataFrame(
                geometry=gpd.points_from_xy(node_coords_wgs84[:, 0], node_coords_wgs84[:, 1]),
                crs="EPSG:4326"
            ).to_crs(epsg=32643)
            node_coords_projected = np.array([[geom.x, geom.y] for geom in gdf_nodes.geometry])

            # 4. Query KDTree: count crimes within 250m
            radius_meters = 250.0
            #Har road node ke aas paas 250 meter ke andar kitne crimes hain?
            counts = [len(indices) for indices in self.crime_tree.query_ball_point(node_coords_projected, r=radius_meters)]
            max_count = max(counts) if counts else 1

            # Calculate node risks (0 to 1)
            node_risk = {node_ids[i]: (counts[i] / max_count) for i in range(len(node_ids))}

            # 5. Assign weights to edges
            for u, v, k, data in G.edges(keys=True, data=True):
                length = data.get('length', 1.0)
                edge_risk = (node_risk[u] + node_risk[v]) / 2.0
                data['weight'] = length * (1.0 + BETA * edge_risk)
                data['weight_balanced'] = length * (1.0 + 1.0 * edge_risk)  # Balanced penalty

            start_node = ox.nearest_nodes(G, start_lon, start_lat)
            end_node = ox.nearest_nodes(G, end_lon, end_lat)

            # 6. Calculate shortest path and safety-optimized path
            path_shortest = nx.shortest_path(G, start_node, end_node, weight='length')
            path_safe = nx.shortest_path(G, start_node, end_node, weight='weight')

            def get_path_length(path_nodes):
                length_sum = 0.0
                for i in range(len(path_nodes) - 1):
                    u, v = path_nodes[i], path_nodes[i+1]
                    edge_data = G.get_edge_data(u, v)
                    if edge_data:
                        first_key = list(edge_data.keys())[0]
                        length_sum += edge_data[first_key].get('length', 0)
                return length_sum

            len_shortest = get_path_length(path_shortest)
            len_safe = get_path_length(path_safe)

            # 7. Apply Detour Constraint Guard
            if len_safe > 1.5 * len_shortest:
                # Recalculate with balanced BETA = 1.0
                path_balanced = nx.shortest_path(G, start_node, end_node, weight='weight_balanced')
                len_balanced = get_path_length(path_balanced)
                if len_balanced > 1.5 * len_shortest:
                    path = path_shortest
                    used_path_type = f"Shortest Path (Fallback due to excessive detour: {len_safe/len_shortest:.2f}x)"
                else:
                    path = path_balanced
                    used_path_type = f"Balanced Path (BETA=1.0 due to detour: {len_safe/len_shortest:.2f}x)"
            else:
                path = path_safe
                used_path_type = "Safety-Optimized Path (BETA=3.0)"

            distance_km = round(get_path_length(path) / 1000.0, 2)
            eta_min = int(round((distance_km / 5.0) * 60.0 + (len(path) * 0.15)))  # walking ETA

            route_cells_info = []
            for node in path:
                lat = G.nodes[node]['y']
                lon = G.nodes[node]['x']
                node_safety = 1.0 - node_risk[node]
                route_cells_info.append({
                    "node_id": int(node),
                    "lat": float(lat),
                    "lon": float(lon),
                    "safety_score": float(node_safety)
                })

            logging.info(f"Successfully computed path via OSMnx: {used_path_type}")
            return {
                "route_found": True,
                "path": path,
                "distance_km": distance_km,
                "eta_min": eta_min,
                "route_cells": route_cells_info,
                "path_type": used_path_type
            }

        except Exception as e:
            logging.warning(f"OSMnx road routing failed: {e}. Falling back to precalculated grid routing.")
            return self._find_grid_route_fallback(start_lat, start_lon, end_lat, end_lon)

    def _find_grid_route_fallback(self, start_lat, start_lon, end_lat, end_lon):
        """Precalculated grid pathfinder used as offline fallback."""
        start_point = gpd.GeoSeries([Point(start_lon, start_lat)], crs="EPSG:4326").to_crs(self.grid_gdf.crs)[0]
        end_point = gpd.GeoSeries([Point(end_lon, end_lat)], crs="EPSG:4326").to_crs(self.grid_gdf.crs)[0]

        def nearest_node(point):
            distances = self.grid_gdf.geometry.distance(point)
            return distances.idxmin()

        start_node = nearest_node(start_point)
        end_node = nearest_node(end_point)

        if not nx.has_path(self.G_grid, start_node, end_node):
            path = None
        else:
            path = nx.shortest_path(self.G_grid, source=start_node, target=end_node, weight="weight")

        distance_km = 0.0
        eta_min = 0.0
        route_cells_info = []

        grid_gdf_4326 = self.grid_gdf.to_crs(epsg=4326)

        if path is not None:
            for node in path:
                centroid = grid_gdf_4326.loc[node].geometry.centroid
                route_cells_info.append({
                    "node_id": int(node),
                    "lat": float(centroid.y),
                    "lon": float(centroid.x),
                    "safety_score": float(self.grid_gdf.loc[node].safety_score)
                })

            if len(path) > 1:
                proj_coords = [self.grid_gdf.loc[node].geometry.centroid for node in path]
                total_dist_meters = 0.0
                for i in range(len(proj_coords) - 1):
                    total_dist_meters += proj_coords[i].distance(proj_coords[i+1])
                distance_km = round(total_dist_meters / 1000.0, 2)

            eta_min = int(round((distance_km / 5.0) * 60.0 + (len(path) * 0.2)))

        return {
            "route_found": path is not None,
            "path": path,
            "distance_km": distance_km,
            "eta_min": eta_min,
            "route_cells": route_cells_info,
            "path_type": "Grid-Based Route (Offline Fallback)"
        }
