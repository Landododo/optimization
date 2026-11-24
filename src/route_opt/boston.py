"""Build a routable street graph for a region around Boston.

Pipeline: OSM streets -> largest strongly connected component -> UTM metres
-> DEM elevations at nodes and along densified street geometry
-> DiGraph with x, y, elev on nodes and length, ascent, descent on edges.
"""
from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import rasterio
import shapely
from pyproj import Transformer
from shapely.geometry import LineString, box
from route_opt.costs import heading_deg, turn_angle_deg
import time
import random
import requests

@dataclass(frozen=True)
class Region:
    west: float
    south: float
    east: float
    north: float

    @property
    def bbox(self):
        # OSMnx 2.x order: (left, bottom, right, top) = (west, south, east, north)
        return (self.west, self.south, self.east, self.north)


# Presets. Start with ESPLANADE; the wide box is for later, once solve times are known.
ESPLANADE = Region(west=-71.09, south=42.345, east=-71.06, north=42.362)
BOSTON_WIDE = Region(west=-71.12, south=42.34, east=-71.05, north=42.37)
MARATHON = Region(west=-71.16, south=42.31, east=-71.03, north=42.39)
# In src/route_opt/boston.py
from shapely.geometry import LineString, Point, box
from shapely.ops import substring


def insert_exact_node(
    D: nx.DiGraph,
    lon: float,
    lat: float,
    node_id: str = "boston_finish",
    crs_m: str = "EPSG:32619",  # UTM Zone 19N (Boston)
):
    """Project (lon, lat) onto the closest street edge in D, split that edge, and return node_id."""
    if node_id in D:
        return node_id

    to_utm = Transformer.from_crs("EPSG:4326", crs_m, always_xy=True)
    to_ll = Transformer.from_crs(crs_m, "EPSG:4326", always_xy=True)
    px, py = to_utm.transform(lon, lat)
    pt = Point(px, py)

    # Find the closest street segment in the graph
    best_edge, best_dist = None, float("inf")
    for u, v, d in D.edges(data=True):
        dist = d["geometry"].distance(pt)
        if dist < best_dist:
            best_dist = dist
            best_edge = (u, v)

    u, v = best_edge
    ls: LineString = D.edges[u, v]["geometry"]
    frac = ls.project(pt, normalized=True)

    # If already within 2% of an endpoint intersection, use that intersection directly
    if frac <= 0.02:
        return u
    if frac >= 0.98:
        return v

    # Compute exact snapped point on the street centerline
    ls1 = substring(ls, 0.0, frac, normalized=True)
    snap_x, snap_y = ls1.coords[-1]
    snap_lon, snap_lat = to_ll.transform(snap_x, snap_y)
    snap_elev = (1.0 - frac) * D.nodes[u]["elev"] + frac * D.nodes[v]["elev"]

    D.add_node(
        node_id,
        x=float(snap_x),
        y=float(snap_y),
        elev=float(snap_elev),
        lon=float(snap_lon),
        lat=float(snap_lat),
        is_intersection=False,
    )

    def _split_directed_edge(a, b, f):
        edge_data = dict(D.edges[a, b])
        edge_ls: LineString = edge_data["geometry"]
        seg1 = substring(edge_ls, 0.0, f, normalized=True)
        seg2 = substring(edge_ls, f, 1.0, normalized=True)

        in_b1, out_b1 = _edge_bearings(seg1)
        in_b2, out_b2 = _edge_bearings(seg2)

        D.remove_edge(a, b)
        D.add_edge(
            a,
            node_id,
            length=float(seg1.length),
            ascent=edge_data["ascent"] * f,
            descent=edge_data["descent"] * f,
            in_bearing=in_b1,
            out_bearing=out_b1,
            geometry=seg1,
        )
        D.add_edge(
            node_id,
            b,
            length=float(seg2.length),
            ascent=edge_data["ascent"] * (1.0 - f),
            descent=edge_data["descent"] * (1.0 - f),
            in_bearing=in_b2,
            out_bearing=out_b2,
            geometry=seg2,
        )

    # Split both directions if the street is two-way
    has_rev = D.has_edge(v, u)
    _split_directed_edge(u, v, frac)
    if has_rev:
        _split_directed_edge(v, u, 1.0 - frac)

    return node_id

def fetch_dem(region: Region, path, resolution_m: int = 10) -> Path:
    """Download a USGS 3DEP DEM for the region as a GeoTIFF (requires py3dep, rioxarray)."""
    import py3dep
    import rioxarray  # noqa: F401  (registers the .rio accessor)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    da = py3dep.get_dem(box(*region.bbox), resolution=resolution_m, crs="EPSG:4326")
    da.rio.to_raster(path)
    return path


def _smooth(z, window):
    """Moving average with edge padding. Reduces DEM noise before summing climbs."""
    if window <= 1 or len(z) < window:
        return z
    pad = window // 2
    zp = np.pad(z, pad, mode="edge")
    return np.convolve(zp, np.ones(window) / window, mode="valid")


def _sample_dem(dem_path, xy, crs_metric):
    """Sample DEM elevations at metric points xy (N x 2). Fills nodata gaps."""
    with rasterio.open(dem_path) as src:
        to_raster = Transformer.from_crs(crs_metric, src.crs, always_xy=True)
        xs, ys = to_raster.transform(xy[:, 0], xy[:, 1])
        vals = np.array([v[0] for v in src.sample(list(zip(xs, ys)))], dtype=float)
        nodata = src.nodata
    if nodata is not None:
        vals[vals == nodata] = np.nan
    bad = ~np.isfinite(vals)
    if bad.all():
        raise ValueError("DEM has no valid data under the street network")
    if bad.any():
        good = np.flatnonzero(~bad)
        vals[bad] = np.interp(np.flatnonzero(bad), good, vals[good])
    return vals

def _edge_bearings(ls: LineString) -> tuple[float, float]:
    """Exact departure bearing at start and arrival bearing at end of a LineString."""
    coords = list(ls.coords)
    in_b = heading_deg(coords[0], coords[1])
    out_b = heading_deg(coords[-2], coords[-1])
    return in_b, out_b


def _split_sharp_bends(G: nx.MultiDiGraph, bend_threshold_deg: float = 30.0) -> nx.MultiDiGraph:
    """Split MultiDiGraph edges at interior LineString vertices with turns > bend_threshold_deg."""
    H = nx.MultiDiGraph(crs=G.graph["crs"])
    for n, d in G.nodes(data=True):
        H.add_node(n, **d, is_intersection=True)

    next_bend_id = 0
    for u, v, k, d in G.edges(keys=True, data=True):
        ls: LineString = d["geometry"]
        coords = list(ls.coords)
        if len(coords) <= 2 or ls.length < 15.0:
            in_b, out_b = _edge_bearings(ls)
            H.add_edge(u, v, geometry=ls, length=ls.length, in_bearing=in_b, out_bearing=out_b, **sub_d)
            continue

        # Find interior indices where deflection angle exceeds threshold
        split_idxs = [0]
        for i in range(1, len(coords) - 1):
            b_in = heading_deg(coords[i - 1], coords[i])
            b_out = heading_deg(coords[i], coords[i + 1])
            if turn_angle_deg(b_in, b_out) > bend_threshold_deg:
                split_idxs.append(i)
        split_idxs.append(len(coords) - 1)

        prev_node = u
        for seg_i in range(len(split_idxs) - 1):
            i0, i1 = split_idxs[seg_i], split_idxs[seg_i + 1]
            sub_ls = LineString(coords[i0 : i1 + 1])
            if seg_i < len(split_idxs) - 2:
                curr_node = f"bend_{next_bend_id}"
                next_bend_id += 1
                bx, by = coords[i1]
                H.add_node(curr_node, x=bx, y=by, is_intersection=False)
            else:
                curr_node = v

            in_b, out_b = _edge_bearings(sub_ls)
            sub_d = {k_attr: v_attr for k_attr, v_attr in d.items() if k_attr not in ("geometry", "length")}
            H.add_edge(
                prev_node,
                curr_node,
                geometry=sub_ls,
                length=sub_ls.length,
                in_bearing=in_b,
                out_bearing=out_b,
                **sub_d,
            )
            prev_node = curr_node

    return H

def prepare_graph(
    G_ll,
    dem_path,
    segment_m: float = 25.0,
    smooth_window: int = 3,
    bend_threshold_deg: float = 30.0,
) -> nx.DiGraph:
    """Turn an OSMnx lon/lat MultiDiGraph into a solver-ready DiGraph."""
    G_ll = ox.truncate.largest_component(G_ll, strongly=True)
    G = ox.project_graph(G_ll)            # metres (UTM)
    crs_m = G.graph["crs"]

    # 1. Every edge needs geometry in metres (straight line if OSM gave none)
    for u, v, k, d in G.edges(keys=True, data=True):
        if "geometry" not in d:
            d["geometry"] = LineString([
                (G.nodes[u]["x"], G.nodes[u]["y"]),
                (G.nodes[v]["x"], G.nodes[v]["y"]),
            ])
        d["length"] = d["geometry"].length

    # 2. Split edges at sharp mid-street bends before elevation sampling
    G = _split_sharp_bends(G, bend_threshold_deg=bend_threshold_deg)

    # 3. Densify each sub-segment, then sample every point in one batch
    node_order = list(G.nodes)
    node_xy = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in node_order])
    dense, chunks = {}, [node_xy]
    for u, v, k, d in G.edges(keys=True, data=True):
        coords = np.asarray(shapely.segmentize(d["geometry"], segment_m).coords)
        dense[(u, v, k)] = coords
        chunks.append(coords)
    z_all = _sample_dem(dem_path, np.vstack(chunks), crs_m)

    node_z = dict(zip(node_order, z_all[: len(node_order)]))
    pos = len(node_order)
    climb = {}
    for key, coords in dense.items():
        m = len(coords)
        z = _smooth(z_all[pos : pos + m], smooth_window)
        pos += m
        dz = np.diff(z)
        climb[key] = (float(dz[dz > 0].sum()), float(-dz[dz < 0].sum()))

    # 4. Inverse-project UTM coordinates so synthetic bend nodes also get lon/lat
    to_ll = Transformer.from_crs(crs_m, "EPSG:4326", always_xy=True)
    node_lons, node_lats = to_ll.transform(node_xy[:, 0], node_xy[:, 1])

    # 5. Build the solver graph with bearings and intersection flags
    D = nx.DiGraph()
    for idx, n in enumerate(node_order):
        d = G.nodes[n]
        if n in G_ll.nodes:
            lon, lat = G_ll.nodes[n]["x"], G_ll.nodes[n]["y"]
        else:
            lon, lat = float(node_lons[idx]), float(node_lats[idx])
        D.add_node(
            n,
            x=d["x"],
            y=d["y"],
            elev=float(node_z[n]),
            lon=lon,
            lat=lat,
            is_intersection=d.get("is_intersection", True),
        )

    for u, v, k, d in G.edges(keys=True, data=True):
        if D.has_edge(u, v) and D[u][v]["length"] <= d["length"]:
            continue                       # keep the shorter parallel edge
        asc, desc = climb[(u, v, k)]
        D.add_edge(
            u,
            v,
            length=float(d["length"]),
            ascent=asc,
            descent=desc,
            in_bearing=float(d["in_bearing"]),
            out_bearing=float(d["out_bearing"]),
            geometry=d["geometry"],
        )
    return D

def _download_osm_with_backoff(
    bbox: tuple[float, float, float, float],
    network_type: str,
    max_retries: int = 6,
    base_delay_s: float = 5.0,
) -> nx.MultiDiGraph:
    """Download an OSMnx graph with exponential backoff on 502/504/429 server errors."""
    ox.settings.use_cache = True
    ox.settings.requests_timeout = 300  # allow up to 5 min for large bounding boxes

    for attempt in range(max_retries):
        try:
            return ox.graph_from_bbox(bbox=bbox, network_type=network_type, simplify=True)
        except (ox._errors.ResponseStatusCodeError, requests.exceptions.RequestException) as err:
            if attempt == max_retries - 1:
                raise
            # Exponential backoff: 5s, 10s, 20s, 40s, 80s... plus random jitter
            wait_s = base_delay_s * (2 ** attempt) + random.uniform(0.5, 2.5)
            print(
                f"[Overpass busy] Attempt {attempt + 1}/{max_retries} failed ({type(err).__name__}). "
                f"Retrying in {wait_s:.1f}s...",
                flush=True,
            )
            time.sleep(wait_s)


def build_region_graph(
    region: Region,
    dem_path,
    network_type: str = "walk",
    segment_m: float = 25.0,
    bend_threshold_deg: float = 30.0,
    raw_cache_path: Path | str | None = None,
) -> nx.DiGraph:
    """Download streets for the region (with retry & local caching) and build the solver graph."""
    if raw_cache_path is not None and Path(raw_cache_path).exists():
        print(f"Loading cached raw OSM graph from {raw_cache_path}...", flush=True)
        G_ll = ox.load_graphml(raw_cache_path)
    else:
        G_ll = _download_osm_with_backoff(region.bbox, network_type=network_type)
        if raw_cache_path is not None:
            Path(raw_cache_path).parent.mkdir(parents=True, exist_ok=True)
            ox.save_graphml(G_ll, raw_cache_path)
            print(f"Cached raw OSM graph to {raw_cache_path}", flush=True)

    return prepare_graph(
        G_ll, dem_path, segment_m=segment_m, bend_threshold_deg=bend_threshold_deg
    )

def nearest_node(D: nx.DiGraph, lon: float, lat: float):
    """Closest node to a lon/lat point (brute force, fine for a few thousand nodes)."""
    kx = math.cos(math.radians(lat))
    best, best_d2 = None, float("inf")
    for n, d in D.nodes(data=True):
        d2 = ((d["lon"] - lon) * kx) ** 2 + (d["lat"] - lat) ** 2
        if d2 < best_d2:
            best, best_d2 = n, d2
    return best


def summarize(D: nx.DiGraph) -> dict:
    elev = [d["elev"] for _, d in D.nodes(data=True)]
    return {
        "nodes": D.number_of_nodes(),
        "directed_edges": D.number_of_edges(),
        "total_length_km": sum(d["length"] for _, _, d in D.edges(data=True)) / 2000.0,
        "elev_min_m": min(elev),
        "elev_max_m": max(elev),
    }


def save_graph(D: nx.DiGraph, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(D, f)


def load_graph(path) -> nx.DiGraph:
    """Only load pickles you created yourself."""
    with open(path, "rb") as f:
        return pickle.load(f)