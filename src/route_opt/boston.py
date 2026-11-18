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


def prepare_graph(G_ll, dem_path, segment_m: float = 25.0, smooth_window: int = 3) -> nx.DiGraph:
    """Turn an OSMnx lon/lat MultiDiGraph into a solver-ready DiGraph."""
    G_ll = ox.truncate.largest_component(G_ll, strongly=True)
    G = ox.project_graph(G_ll)            # metres (UTM)
    crs_m = G.graph["crs"]

    # Every edge needs geometry in metres (straight line if OSM gave none)
    for u, v, k, d in G.edges(keys=True, data=True):
        if "geometry" not in d:
            d["geometry"] = LineString([
                (G.nodes[u]["x"], G.nodes[u]["y"]),
                (G.nodes[v]["x"], G.nodes[v]["y"]),
            ])
        d["length"] = d["geometry"].length

    # Densify each edge, then sample every point in one batch
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
        z = _smooth(z_all[pos: pos + m], smooth_window)
        pos += m
        dz = np.diff(z)
        climb[key] = (float(dz[dz > 0].sum()), float(-dz[dz < 0].sum()))

    # Build the solver graph
    D = nx.DiGraph()
    for n in node_order:
        d = G.nodes[n]
        D.add_node(
            n, x=d["x"], y=d["y"], elev=float(node_z[n]),
            lon=G_ll.nodes[n]["x"], lat=G_ll.nodes[n]["y"],
        )
    for u, v, k, d in G.edges(keys=True, data=True):
        if D.has_edge(u, v) and D[u][v]["length"] <= d["length"]:
            continue                       # keep the shorter parallel edge
        asc, desc = climb[(u, v, k)]
        D.add_edge(u, v, length=float(d["length"]), ascent=asc, descent=desc,
                   geometry=d["geometry"])
    return D


def build_region_graph(region: Region, dem_path, network_type: str = "walk",
                       segment_m: float = 25.0) -> nx.DiGraph:
    """Download streets for the region and build the solver graph."""
    G_ll = ox.graph_from_bbox(bbox=region.bbox, network_type=network_type, simplify=True)
    return prepare_graph(G_ll, dem_path, segment_m=segment_m)


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