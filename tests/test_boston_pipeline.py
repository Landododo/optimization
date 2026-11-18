import networkx as nx
import numpy as np
import osmnx  # noqa: F401
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString

from route_opt.boston import nearest_node, prepare_graph, summarize


WEST, NORTH, RES = -71.10, 42.36, 0.0005   # 200 x 200 pixel DEM ~ 0.1 deg square


@pytest.fixture
def dem_path(tmp_path):
    """DEM that rises 500 m per degree of longitude, i.e. eastward."""
    H = W = 200
    lons = WEST + (np.arange(W) + 0.5) * RES
    z = (lons - WEST) * 500.0
    arr = np.tile(z, (H, 1)).astype("float32")
    path = tmp_path / "dem.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=H, width=W, count=1, dtype="float32",
        crs="EPSG:4326", transform=from_origin(WEST, NORTH, RES, RES),
    ) as dst:
        dst.write(arr, 1)
    return path


@pytest.fixture
def G_ll():
    """3 x 3 bidirected grid in lon/lat, ~170 m spacing. Node IDs are integers,
    as in real OSM data."""
    G = nx.MultiDiGraph(crs="EPSG:4326")
    step = 0.002
    idx = lambda i, j: i * 3 + j
    for i in range(3):
        for j in range(3):
            G.add_node(idx(i, j), x=-71.09 + i * step, y=42.35 + j * step)
    for i in range(3):
        for j in range(3):
            for di, dj in [(1, 0), (0, 1)]:
                if i + di < 3 and j + dj < 3:
                    a, b = idx(i, j), idx(i + di, j + dj)
                    for s, t in [(a, b), (b, a)]:
                        ln = LineString([(G.nodes[s]["x"], G.nodes[s]["y"]),
                                         (G.nodes[t]["x"], G.nodes[t]["y"])])
                        G.add_edge(s, t, geometry=ln, length=ln.length, highway="residential")
    return G


def test_graph_has_elevation_and_climb(G_ll, dem_path):
    D = prepare_graph(G_ll, dem_path, segment_m=25.0)
    assert D.number_of_nodes() == 9
    assert all("elev" in d for _, d in D.nodes(data=True))

    # Find an eastward edge (longitude increases, latitude unchanged) and its reverse
    east = next(
        (u, v) for u, v in D.edges
        if D.nodes[v]["lon"] > D.nodes[u]["lon"] + 1e-6
        and abs(D.nodes[v]["lat"] - D.nodes[u]["lat"]) < 1e-9
    )
    fwd = D.edges[east]
    rev = D.edges[(east[1], east[0])]
    assert fwd["ascent"] > 0.5 and fwd["descent"] == pytest.approx(0.0, abs=1e-6)
    assert rev["descent"] > 0.5 and rev["ascent"] == pytest.approx(0.0, abs=1e-6)


def test_summary_and_nearest_node(G_ll, dem_path):
    D = prepare_graph(G_ll, dem_path)
    s = summarize(D)
    assert s["nodes"] == 9 and s["total_length_km"] > 0
    n = nearest_node(D, -71.09, 42.35)
    assert (D.nodes[n]["lon"], D.nodes[n]["lat"]) == pytest.approx((-71.09, 42.35))