"""Small grid graphs for tests and demos."""
import math

import networkx as nx


def grid_graph(cols, rows, spacing=100.0, elev=lambda x, y: 0.0):
    """Bidirected grid. Nodes are (i, j). Attributes: x, y, elev. Edges: length."""
    G = nx.DiGraph()
    for i in range(cols):
        for j in range(rows):
            x, y = i * spacing, j * spacing
            G.add_node((i, j), x=x, y=y, elev=elev(x, y))
    for i in range(cols):
        for j in range(rows):
            for di, dj in [(1, 0), (0, 1)]:
                a, b = (i, j), (i + di, j + dj)
                if b in G:
                    L = math.dist((G.nodes[a]["x"], G.nodes[a]["y"]),
                                  (G.nodes[b]["x"], G.nodes[b]["y"]))
                    G.add_edge(a, b, length=L)
                    G.add_edge(b, a, length=L)
    return G


def hill_elev(x, y, peak=20.0, cx=250.0, cy=250.0, sigma=120.0):
    """Gaussian hill in the middle of the grid, for demos."""
    return peak * math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))