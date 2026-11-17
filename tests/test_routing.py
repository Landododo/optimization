import math

import networkx as nx
import pytest

from route_opt.costs import turn_angle_deg, turn_penalty
from route_opt.line_graph import shortest_route


def make_graph(nodes, two_way_edges):
    """nodes: {name: (x, y, elev)}; two_way_edges: [(u, v), ...]."""
    G = nx.DiGraph()
    for n, (x, y, e) in nodes.items():
        G.add_node(n, x=x, y=y, elev=e)
    for u, v in two_way_edges:
        for a, b in [(u, v), (v, u)]:
            length = math.dist(
                (G.nodes[a]["x"], G.nodes[a]["y"]), (G.nodes[b]["x"], G.nodes[b]["y"])
            )
            G.add_edge(a, b, length=length)
    return G


def test_turn_angle_wraps():
    assert turn_angle_deg(350, 10) == pytest.approx(20)


def test_turn_penalty_zero_below_threshold_and_step_variant():
    assert turn_penalty(29, threshold=30) == 0.0
    assert turn_penalty(45, threshold=30, k=2.0, convex=False) == 2.0


def test_prefers_flat_route_over_hill_of_equal_length():
    # Hilly route S-H-E climbs 30 m. Flat route S-K-E has the same length.
    G = make_graph(
        {"S": (0, 0, 0), "H": (100, 0, 30), "K": (100, 0, 0), "E": (200, 0, 0)},
        [("S", "H"), ("H", "E"), ("S", "K"), ("K", "E")],
    )
    path, _ = shortest_route(G, "S", "E")
    assert path == ["S", "K", "E"]


def test_prefers_straight_route_over_sharp_turn():
    # Via A: two 90-degree turns' worth of penalty. Via C: straight diagonal, no turns.
    G = make_graph(
        {
            "S": (0, 0, 0),
            "A": (0, 100, 0),
            "B": (100, 0, 0),
            "C": (50, 50, 0),
            "E": (100, 100, 0),
        },
        [("S", "A"), ("A", "E"), ("S", "B"), ("B", "E"), ("S", "C"), ("C", "E")],
    )
    path, _ = shortest_route(G, "S", "E")
    assert path == ["S", "C", "E"]