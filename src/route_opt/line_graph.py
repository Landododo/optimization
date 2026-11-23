"""Line graph construction and a Dijkstra baseline."""
import networkx as nx

from route_opt.costs import heading_deg, turn_angle_deg, turn_penalty, street_cost
import networkx as nx
from route_opt.config import RouteConfig

START, END = ("__start__",), ("__end__",)


def build_line_graph(G: nx.DiGraph, cfg: RouteConfig) -> nx.DiGraph:
    """
    Nodes in L are directed street segments e = (u, v) from G.
    Directed arcs ((u, v), (v, w)) represent valid transitions at node v (w != u).
    """
    L = nx.DiGraph()
    for u, v, d in G.edges(data=True):
        L.add_node(
            (u, v),
            cost=cfg.street_cost(G, u, v),
            length=float(d["length"]),
        )

    for v in G.nodes:
        for u in G.predecessors(v):
            for w in G.successors(v):
                if w == u:
                    continue
                pen = cfg.turn_cost(G, u, v, w)
                L.add_edge(
                    (u, v),
                    (v, w),
                    turn=pen,
                    weight=pen + L.nodes[(v, w)]["cost"],
                )
    return L


def shortest_route(G, start, end, turn_kwargs=None, street_kwargs=None):
    """Point-to-point route. Returns (list of nodes, total cost)."""
    L = build_line_graph(G, turn_kwargs, street_kwargs)

    # Dummy start and end nodes connect to the streets leaving start and entering end.
    for w in G.successors(start):
        L.add_edge(START, (start, w), weight=L.nodes[(start, w)]["cost"])
    for u in G.predecessors(end):
        L.add_edge((u, end), END, weight=0.0)

    cost = nx.shortest_path_length(L, START, END, weight="weight")
    streets = nx.shortest_path(L, START, END, weight="weight")[1:-1]
    path = [start] + [v for _, v in streets]
    return path, cost