"""Line graph construction and a Dijkstra baseline."""
import networkx as nx

from route_opt.costs import heading_deg, turn_angle_deg, turn_penalty, street_cost

START, END = ("__start__",), ("__end__",)


def build_line_graph(G, turn_kwargs=None, street_kwargs=None):
    """
    G: nx.DiGraph with node attrs x, y, elev and edge attr length.
    Line graph nodes are directed streets (u, v). Arc (i, j) exists for
    i = (u, v), j = (v, w) with w != u (no U-turns). Arc weight is the
    turn penalty plus the cost of the street being entered.
    """
    turn_kwargs = turn_kwargs or {}
    street_kwargs = street_kwargs or {}

    def pos(n):
        return (G.nodes[n]["x"], G.nodes[n]["y"])

    L = nx.DiGraph()
    for u, v in G.edges:
        cost = street_cost(
            G.nodes[u]["elev"], G.nodes[v]["elev"], G.edges[u, v]["length"], **street_kwargs
        )
        L.add_node((u, v), cost=cost)

    for v in G.nodes:
        for u in G.predecessors(v):
            for w in G.successors(v):
                if w == u:
                    continue
                theta = turn_angle_deg(heading_deg(pos(u), pos(v)), heading_deg(pos(v), pos(w)))
                pen = turn_penalty(theta, **turn_kwargs)
                L.add_edge((u, v), (v, w), turn=pen, weight=pen + L.nodes[(v, w)]["cost"])
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