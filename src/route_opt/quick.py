"""Heuristic loop generator used to seed the MILP. Not optimal, but fast and feasible."""
import random

import networkx as nx

from route_opt.evaluate import route_stats


def _loop_via(D, start, mid, cfg):
    """Out to mid, then back without reusing any street or intermediate intersection."""
    def cost(u, v, d):
        return cfg.street_cost(D, u, v)

    try:
        out = nx.shortest_path(D, start, mid, weight=cost)
        H = D.copy()
        # Block the outbound path's streets and intermediate nodes.
        H.remove_nodes_from(out[1:-1])
        for a, b in zip(out[:-1], out[1:]):
            if H.has_edge(a, b):
                H.remove_edge(a, b)
            if H.has_edge(b, a):
                H.remove_edge(b, a)
        back = nx.shortest_path(H, mid, start, weight=cost)
    except nx.NetworkXNoPath:
        return None

    route = out + back[1:]
    if len(route) < 4 or route[0] != start or route[-1] != start:
        return None
    if len(set(route[:-1])) != len(route) - 1:       # every intersection at most once
        return None
    return route


def quick_loop(D, start, cfg, candidates=150, seed=0):
    """Best closed loop found by trying random midpoints. Returns a route or None."""
    target = cfg.target_length_m
    dist = nx.single_source_dijkstra_path_length(D, start, weight="length", cutoff=target)
    pool = [n for n, d in dist.items() if 0.3 * target <= d <= 0.7 * target and n != start]
    if not pool:
        return None

    rng = random.Random(seed)
    rng.shuffle(pool)

    best, best_err = None, float("inf")
    for mid in pool[:candidates]:
        route = _loop_via(D, start, mid, cfg)
        if route is None:
            continue
        err = abs(route_stats(D, route, cfg).length_m - target)
        if err < best_err:
            best, best_err = route, err
    return best