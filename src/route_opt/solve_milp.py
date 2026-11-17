"""Closed-loop route via MILP (PuLP 4.x, COIN-OR CBC solver).

Decision variables
  x[e]  binary: directed street e=(u,v) is traversed
  y[v]  binary: intersection v is visited
  f[e]  continuous: single-commodity flow, enforces a connected loop
  z     continuous in [0,1]: turn (u,v,w) is used, linearised as z >= x_uv + x_vw - 1
"""
from dataclasses import dataclass

import networkx as nx
import pulp

from route_opt.config import RouteConfig
from route_opt.evaluate import route_stats


@dataclass
class RouteResult:
    route: list          # closed node sequence, route[0] == route[-1] == start
    objective: float
    stats: object        # RouteStats


def solve_loop(G, start, cfg: RouteConfig, time_limit=None):
    if start not in G:
        raise ValueError(f"start node {start!r} not in graph")

    nodes = list(G.nodes)
    edges = list(G.edges)
    N = len(nodes)

    prob = pulp.LpProblem("marathon_loop", pulp.LpMinimize)

    # PuLP 4.x: variables are created by the problem, not by LpVariable(...)
    x = {e: prob.add_variable(f"x_{i}", cat="Binary") for i, e in enumerate(edges)}
    f = {e: prob.add_variable(f"f_{i}", lowBound=0) for i, e in enumerate(edges)}
    y = {v: prob.add_variable(f"y_{i}", cat="Binary") for i, v in enumerate(nodes)}

    # Turn variables and objective terms
    turn_terms = []
    n_z = 0
    for v in nodes:
        for u in G.predecessors(v):
            for w in G.successors(v):
                if w == u:          # no U-turns
                    continue
                pen = cfg.turn_cost(G, u, v, w)
                if pen <= 0:
                    continue
                z = prob.add_variable(f"z_{n_z}", lowBound=0, upBound=1)
                n_z += 1
                prob += z >= x[(u, v)] + x[(v, w)] - 1
                turn_terms.append(pen * z)

    # Flow balance, loop closure, connectivity
    for v in nodes:
        in_e = list(G.in_edges(v))
        out_e = list(G.out_edges(v))
        x_in = pulp.lpSum(x[e] for e in in_e)
        x_out = pulp.lpSum(x[e] for e in out_e)
        prob += x_in == x_out

        if v == start:
            prob += x_in == 1          # leave and return exactly once
            prob += y[v] == 1
        else:
            if cfg.intersections_once:
                prob += x_in <= 1
            prob += x_in <= len(in_e) * y[v]
            prob += y[v] <= x_in
            prob += (pulp.lpSum(f[e] for e in in_e)
                     - pulp.lpSum(f[e] for e in out_e) == y[v])

    for e in edges:
        prob += f[e] <= N * x[e]

    # Each undirected street at most once (both directions counted together)
    by_pair = {}
    for e in edges:
        by_pair.setdefault(frozenset(e), []).append(e)
    for es in by_pair.values():
        if len(es) > 1:
            prob += pulp.lpSum(x[e] for e in es) <= 1

    # Length window
    total_len = pulp.lpSum(G.edges[e]["length"] * x[e] for e in edges)
    prob += total_len >= cfg.target_length_m - cfg.length_tol_m
    prob += total_len <= cfg.target_length_m + cfg.length_tol_m

    # Objective
    street_terms = [cfg.street_cost(G, *e) * x[e] for e in edges]
    obj = pulp.lpSum(street_terms) + pulp.lpSum(turn_terms)
    prob += obj

    # PuLP 4.x: COIN_CMD replaces PULP_CBC_CMD; solve() returns LpSolveStats
    stats = prob.solve(pulp.COIN_CMD(msg=False, timeLimit=time_limit))
    if not stats.has_solution:
        raise RuntimeError(f"solver found no solution: {stats}")

    chosen = [e for e in edges if pulp.value(x[e]) > 0.5]
    H = nx.DiGraph()
    H.add_edges_from(chosen)
    circuit = list(nx.eulerian_circuit(H, source=start))
    route = [start] + [v for _, v in circuit]

    return RouteResult(
        route=route,
        objective=pulp.value(obj),
        stats=route_stats(G, route, cfg),
    )