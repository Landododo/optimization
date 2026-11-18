"""Closed-loop route MILP (PuLP 4.x, COIN-OR CBC).

Variables
  x[e]    binary: directed street e=(u,v) is traversed
  z       continuous in [0,1]: linearised turn penalty, z >= x_uv + x_vw - 1
  dp, dm  continuous >= 0: length overshoot / undershoot (soft length mode)

Length: by default a penalty on |length - target| (always feasible, so CBC
finds an incumbent quickly). Pass length_penalty=None for a hard window.

Connectivity: solved lazily. If the chosen streets contain a cycle that does
not pass through start, add a subtour cut and re-solve.
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


def prune_ellipse(G, start, max_len):
    """Keep only edges that could lie on a closed loop of length <= max_len through start."""
    ds = nx.single_source_dijkstra_path_length(G, start, weight="length", cutoff=max_len)
    dt = nx.single_source_dijkstra_path_length(
        G.reverse(copy=False), start, weight="length", cutoff=max_len
    )
    keep = [
        (u, v) for u, v, d in G.edges(data=True)
        if u in ds and v in dt and ds[u] + d["length"] + dt[v] <= max_len
    ]
    return G.edge_subgraph(keep).copy()


def solve_loop(G, start, cfg: RouteConfig, time_limit=None, gap_rel=0.05,
               prune=True, max_cut_rounds=200, length_penalty=10.0, msg=False):
    """
    length_penalty: cost per metre of deviation from target (soft mode).
                    Keep it well above street_cost's per-metre rate (gamma + climb terms),
                    or the solver will prefer shorter loops.
                    None = hard window of target +/- length_tol_m.
    msg: True prints CBC's log, useful for seeing whether it finds solutions.
    """
    if prune:
        G = prune_ellipse(G, start, cfg.target_length_m * 1.1)
    if start not in G:
        raise ValueError("start has no usable streets within the length budget")

    nodes = list(G.nodes)
    edges = list(G.edges)

    prob = pulp.LpProblem("marathon_loop", pulp.LpMinimize)
    x = {e: prob.add_variable(f"x_{i}", cat="Binary") for i, e in enumerate(edges)}

    # Degree constraints. Inflow at a node is also "node visited".
    x_in = {}
    for v in nodes:
        in_e, out_e = list(G.in_edges(v)), list(G.out_edges(v))
        if not in_e or not out_e:
            continue
        xi = pulp.lpSum(x[e] for e in in_e)
        x_in[v] = xi
        prob += xi == pulp.lpSum(x[e] for e in out_e)
        if v == start:
            prob += xi == 1                    # leave and return exactly once
        elif cfg.intersections_once:
            prob += xi <= 1

    # Turn penalties (only turns above threshold get a variable)
    turn_terms, k = [], 0
    for v in nodes:
        for u in G.predecessors(v):
            for w in G.successors(v):
                if w == u:                     # no U-turns
                    continue
                pen = cfg.turn_cost(G, u, v, w)
                if pen <= 0:
                    continue
                z = prob.add_variable(f"z_{k}", lowBound=0, upBound=1)
                k += 1
                prob += z >= x[(u, v)] + x[(v, w)] - 1
                turn_terms.append(pen * z)

    # Each undirected street at most once (both directions together)
    by_pair = {}
    for e in edges:
        by_pair.setdefault(frozenset(e), []).append(e)
    for es in by_pair.values():
        if len(es) > 1:
            prob += pulp.lpSum(x[e] for e in es) <= 1

    # Length
    total_len = pulp.lpSum(G.edges[e]["length"] * x[e] for e in edges)
    if length_penalty is None:
        prob += total_len >= cfg.target_length_m - cfg.length_tol_m
        prob += total_len <= cfg.target_length_m + cfg.length_tol_m
        length_terms = []
    else:
        dp = prob.add_variable("len_over", lowBound=0)
        dm = prob.add_variable("len_under", lowBound=0)
        prob += total_len - dp + dm == cfg.target_length_m
        length_terms = [length_penalty * (dp + dm)]

    # Objective
    street_terms = [cfg.street_cost(G, *e) * x[e] for e in edges]
    obj = pulp.lpSum(street_terms) + pulp.lpSum(turn_terms) + pulp.lpSum(length_terms)
    prob += obj

    # Solve, then cut any cycle that does not pass through start
    for _ in range(max_cut_rounds):
        stats = prob.solve(pulp.COIN_CMD(msg=msg, timeLimit=time_limit, gapRel=gap_rel))
        if not stats.has_solution:
            raise RuntimeError(f"solver found no solution: {stats}")

        chosen = [e for e in edges if pulp.value(x[e]) > 0.5]
        H = nx.DiGraph()
        H.add_edges_from(chosen)
        bad = [c for c in nx.weakly_connected_components(H) if start not in c]
        if not bad:
            break
        for comp in bad:
            leaving = [x[(a, b)] for a in comp for b in G.successors(a) if b not in comp]
            v0 = next(iter(comp))
            # If any street inside this cycle is used, a street must leave the set
            prob += pulp.lpSum(leaving) >= x_in[v0]
    else:
        raise RuntimeError("no single loop found within max_cut_rounds")

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