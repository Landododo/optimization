"""Closed-loop route MILP (PuLP 4.x, COIN-OR CBC).

Variables
  x[e]    binary: directed street e=(u,v) is traversed
  z       continuous in [0,1]: linearised turn penalty, z >= x_uv + x_vw - 1
  dp, dm  continuous >= 0: length overshoot / undershoot (soft length mode)

If initial_route is given, its streets are passed to CBC as a starting solution.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import logging
import threading
import time

import networkx as nx
import pulp

from route_opt.config import RouteConfig
from route_opt.evaluate import route_stats

log = logging.getLogger("route_opt")


@contextmanager
def _heartbeat(label: str, interval_s: float = 15.0):
    """Print periodic status updates while a blocking C++ solver call runs."""
    stop_event = threading.Event()
    t0 = time.perf_counter()

    def _worker():
        while not stop_event.wait(interval_s):
            log.info(
                "[heartbeat] %s still running (%.0fs elapsed)...",
                label,
                time.perf_counter() - t0,
            )

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=1.0)


@dataclass
class RouteResult:
    route: list
    objective: float
    stats: object


def prune_ellipse(G, start, max_len):
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
               prune=True, max_cut_rounds=200, length_penalty=10.0, msg=False,
               initial_route=None):
    t_start = time.perf_counter()
    log.info(
        "Starting solve_loop: start=%s, target=%.1fm (tol=%.1fm), intersections_once=%s, "
        "length_penalty=%s, time_limit=%s, gap_rel=%.3f, warm_start=%s",
        start,
        cfg.target_length_m,
        cfg.length_tol_m,
        cfg.intersections_once,
        length_penalty,
        time_limit,
        gap_rel,
        initial_route is not None,
    )

    if initial_route is not None and not all(
        G.has_edge(a, b) for a, b in zip(initial_route[:-1], initial_route[1:])
    ):
        raise ValueError("initial_route uses streets that are not in G (pruned or different graph)")

    if prune:
        t0 = time.perf_counter()
        n_before, e_before = G.number_of_nodes(), G.number_of_edges()
        G = prune_ellipse(G, start, cfg.target_length_m * 1.1)
        log.info(
            "Pruned graph: %d -> %d nodes, %d -> %d edges (%.2fs)",
            n_before,
            G.number_of_nodes(),
            e_before,
            G.number_of_edges(),
            time.perf_counter() - t0,
        )
    if start not in G:
        raise ValueError("start has no usable streets within the length budget")

    t0 = time.perf_counter()
    nodes = list(G.nodes)
    edges = list(G.edges)

    prob = pulp.LpProblem("marathon_loop", pulp.LpMinimize)
    x = {e: prob.add_variable(f"x_{i}", cat="Binary") for i, e in enumerate(edges)}

    x_in = {}
    for v in nodes:
        in_e, out_e = list(G.in_edges(v)), list(G.out_edges(v))
        if not in_e or not out_e:
            continue
        xi = pulp.lpSum(x[e] for e in in_e)
        x_in[v] = xi
        prob += xi == pulp.lpSum(x[e] for e in out_e)
        if v == start:
            prob += xi == 1
        elif cfg.intersections_once:
            prob += xi <= 1

    turn_terms, turn_vars, k = [], [], 0
    for v in nodes:
        for u in G.predecessors(v):
            for w in G.successors(v):
                if w == u:
                    continue
                pen = cfg.turn_cost(G, u, v, w)
                if pen <= 0:
                    continue
                z = prob.add_variable(f"z_{k}", lowBound=0, upBound=1)
                k += 1
                prob += z >= x[(u, v)] + x[(v, w)] - 1
                turn_terms.append(pen * z)
                turn_vars.append((z, (u, v), (v, w)))

    by_pair = {}
    for e in edges:
        by_pair.setdefault(frozenset(e), []).append(e)
    for es in by_pair.values():
        if len(es) > 1:
            prob += pulp.lpSum(x[e] for e in es) <= 1

    total_len = pulp.lpSum(G.edges[e]["length"] * x[e] for e in edges)
    if length_penalty is None:
        prob += total_len >= cfg.target_length_m - cfg.length_tol_m
        prob += total_len <= cfg.target_length_m + cfg.length_tol_m
        dp = dm = None
        length_terms = []
    else:
        dp = prob.add_variable("len_over", lowBound=0)
        dm = prob.add_variable("len_under", lowBound=0)
        prob += total_len - dp + dm == cfg.target_length_m
        length_terms = [length_penalty * (dp + dm)]

    street_terms = [cfg.street_cost(G, *e) * x[e] for e in edges]
    obj = pulp.lpSum(street_terms) + pulp.lpSum(turn_terms) + pulp.lpSum(length_terms)
    prob += obj

    log.info(
        "Built MILP: %d street vars, %d turn vars, %d total vars, %d constraints (%.2fs)",
        len(x),
        len(turn_vars),
        prob.numVariables(),
        prob.numConstraints(),
        time.perf_counter() - t0,
    )

    # Warm start: give CBC the quick route as a starting solution
    warm = initial_route is not None
    if warm:
        used = set(zip(initial_route[:-1], initial_route[1:]))
        for e in edges:
            x[e].setInitialValue(1.0 if e in used else 0.0)
        for z, a, b in turn_vars:
            z.setInitialValue(1.0 if (a in used and b in used) else 0.0)
        init_len = sum(G.edges[e]["length"] for e in used)
        if dp is not None:
            dp.setInitialValue(max(0.0, init_len - cfg.target_length_m))
            dm.setInitialValue(max(0.0, cfg.target_length_m - init_len))
        log.info(
            "Applied warm-start route: %d streets, length=%.1fm",
            len(used),
            init_len,
        )

    total_cuts = 0
    for cut_round in range(1, max_cut_rounds + 1):
        t_round = time.perf_counter()
        log.info("Cut round %d/%d: launching CBC solver...", cut_round, max_cut_rounds)

        solver = pulp.HiGHS(msg=msg, timeLimit=time_limit, gapRel=0.99)
        with _heartbeat(f"Cut round {cut_round}"):
            stats = prob.solve(solver)

        if not stats.has_solution:
            raise RuntimeError(f"solver found no solution: {stats}")

        chosen = [e for e in edges if pulp.value(x[e]) > 0.5]
        H = nx.DiGraph()
        H.add_edges_from(chosen)
        bad = [c for c in nx.weakly_connected_components(H) if start not in c]

        if not bad:
            log.info(
                "Cut round %d finished in %.1fs: valid single loop found! "
                "(edges=%d, obj=%.2f, total_time=%.1fs)",
                cut_round,
                time.perf_counter() - t_round,
                len(chosen),
                pulp.value(obj),
                time.perf_counter() - t_start,
            )
            break

        round_cuts = 0
        for comp in bad:
            leaving = [x[(a, b)] for a in comp for b in G.successors(a) if b not in comp]
            v0 = next(iter(comp))
            prob += pulp.lpSum(leaving) >= x_in[v0]
            round_cuts += 1
        total_cuts += round_cuts
        log.info(
            "Cut round %d finished in %.1fs: obj=%.2f, %d disconnected cycle(s) "
            "(sizes=%s), added %d subtour cut(s) (total_cuts=%d)",
            cut_round,
            time.perf_counter() - t_round,
            pulp.value(obj),
            len(bad),
            [len(c) for c in bad],
            round_cuts,
            total_cuts,
        )
    else:
        raise RuntimeError("no single loop found within max_cut_rounds")

    circuit = list(nx.eulerian_circuit(H, source=start))
    route = [start] + [v for _, v in circuit]
    res_stats = route_stats(G, route, cfg)
    log.info(
        "Completed in %.1fs: %d nodes, length=%.1fm, ascent=%.1fm, descent=%.1fm, "
        "penalised_turns=%d, total_cost=%.2f",
        time.perf_counter() - t_start,
        len(route),
        res_stats.length_m,
        res_stats.ascent_m,
        res_stats.descent_m,
        res_stats.n_penalised_turns,
        res_stats.total_cost,
    )

    return RouteResult(
        route=route,
        objective=pulp.value(obj),
        stats=res_stats,
    )