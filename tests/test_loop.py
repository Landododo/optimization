import pytest

from route_opt.config import RouteConfig
from route_opt.evaluate import route_stats
from route_opt.solve_milp import solve_loop
from route_opt.toy import grid_graph, hill_elev
from route_opt.viz import plot_route


def cfg(**kw):
    base = dict(target_length_m=1200.0, length_tol_m=50.0)
    base.update(kw)
    return RouteConfig(**base)


@pytest.fixture
def G():
    return grid_graph(4, 4, spacing=100.0, elev=hill_elev)


START = (0, 0)


def test_route_is_closed_loop(G):
    res = solve_loop(G, START, cfg())
    assert res.route[0] == START
    assert res.route[-1] == START


def test_streets_used_at_most_once(G):
    res = solve_loop(G, START, cfg())
    directed = list(zip(res.route[:-1], res.route[1:]))
    assert len(directed) == len(set(directed))
    undirected = [frozenset(e) for e in directed]
    assert len(undirected) == len(set(undirected))


def test_intersections_once_when_enabled(G):
    res = solve_loop(G, START, cfg(intersections_once=True))
    visits = res.route[:-1]  # last node is the repeated start
    assert len(visits) == len(set(visits))


def test_length_within_bounds(G):
    c = cfg()
    res = solve_loop(G, START, c)
    assert abs(res.stats.length_m - c.target_length_m) <= c.length_tol_m


def test_objective_matches_independent_evaluation(G):
    c = cfg(intersections_once=True)
    res = solve_loop(G, START, c)
    assert route_stats(G, res.route, c).total_cost == pytest.approx(res.objective, rel=1e-6)


def test_turn_threshold_is_tunable(G):
    # Any closed route on a grid has 90-degree turns, so a 30-degree threshold penalises them.
    c_low = cfg(intersections_once=True, turn_threshold_deg=30.0, turn_k=1.0)
    res = solve_loop(G, START, c_low)
    assert route_stats(G, res.route, c_low).n_penalised_turns > 0

    # Raising the threshold above 90 degrees removes all penalties for the same route.
    c_high = cfg(intersections_once=True, turn_threshold_deg=100.0, turn_k=1.0)
    assert route_stats(G, res.route, c_high).turn_penalty == 0.0


def test_multi_visit_mode_still_gives_valid_loop(G):
    res = solve_loop(G, START, cfg(intersections_once=False))
    assert res.route[0] == res.route[-1] == START
    directed = list(zip(res.route[:-1], res.route[1:]))
    assert len(directed) == len(set(directed))


def test_plot_writes_png(G, tmp_path):
    res = solve_loop(G, START, cfg(intersections_once=True))
    out = tmp_path / "route.png"
    fig = plot_route(G, res.route, path=out, title="test")
    assert out.exists() and out.stat().st_size > 0