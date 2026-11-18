"""Generate a small closed walk quickly (heuristic, no MILP).

Usage:
  python scripts/quick_walk.py --region small --km 1.5
"""
import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from route_opt.boston import load_graph, nearest_node  # noqa: E402
from route_opt.config import RouteConfig  # noqa: E402
from route_opt.evaluate import route_stats  # noqa: E402
from route_opt.viz import plot_route  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--region", default="small")
p.add_argument("--network", default="walk")
p.add_argument("--km", type=float, default=1.5)
p.add_argument("--start-lon", type=float, default=-71.0755)
p.add_argument("--start-lat", type=float, default=42.3560)
p.add_argument("--candidates", type=int, default=150)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

D = load_graph(ROOT / "data" / f"{args.region}_{args.network}.pkl")
start = nearest_node(D, args.start_lon, args.start_lat)
cfg = RouteConfig(target_length_m=args.km * 1000)
target = cfg.target_length_m


def cost(u, v, d):
    return cfg.street_cost(D, u, v)


def loop_via(mid):
    """Out to mid on the cheapest path, back on the cheapest path that reuses no street."""
    try:
        out = nx.shortest_path(D, start, mid, weight=cost)
        H = D.copy()
        for a, b in zip(out[:-1], out[1:]):
            H.remove_edge(a, b)
            if H.has_edge(b, a):
                H.remove_edge(b, a)
        back = nx.shortest_path(H, mid, start, weight=cost)
    except nx.NetworkXNoPath:
        return None
    route = out + back[1:]
    if len(route) < 4 or route[0] != start or route[-1] != start:
        return None
    return route


# Candidate midpoints: nodes roughly half the target away from start
dist = nx.single_source_dijkstra_path_length(D, start, weight="length", cutoff=target)
pool = [n for n, d in dist.items() if 0.3 * target <= d <= 0.7 * target]
if not pool:
    raise SystemExit("no nodes at a suitable distance; try a longer --km or a bigger region")
random.seed(args.seed)
random.shuffle(pool)

best, best_err = None, float("inf")
for mid in pool[: args.candidates]:
    route = loop_via(mid)
    if route is None:
        continue
    err = abs(route_stats(D, route, cfg).length_m - target)
    if err < best_err:
        best, best_err = route, err

if best is None:
    raise SystemExit("no closed loop found; try another start point or a larger region")

s = route_stats(D, best, cfg)
print(f"length  {s.length_m:8.0f} m   (target {target:.0f}, off by {s.length_m - target:+.0f} m)")
print(f"ascent  {s.ascent_m:8.1f} m   descent {s.descent_m:8.1f} m")
print(f"turns penalised: {s.n_penalised_turns}")

out = ROOT / "output" / f"{args.region}_quick_{args.km:.1f}km.png"
out.parent.mkdir(exist_ok=True)
plot_route(D, best, path=out, title=f"{args.region} quick loop, {args.km:.1f} km")
print(f"saved {out}")
plt.close("all")