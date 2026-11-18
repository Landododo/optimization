"""Solve a closed loop on the cached Boston graph and plot it.

Usage: python scripts/solve_boston_loop.py [--region esplanade] [--km 5]
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from route_opt.boston import load_graph, nearest_node  # noqa: E402
from route_opt.config import RouteConfig  # noqa: E402
from route_opt.solve_milp import solve_loop  # noqa: E402
from route_opt.viz import plot_route  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--region", default="esplanade")
p.add_argument("--network", default="walk")
p.add_argument("--km", type=float, default=5.0, help="target loop length")
p.add_argument("--start-lon", type=float, default=-71.0775)   # near Arlington St
p.add_argument("--start-lat", type=float, default=42.3536)
p.add_argument("--time-limit", type=int, default=600, help="solver seconds")
p.add_argument("--length-penalty", type=float, default=10.0,
               help="cost per metre off target; use 0 for a hard window")
p.add_argument("--msg", action="store_true", help="show CBC log")
args = p.parse_args()

D = load_graph(ROOT / "data" / f"{args.region}_{args.network}.pkl")
start = nearest_node(D, args.start_lon, args.start_lat)
print(f"start node {start} at lon/lat {D.nodes[start]['lon']:.5f}, {D.nodes[start]['lat']:.5f}")

cfg = RouteConfig(
    target_length_m=args.km * 1000,
    length_tol_m=0.03 * args.km * 1000,   # +/- 3 percent
    turn_threshold_deg=30.0,
    turn_k=0.2,
    turn_convex=True,
    intersections_once=True,
)

lp = None if args.length_penalty == 0 else args.length_penalty
res = solve_loop(D, start, cfg, time_limit=args.time_limit,
                 length_penalty=lp, msg=args.msg)
s = res.stats
print(f"length  {s.length_m:8.0f} m")
print(f"ascent  {s.ascent_m:8.1f} m   descent {s.descent_m:8.1f} m")
print(f"turns penalised: {s.n_penalised_turns}   total cost {s.total_cost:.1f}")
print(f"target  {cfg.target_length_m:8.0f} m   off by {s.length_m - cfg.target_length_m:+.0f} m")

out = ROOT / "output" / f"{args.region}_loop.png"
out.parent.mkdir(exist_ok=True)
plot_route(D, res.route, path=out, title=f"{args.region} loop, {args.km:.1f} km")
print(f"saved {out}")
plt.close("all")