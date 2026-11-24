"""Solve a closed loop on the cached Boston graph and plot it.

Usage: python -u scripts/solve_boston_loop.py [--region marathon] [--km 42.195]
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from route_opt.boston import insert_exact_node, load_graph  # noqa: E402
from route_opt.config import RouteConfig  # noqa: E402
from route_opt.solve_milp import solve_loop  # noqa: E402
from route_opt.viz import plot_route  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger("route_opt")

p = argparse.ArgumentParser()
p.add_argument("--region", default="marathon")
p.add_argument("--network", default="walk")
p.add_argument("--km", type=float, default=42.195, help="target loop length")
p.add_argument("--start-lon", type=float, default=-71.07847, help="Boston Marathon finish line lon")
p.add_argument("--start-lat", type=float, default=42.34962, help="Boston Marathon finish line lat")
p.add_argument("--time-limit", type=int, default=7200, help="solver seconds (default 2 hours)")
p.add_argument("--length-penalty", type=float, default=0.0,
               help="cost per metre off target; 0 for a hard window")
p.add_argument("--quiet", action="store_true", help="hide live HiGHS solver log")
args = p.parse_args()

t0 = time.perf_counter()
pkl_path = ROOT / "data" / f"{args.region}_{args.network}.pkl"
log.info("Loading cached graph from %s ...", pkl_path)
D = load_graph(pkl_path)
log.info("Loaded graph (%d nodes, %d edges) in %.2fs",
         D.number_of_nodes(), D.number_of_edges(), time.perf_counter() - t0)

start = insert_exact_node(D, args.start_lon, args.start_lat, node_id="boston_finish")
log.info("Start/finish node %s at (%.5f, %.5f)", start, D.nodes[start]["lon"], D.nodes[start]["lat"])

cfg = RouteConfig(
    target_length_m=args.km * 1000,
    length_tol_m=min(200.0, 0.03 * args.km * 1000),
    turn_threshold_deg=30.0,
    turn_k=0.2,
    turn_convex=True,
    intersections_once=True,
)

tl = None if args.time_limit <= 0 else args.time_limit
lp = None if args.length_penalty == 0 else args.length_penalty
res = solve_loop(D, start, cfg, time_limit=tl, length_penalty=lp, msg=not args.quiet)

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