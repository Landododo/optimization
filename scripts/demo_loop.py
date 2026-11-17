"""Solve a sample loop on a hilly grid and save a map. Run from the repo root."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt

from route_opt.config import RouteConfig
from route_opt.solve_milp import solve_loop
from route_opt.toy import grid_graph, hill_elev
from route_opt.viz import plot_route

G = grid_graph(6, 6, spacing=100.0, elev=hill_elev)   # 500 m x 500 m
cfg = RouteConfig(
    target_length_m=2000.0,
    length_tol_m=100.0,
    turn_threshold_deg=30.0,
    turn_k=0.2,
    turn_convex=True,
    intersections_once=True,
)

res = solve_loop(G, (0, 0), cfg)
s = res.stats
print(f"length  {s.length_m:8.1f} m")
print(f"ascent  {s.ascent_m:8.1f} m   descent {s.descent_m:8.1f} m")
print(f"turns penalised: {s.n_penalised_turns}   total cost {s.total_cost:.1f}")

out_dir = Path(__file__).resolve().parents[1] / "output"
out_dir.mkdir(exist_ok=True)
plot_route(G, res.route, path=out_dir / "loop_route.png", title="Demo loop")
print(f"saved {out_dir / 'loop_route.png'}")
plt.close("all")