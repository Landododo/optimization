"""Download streets + DEM for a Boston region, build the solver graph, and cache it.

Usage (from repo root, venv active):
  python scripts/build_boston_graph.py                      # Esplanade, walk network
  python scripts/build_boston_graph.py --region wide
  python scripts/build_boston_graph.py --dem data/my_dem.tif   # use a DEM you downloaded
"""
import argparse
import sys
from pathlib import Path
from route_opt.boston import Region
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from route_opt.boston import BOSTON_WIDE, ESPLANADE, MARATHON, build_region_graph, fetch_dem, save_graph, summarize
SMALL = Region(west=-71.082, south=42.352, east=-71.068, north=42.360)

REGIONS = {"small": SMALL, "esplanade": ESPLANADE, "wide": BOSTON_WIDE, "marathon": MARATHON}

p = argparse.ArgumentParser()
p.add_argument("--region", choices=REGIONS, default="esplanade")
p.add_argument("--network", choices=["walk", "bike", "drive"], default="walk")
p.add_argument("--dem", help="existing DEM GeoTIFF (lon/lat or any CRS)")
p.add_argument("--segment-m", type=float, default=25.0)
args = p.parse_args()

region = REGIONS[args.region]
data = ROOT / "data"
dem = Path(args.dem) if args.dem else fetch_dem(region, data / f"{args.region}_dem.tif")
print(f"DEM: {dem}")

raw_cache = data / f"raw_{args.region}_{args.network}.graphml"
D = build_region_graph(
    region,
    dem,
    network_type=args.network,
    segment_m=args.segment_m,
    raw_cache_path=raw_cache,
)
out = data / f"{args.region}_{args.network}.pkl"
save_graph(D, out)

for k, v in summarize(D).items():
    print(f"{k:>18}: {v:.2f}" if isinstance(v, float) else f"{k:>18}: {v}")
print(f"saved {out}")