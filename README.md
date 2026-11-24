# Marathon Route Optimization

This project utilizes mixed-integer linear programming (MILP) and graph-search for generating closed-loop running and marathon routes from any starting coordinate. Using OpenStreetMap (OSM) street networks and USGS 3DEP Digital Elevation Models (DEM), the solver finds simple loops of a target distance (e.g., $42.195\text{ km}$ for a marathon) that minimize cumulative uphill/downhill fatigue and sharp deceleration turns.

---

## Problem Formulation & Features

* **Line-Graph & Turn Modeling:** Directed street segments are modeled with exact road-tangent entry and exit bearings. Curving streets with sharp interior bends ($> 30^\circ$) are automatically split into sub-segments so mid-street hairpins are penalized alongside intersection turns.
* **Asymmetric Elevation Profiling:** Street geometries are densified every $25\text{ m}$ and sampled against high-resolution DEM rasters (with moving-average smoothing) to compute separate cumulative climb ($\alpha$) and descent ($\beta$) costs within every segment.
* **Degree-Based Turn Penalties:** Turns below a configurable angle threshold (default $30^\circ$) incur zero penalty; sharper turns incur either a convex power penalty $k(\theta - \theta_0)^2$ or a step penalty (based on turn_convex value).
* **Exact Route Constraints:**
  * Closed loop starting and ending at the same chosen coordinate.
  * Target distance enforcement via either a hard window (`--length-penalty 0`) or a soft deviation penalty.
  * Simple path enforcement: each physical intersection and undirected street segment is traversed at most once.
  * Lazy subtour elimination cuts to guarantee a single connected loop.

---

## Repository Structure

```text
├── scripts/
│   ├── build_boston_graph.py   # Download OSM + DEM data, split sharp bends, cache graph (.pkl)
│   ├── solve_boston_loop.py    # Run the MILP solver and export route map + elevation profile
│   ├── demo_loop.py            # Fast MILP demo on a synthetic 6x6 Gaussian hill grid
│   └── quick_walk.py           # Fast midpoint shortest-path heuristic baseline (no MILP)
├── src/route_opt/
│   ├── boston.py               # OSMnx downloader (with retry/cache), bend splitter, DEM sampler
│   ├── config.py               # Tunable weights (alpha, beta, gamma, turn threshold/convexity)
│   ├── costs.py                # Compass bearings, turn angles, elevation & turn cost functions
│   ├── line_graph.py           # Directed line-graph builder and point-to-point baseline
│   ├── solve_milp.py           # PuLP + HiGHS MILP formulation with lazy subtour elimination
│   ├── evaluate.py             # Independent route verification & statistics
│   ├── toy.py                  # Synthetic grid generators for unit tests
│   └── viz.py                  # Matplotlib map + elevation profile plotter
├── tests/                      # Pytest suite
├── data/                       # Cached DEMs (.tif), raw OSM (.graphml), and solver graphs (.pkl)
└── output/                     # Generated route PNG visualizations
```

---

## Setup & Installation

1. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies and the package in editable mode:**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

3. **Verify your installation with the test suite and synthetic grid demo:**
   ```bash
   pytest
   python scripts/demo_loop.py
   ```
   This solves a $2\text{ km}$ loop around a synthetic hill and saves `output/loop_route.png`.

---

## Step 1: Build & Cache the Street + Elevation Graph

Before solving a route, build the processed graph for your target region. `build_boston_graph.py` downloads the street network from OpenStreetMap, fetches a $10\text{ m}$ USGS 3DEP DEM (or reads a local GeoTIFF), splits sharp mid-street bends, computes segment climbs/descents, and caches the result in `data/`.

### Available Region Presets
* `small`: Compact Back Bay / Esplanade test box
* `esplanade`: Charles River Esplanade & surrounding neighborhoods (good for $3\text{–}10\text{ km}$ runs)
* `wide`: Central Boston
* `marathon`: Greater Boston bounding box ($\approx 12\text{ km} \times 9\text{ km}$, sized for full $42.195\text{ km}$ non-repeating loops)

### Build Commands

**Recommended for full marathons (`bike` network includes roads + paved paths like the Esplanade while keeping graph size tractable):**
```bash
python scripts/build_boston_graph.py --region marathon --network bike
```

**For smaller neighborhood runs (`walk` network includes footpaths and crosswalks):**
```bash
python scripts/build_boston_graph.py --region esplanade --network walk
```

**Using a manually downloaded DEM GeoTIFF:**
If you download an elevation raster directly from the [USGS National Map Downloader](https://apps.nationalmap.gov/downloader/#/elevation), pass `--dem`:
```bash
python scripts/build_boston_graph.py --region marathon --network bike --dem data/boston_dem.tif
```

> **Note on Caching & Overpass Rate Limits:** The builder automatically retries transient `502 Bad Gateway` / `504` Overpass API errors with exponential backoff and saves the raw OSM download to `data/raw_<region>_<network>.graphml`. Re-running the script with different `--segment-m` settings will reuse the local `.graphml` without hitting Overpass again.

---

## Step 2: Solve for an Optimal Route & Generate Visuals

Run `solve_boston_loop.py` with Python's unbuffered flag (`-u`) so you get real-time timestamped logs, live HiGHS branch-and-bound output, and 15-second heartbeat updates during lazy subtour-cut rounds.

### 1. Quick Smoke Test ($5\text{ km}$ Loop)
```bash
python -u scripts/solve_boston_loop.py \
  --region marathon \
  --network bike \
  --km 5 \
  --length-penalty 0 \
  --time-limit 0
```

### 2. Full Marathon ($42.195\text{ km}$ Loop)
```bash
python -u scripts/solve_boston_loop.py \
  --region marathon \
  --network bike \
  --km 42.195 \
  --start-lon -71.0775 \
  --start-lat 42.3536 \
  --length-penalty 0 \
  --time-limit 0
```

### CLI Options for `solve_boston_loop.py`

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--region` | `esplanade` / `marathon` | Cached region name matching `data/<region>_<network>.pkl` |
| `--network` | `walk` | Network type (`bike` recommended for $42.195\text{ km}$; `walk` for $\le 10\text{ km}$) |
| `--km` | `5.0` | Target loop distance in kilometers (use `42.195` for a marathon) |
| `--start-lon` | `-71.07847` | Starting longitude (snaps to nearest graph node) |
| `--start-lat` | `42.34962` | Starting latitude (snaps to nearest graph node) |
| `--length-penalty` | `10.0` | Cost per meter off target; pass `0` to enforce a hard distance window |
| `--time-limit` | `0` | Solver time limit in seconds (`0` = run until a valid single loop is found) |
| `--quiet` | `False` | Suppress live HiGHS solver table output (keeps timestamped stage logs) |

---

## Step 3: Fast Heuristic Baseline (Optional)

If you want an instant route preview without running the MILP solver, `quick_walk.py` samples candidate midpoints at roughly half the target distance and stitches together two edge-disjoint shortest paths:

```bash
python scripts/quick_walk.py --region esplanade --network walk --km 5.0
```

---

## Generated Outputs

Each run prints summary metrics to the terminal and saves a two-panel figure in `output/`:

* **`output/<region>_loop.png`** (from `solve_boston_loop.py`)
* **`output/<region>_quick_<km>km.png`** (from `quick_walk.py`)
* **`output/loop_route.png`** (from `demo_loop.py`)

**What is shown in the generated PNG:**
1. **Left Panel (Route Map):** The background street network in light gray overlaid with the optimal closed loop colored by elevation (`viridis`), directional arrows showing travel direction along every segment, and a red star marking the start/finish line.
2. **Right Panel (Elevation Profile):** Cumulative distance ($\text{km}$) plotted against elevation ($\text{m}$) along the entire route.

---

## Tuning Cost Weights (`src/route_opt/config.py`)

Edit `RouteConfig` in `src/route_opt/config.py` (or inside `scripts/solve_boston_loop.py`) to experiment with different runner trade-offs:

* `alpha` (default `1.0`): Penalty weight per meter of uphill climb.
* `beta` (default `0.5`): Penalty weight per meter of downhill descent.
* `gamma` (default `1.0`): Base cost per meter of distance traveled.
* `turn_threshold_deg` (default `30.0`): Minimum deflection angle (in degrees) before a turn is penalized.
* `turn_k` (default `0.2`): Multiplier controlling how aggressively sharp turns are avoided relative to hill climbs.
* `turn_convex` (default `True`): `True` applies $k(\theta - \theta_0)^2$; `False` applies a flat step penalty $k$.

# optimization
steps:
download file from https://apps.nationalmap.gov/downloader/#/elevation
    curl the file directly to download once you get download link (like curl -L -o data/boston_dem.tif "LINK")

python scripts/build_boston_graph.py --dem PATH_TO_ELEVATION.tif
python scripts/solve_boston_loop.py --km 3

# future improvements
In the future, we could improve upon these heuristic penalties, including some sort of convex penalty for elevation as well,
with different penalties for 'alpha' and 'beta' (uphill vs downhill). Also, further fine-tuning on what the turn penalties should be a route of further improvements. Finally, we also want better visualization to visualize amount of turns and how our model optimized.