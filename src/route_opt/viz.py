"""Route visualisation: map + elevation profile."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


def _xy(G, n):
    return (G.nodes[n]["x"], G.nodes[n]["y"])


def plot_route(G, route, path=None, title=None, show=False):
    """
    G: graph with x, y (metres if projected), elev on nodes, length on edges.
    route: closed node list, route[0] == route[-1].
    path: if given, save PNG here.
    Returns the matplotlib Figure.
    """
    fig, (ax, ax_prof) = plt.subplots(
        1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [2, 1]}
    )

    # Background network
    bg = [(_xy(G, u), _xy(G, v)) for u, v in G.edges]
    ax.add_collection(LineCollection(bg, colors="0.85", linewidths=0.8, zorder=1))

    # Route, coloured by mean elevation of each segment
    segs, seg_elev, dists = [], [], [0.0]
    for u, v in zip(route[:-1], route[1:]):
        segs.append((_xy(G, u), _xy(G, v)))
        seg_elev.append(0.5 * (G.nodes[u]["elev"] + G.nodes[v]["elev"]))
        dists.append(dists[-1] + G.edges[u, v]["length"])

    elev_route = [G.nodes[n]["elev"] for n in route]
    lo, hi = min(elev_route), max(elev_route)
    norm = Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1.0)

    lc = LineCollection(segs, cmap="viridis", norm=norm, linewidths=4, zorder=2)
    lc.set_array(np.array(seg_elev))
    ax.add_collection(lc)
    fig.colorbar(lc, ax=ax, label="elevation (m)", fraction=0.03, pad=0.02)

    # Direction arrows at each segment's midpoint
    P = np.array([p for p, _ in segs])
    Q = np.array([q for _, q in segs])
    if len(segs):
        D = Q - P
        M = (P + Q) / 2
        ax.quiver(
            M[:, 0] - 0.2 * D[:, 0], M[:, 1] - 0.2 * D[:, 1],
            0.4 * D[:, 0], 0.4 * D[:, 1],
            angles="xy", scale_units="xy", scale=1, width=0.004, color="k", zorder=3,
        )

    # Start/finish marker
    sx, sy = _xy(G, route[0])
    ax.scatter([sx], [sy], marker="*", s=250, c="red", zorder=4, label="start / finish")
    ax.legend(loc="best")

    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)

    # Elevation profile
    d_km = np.array(dists) / 1000.0
    ax_prof.plot(d_km, elev_route, color="k")
    ax_prof.fill_between(d_km, elev_route, min(elev_route), alpha=0.25)
    ax_prof.set_xlabel("distance (km)")
    ax_prof.set_ylabel("elevation (m)")
    ax_prof.set_title("Elevation profile")
    ax_prof.grid(alpha=0.3)

    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig