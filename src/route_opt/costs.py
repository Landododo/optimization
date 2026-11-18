"""Street and turn costs. Coordinates are planar (x = east, y = north), in meters."""
import math


def heading_deg(p, q):
    """Compass heading from p to q in [0, 360)."""
    dx = q[0] - p[0]
    dy = q[1] - p[1]
    return math.degrees(math.atan2(dx, dy)) % 360.0


def turn_angle_deg(h_in, h_out):
    """Absolute change in heading, wrapped to [0, 180]."""
    d = abs(h_out - h_in) % 360.0
    return min(d, 360.0 - d)


def turn_penalty(theta, threshold=30.0, k=0.1, convex=True):
    """Zero below threshold. Convex: k * (theta - threshold)^2. Step: k."""
    if theta <= threshold:
        return 0.0
    if convex:
        return k * (theta - threshold) ** 2
    return k


def street_cost(elev_u, elev_v, length, alpha=1.0, beta=0.5, gamma=1.0):
    """alpha weights ascent, beta weights descent, gamma weights length."""
    ascent = max(0.0, elev_v - elev_u)
    descent = max(0.0, elev_u - elev_v)
    return alpha * ascent + beta * descent + gamma * length

def edge_climb(G, u, v):
    """(ascent, descent) in metres for street u->v.
    Uses edge attributes from the DEM pipeline if present; otherwise node elevations."""
    d = G.edges[u, v]
    if "ascent" in d and "descent" in d:
        return d["ascent"], d["descent"]
    dz = G.nodes[v]["elev"] - G.nodes[u]["elev"]
    return max(0.0, dz), max(0.0, -dz)