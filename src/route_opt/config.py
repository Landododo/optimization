"""All tunable parameters live here."""
from dataclasses import dataclass

from route_opt import costs


def _pos(G, n):
    return (G.nodes[n]["x"], G.nodes[n]["y"])


# In src/route_opt/config.py
@dataclass
class RouteConfig:
    alpha: float = 1.0
    beta: float = 0.5
    gamma: float = 1.0

    turn_threshold_deg: float = 30.0
    turn_k: float = 0.1
    turn_convex: bool = True

    target_length_m: float = 42195.0
    length_tol_m: float = 200.0
    intersections_once: bool = True   # Default to True per problem constraints

    def street_cost(self, G, u, v):
        asc, desc = costs.edge_climb(G, u, v)
        return self.alpha * asc + self.beta * desc + self.gamma * G.edges[u, v]["length"]

    def turn_cost(self, G, u, v, w):
        e1, e2 = G.edges[u, v], G.edges[v, w]
        h_in = e1.get("out_bearing", costs.heading_deg(_pos(G, u), _pos(G, v)))
        h_out = e2.get("in_bearing", costs.heading_deg(_pos(G, v), _pos(G, w)))
        theta = costs.turn_angle_deg(h_in, h_out)
        return costs.turn_penalty(
            theta, threshold=self.turn_threshold_deg, k=self.turn_k, convex=self.turn_convex
        )