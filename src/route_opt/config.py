"""All tunable parameters live here."""
from dataclasses import dataclass

from route_opt import costs


def _pos(G, n):
    return (G.nodes[n]["x"], G.nodes[n]["y"])


@dataclass
class RouteConfig:
    # Street cost weights
    alpha: float = 1.0      # per metre of ascent
    beta: float = 0.5       # per metre of descent
    gamma: float = 1.0      # per metre of length

    # Turn penalty
    turn_threshold_deg: float = 30.0
    turn_k: float = 0.1
    turn_convex: bool = True    # False = step function

    # Loop constraints
    target_length_m: float = 42195.0
    length_tol_m: float = 200.0

    # Visit rules
    intersections_once: bool = False   # True = each intersection at most once

    def street_cost(self, G, u, v):
        asc, desc = costs.edge_climb(G, u, v)
        return self.alpha * asc + self.beta * desc + self.gamma * G.edges[u, v]["length"]

    def turn_cost(self, G, u, v, w):
        theta = costs.turn_angle_deg(
            costs.heading_deg(_pos(G, u), _pos(G, v)),
            costs.heading_deg(_pos(G, v), _pos(G, w)),
        )
        return costs.turn_penalty(
            theta, threshold=self.turn_threshold_deg, k=self.turn_k, convex=self.turn_convex
        )