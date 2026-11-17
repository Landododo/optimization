"""Independent cost evaluation for a closed route."""
from dataclasses import dataclass


@dataclass
class RouteStats:
    length_m: float
    ascent_m: float
    descent_m: float
    street_cost: float
    turn_penalty: float
    n_penalised_turns: int

    @property
    def total_cost(self):
        return self.street_cost + self.turn_penalty


def route_stats(G, route, cfg):
    if len(route) < 3 or route[0] != route[-1]:
        raise ValueError("route must be a closed loop: route[0] == route[-1]")

    length = asc = desc = scost = 0.0
    for u, v in zip(route[:-1], route[1:]):
        length += G.edges[u, v]["length"]
        dz = G.nodes[v]["elev"] - G.nodes[u]["elev"]
        asc += max(0.0, dz)
        desc += max(0.0, -dz)
        scost += cfg.street_cost(G, u, v)

    # Cyclic turns: includes the turn at the start/finish node.
    seq = route[:-1]
    n = len(seq)
    tpen, nturns = 0.0, 0
    for i in range(n):
        a, b, c = seq[i - 1], seq[i], seq[(i + 1) % n]
        p = cfg.turn_cost(G, a, b, c)
        if p > 0:
            nturns += 1
            tpen += p

    return RouteStats(length, asc, desc, scost, tpen, nturns)