"""Independent cost evaluation for a closed route."""
from dataclasses import dataclass
from route_opt.costs import edge_climb


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
        a, d = edge_climb(G, u, v)
        asc += a
        desc += d
        scost += cfg.street_cost(G, u, v)

    # Interior turns only: excludes the start/finish line where the runner starts and stops.
    tpen, nturns = 0.0, 0
    for a, b, c in zip(route[:-2], route[1:-1], route[2:]):
        p = cfg.turn_cost(G, a, b, c)
        if p > 0:
            nturns += 1
            tpen += p

    return RouteStats(length, asc, desc, scost, tpen, nturns)