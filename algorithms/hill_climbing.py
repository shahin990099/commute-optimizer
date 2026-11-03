"""
algorithms/hill_climbing.py
Hill Climbing search with sideways move limits and random restarts.
"""
import random
from typing import Tuple
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors


class HillClimbingSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.max_sideways: int = self.params.get("max_sideways", 100)
        self.max_restarts: int = self.params.get("max_restarts", 5)

    def solve(self) -> Tuple[State, float]:
        """
        Executes Hill Climbing optimization.

        Guidelines:
        - Search through the neighborhood greedily for candidates that minimize total travel time.
        - Handle flat local maxima / shoulders by allowing up to `max_sideways` non-improving steps.
        - Apply up to `max_restarts` to explore different starting basins.
        - Track the best objective cost seen at each major step in self.history.

        Prohibited:
        - Looping indefinitely on flat plateaus.
        """
        # TODO: Implement Hill Climbing with sideways moves and restarts
        problem = self.problem
        times = problem.get_valid_time_domain()
        start = problem.generate_initial_state()

        best_state = start
        best_cost = problem.evaluate(start)
        self.history.append(best_cost)

        for restart in range(self.max_restarts + 1):
            if restart == 0:
                current = start
            else:
                current = start
                for _ in range(random.randint(1, 5)):
                    nbrs = get_neighbors(current, problem)
                    if nbrs:
                        current = random.choice(nbrs)
                current = State(current.path, random.choice(times))

            cost = problem.evaluate(current)
            visited = {current}
            sideways = 0

            while True:
                nbrs = get_neighbors(current, problem)
                if not nbrs:
                    break
                scored = [(problem.evaluate(n), n) for n in nbrs]
                min_cost = min(c for c, _ in scored)

                if min_cost < cost - 1e-9:
                    choices = [n for c, n in scored if abs(c - min_cost) <= 1e-9]
                    current = random.choice(choices)
                    cost = min_cost
                    sideways = 0
                elif abs(min_cost - cost) <= 1e-9 and sideways < self.max_sideways:
                    choices = [n for c, n in scored
                               if abs(c - min_cost) <= 1e-9 and n not in visited]
                    if not choices:
                        break
                    current = random.choice(choices)
                    sideways += 1
                else:
                    break

                visited.add(current)
                if cost < best_cost:
                    best_state, best_cost = current, cost
                self.history.append(best_cost)

            if cost < best_cost:
                best_state, best_cost = current, cost
            self.history.append(best_cost)

        return best_state, best_cost

        # raise NotImplementedError("Implement solve() in algorithms/hill_climbing.py")
