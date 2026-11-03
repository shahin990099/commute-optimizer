"""
algorithms/beam_search.py
Local Beam Search maintaining K parallel candidate states.
"""

from typing import Tuple
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors


class LocalBeamSearchSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.beam_width: int = self.params.get("beam_width", 5)
        self.max_iterations: int = self.params.get("max_iterations", 150)

    def solve(self) -> Tuple[State, float]:
        """
        Executes Local Beam Search.

        Guidelines:
        - Maintain a set of `beam_width` parallel configurations.
        - In each iteration, generate neighbors from all active states in the beam.
        - Pool and rank all generated successors collectively, then retain the top `beam_width` states.
        - Halt when the beam ceases to improve or reaches max_iterations.
        - Record the best cost of the population at each step in self.history.

        Encouraged:
        - Ensuring diversity across states in the beam to prevent collapse into a single redundant path.

        Prohibited:
        - Treating this as K independent, non-communicating searches.
        """
        # TODO: Implement Local Beam Search
        problem = self.problem
        beam = list(dict.fromkeys(problem.generate_initial_state() for _ in range(self.beam_width)))

        best_state = min(beam, key=problem.evaluate)
        best_cost = problem.evaluate(best_state)
        self.history.append(best_cost)

        for _ in range(self.max_iterations):
            pool = {}
            for state in beam:
                for neighbor in get_neighbors(state, problem):
                    pool[neighbor] = problem.evaluate(neighbor)
            if not pool:
                break

            ranked = sorted(pool, key=pool.get)
            new_beam = []
            path_count = {}
            for candidate in ranked:
                if len(new_beam) == self.beam_width:
                    break
                if path_count.get(candidate.path, 0) >= 2:
                    continue
                new_beam.append(candidate)
                path_count[candidate.path] = path_count.get(candidate.path, 0) + 1

            step_best_cost = pool[ranked[0]]
            self.history.append(step_best_cost)

            if step_best_cost < best_cost:
                best_cost = step_best_cost
                best_state = ranked[0]
                beam = new_beam
            else:
                break

        return best_state, best_cost
        # raise NotImplementedError("Implement solve() in algorithms/beam_search.py")
