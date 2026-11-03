"""
algorithms/beam_search.py
Local Beam Search maintaining K parallel candidate states.
"""
import random
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
        self.history.clear()

        if self.beam_width < 1:
            raise ValueError("beam_width must be at least 1.")
        if self.max_iterations < 0:
            raise ValueError("max_iterations must be nonnegative.")

        valid_times = problem.get_valid_time_domain()
        cost_cache = {}

        def evaluate(state):
            if state not in cost_cache:
                cost_cache[state] = problem.evaluate(state)
            return cost_cache[state]

        def random_state():
            stack = [(problem.origin, (problem.origin,))]

            while stack:
                node, path = stack.pop()

                if node == problem.destination:
                    return State(path, random.choice(valid_times))

                neighbors = list(problem.network.get_neighbors(node))
                random.shuffle(neighbors)

                for neighbor in neighbors:
                    if neighbor not in path:
                        stack.append((neighbor, path + (neighbor,)))

            raise RuntimeError("No route connects origin to destination.")

        initial_states = {problem.generate_initial_state()}

        for _ in range(max(100, 20 * self.beam_width)):
            if len(initial_states) >= self.beam_width:
                break
            initial_states.add(random_state())

        beam = sorted(
            initial_states,
            key=lambda state: (
                evaluate(state),
                state.path,
                state.departure_time
            )
        )[:self.beam_width]

        best_state = beam[0]
        best_cost = evaluate(best_state)
        self.history.append(best_cost)

        seen_beams = {frozenset(beam)}

        for _ in range(self.max_iterations):
            candidates = set(beam)

            for state in beam:
                for neighbor in get_neighbors(state, problem):
                    if (
                        problem.is_valid_state(neighbor)
                        and neighbor.departure_time in valid_times
                    ):
                        candidates.add(neighbor)

            new_beam = sorted(
                candidates,
                key=lambda state: (
                    evaluate(state),
                    state.path,
                    state.departure_time
                )
            )[:self.beam_width]

            step_best_state = new_beam[0]
            step_best_cost = evaluate(step_best_state)

            if step_best_cost < best_cost:
                best_state = step_best_state
                best_cost = step_best_cost

            self.history.append(best_cost)

            signature = frozenset(new_beam)
            if signature in seen_beams:
                break

            seen_beams.add(signature)
            beam = new_beam

        return best_state, best_cost
        # raise NotImplementedError("Implement solve() in algorithms/beam_search.py")
