"""
algorithms/genetic_algorithm.py
Population-based optimization using selection, recombination, and mutation.
"""
import random
from typing import Tuple
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors, crossover


class GeneticAlgorithmSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.population_size: int = self.params.get("population_size", 30)
        self.generations: int = self.params.get("generations", 100)
        self.crossover_rate: float = self.params.get("crossover_rate", 0.8)
        self.mutation_rate: float = self.params.get("mutation_rate", 0.2)
        self.elitism: int = self.params.get("elitism", 2)

    def solve(self) -> Tuple[State, float]:
        """
        Executes Genetic Algorithm optimization.

        Guidelines:
        - Initialize a population of valid candidate solutions.
        - Design a fitness metric suitable for minimizing commute travel time (lower cost -> higher fitness).
        - Use selection operators (such as tournament or roulette selection) to pick parents.
        - Pair parents to produce offspring via `crossover()`, and apply stochastic mutation.
        - Preserve the top candidates across generations if using elitism.
        - Track best population performance per generation in self.history.

        Encouraged:
        - Experimenting with population size, selection pressure, and operator rates to balance exploration vs. exploitation.
        """
        # TODO: Implement Genetic Algorithm
        problem = self.problem
        self.history.clear()

        if self.population_size < 1:
            raise ValueError("population_size must be at least 1.")
        if self.generations < 0:
            raise ValueError("generations must be nonnegative.")
        if not 0 <= self.elitism <= self.population_size:
            raise ValueError("elitism must be between 0 and population_size.")
        if not 0 <= self.crossover_rate <= 1:
            raise ValueError("crossover_rate must be between 0 and 1.")
        if not 0 <= self.mutation_rate <= 1:
            raise ValueError("mutation_rate must be between 0 and 1.")

        valid_times = problem.get_valid_time_domain()

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

        def tournament(scored):
            contestants = random.sample(scored, min(3, len(scored)))
            return min(contestants, key=lambda item: item[1])[0]

        population = [problem.generate_initial_state()]
        unique_states = set(population)

        for _ in range(max(100, 20 * self.population_size)):
            if len(population) >= self.population_size:
                break

            candidate = random_state()
            if candidate not in unique_states:
                population.append(candidate)
                unique_states.add(candidate)

        while len(population) < self.population_size:
            population.append(random_state())

        scored = sorted(
            ((state, problem.evaluate(state)) for state in population),
            key=lambda item: item[1]
        )

        best_state, best_cost = scored[0]
        self.history.append(best_cost)

        for _ in range(self.generations):
            next_population = [
                state for state, _ in scored[:self.elitism]
            ]

            while len(next_population) < self.population_size:
                parent1 = tournament(scored)
                parent2 = tournament(scored)

                if random.random() < self.crossover_rate:
                    child = crossover(parent1, parent2, problem)
                else:
                    child = random.choice((parent1, parent2))

                if random.random() < self.mutation_rate:
                    if random.random() < 0.5:
                        child = State(
                            child.path,
                            random.choice(valid_times)
                        )
                    else:
                        neighbors = [
                            state
                            for state in get_neighbors(child, problem)
                            if problem.is_valid_state(state)
                            and state.departure_time in valid_times
                        ]
                        if neighbors:
                            child = random.choice(neighbors)

                if (
                    not problem.is_valid_state(child)
                    or child.departure_time not in valid_times
                ):
                    child = random_state()

                next_population.append(child)

            scored = sorted(
                (
                    (state, problem.evaluate(state))
                    for state in next_population
                ),
                key=lambda item: item[1]
            )

            if scored[0][1] < best_cost:
                best_state, best_cost = scored[0]

            self.history.append(best_cost)

        return best_state, best_cost
        # raise NotImplementedError("Implement solve() in algorithms/genetic_algorithm.py")
