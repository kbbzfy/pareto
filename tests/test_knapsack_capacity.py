import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.train import RLTrainer

from src.environments.knapsack_env import KnapsackEnv
from src.problems.knapsack_problem import KnapsackProblem


class TestKnapsackCapacity(unittest.TestCase):
    def make_problem(self, limits, items=None):
        if items is None:
            items = [
                [[10, 5], [4, 1]],
                [[8, 4], [3, 1]],
            ]
        items = np.asarray(items, dtype=float)
        config = {
            'problem': {
                'num_categories': 2,
                'items_per_category': 2,
                'num_objectives': 2,
                'objective_names': ['Value', 'Weight'],
            },
        }
        if limits is not None:
            config['problem']['objective_constraints'] = limits
        with patch.object(KnapsackProblem, '_generate_items', return_value=items):
            return KnapsackProblem(config)

    def test_capacity_changes_solution_for_the_same_items(self):
        for capacity, expected_solution, expected_totals in (
            (9, [0, 0], [18, 9]),
            (4, [1, 1], [7, 2]),
        ):
            with self.subTest(capacity=capacity):
                problem = self.make_problem({'Weight': capacity})
                env = KnapsackEnv(problem)
                state = env.reset()
                solution = []
                done = False
                while not done:
                    valid = np.flatnonzero(state['action_mask'])
                    action = int(valid[0])
                    state, _, done, info = env.step(action)
                    solution.append(action)
                self.assertTrue(info['feasible'])
                self.assertTrue(problem.is_feasible(solution))
                self.assertEqual(solution, expected_solution)
                np.testing.assert_array_equal(problem.compute_objectives(solution), expected_totals)

    def test_over_capacity_action_is_rejected_without_mutating_state(self):
        env = KnapsackEnv(self.make_problem({'Weight': 4}))
        with self.assertRaisesRegex(ValueError, '容量约束'):
            env.step(0)
        self.assertEqual(env.current_state['current_category'], 0)
        self.assertEqual(env.episode_steps, 0)
        np.testing.assert_array_equal(env.current_objectives, [0, 0])

    def test_dead_end_is_not_a_complete_solution(self):
        problem = self.make_problem({'Weight': 5})
        env = KnapsackEnv(problem)
        state, _, done, info = env.step(0)
        self.assertTrue(done)
        self.assertFalse(info['feasible'])
        self.assertFalse(state['action_mask'].any())
        self.assertFalse(problem.is_feasible([0]))

    def test_explicit_empty_constraints_and_legacy_defaults(self):
        self.assertTrue(self.make_problem({}).is_feasible([0, 0]))
        self.assertFalse(self.make_problem(None).is_feasible([0, 0]))

    def test_first_objective_can_be_constrained_and_zero_is_valid(self):
        problem = self.make_problem({'Value': 0}, [[[10, 5], [0, 1]], [[8, 4], [0, 1]]])
        state = KnapsackEnv(problem).reset()
        np.testing.assert_array_equal(state['action_mask'], [0, 1])
        self.assertTrue(problem.is_feasible([1, 1]))

    def test_reports_capacity_below_minimum_required_total(self):
        # Each class has a fitting item by itself, but two classes cannot fit together.
        with self.assertRaisesRegex(ValueError, 'Weight 最少需要 2，容量上限为 1'):
            KnapsackEnv(self.make_problem({'Weight': 1}))

    def test_exported_front_matches_exhaustive_one_category_instance(self):
        items = [[[5, 3], [6, 4], [4, 3], [99, 20]]]
        config = {
            'problem': {
                'type': 'knapsack', 'num_categories': 1, 'items_per_category': 4,
                'num_objectives': 2, 'objective_names': ['Value', 'Weight'],
                'objective_types': [1, -1], 'objective_constraints': {'Weight': 5},
                'items': items,
            },
            'model': {'input_dim': 3, 'hidden_dim': 8, 'n_objectives': 2},
            'training': {
                'num_episodes': 1, 'buffer_size': 8, 'batch_size': 2,
                'gamma': 0.99, 'learning_rate': 0.001, 'entropy_weight': 0.01,
            },
        }
        problem = KnapsackProblem(config)
        trainer = RLTrainer(KnapsackEnv(problem), config)
        with torch.no_grad():
            for parameter in trainer.model.parameters():
                parameter.zero_()
        previous_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as directory, torch.random.fork_rng(devices=[]):
            os.chdir(directory)
            try:
                torch.manual_seed(7)
                metrics, selection, _ = trainer.test(visualize=True)
                report = json.loads(Path('solutions/knapsack_candidates.json').read_text())
                self.assertGreater(Path('solutions/knapsack_pareto_front.png').stat().st_size, 0)
            finally:
                os.chdir(previous_directory)

        candidates = {tuple(row['selection']): row for row in report['candidates']}
        self.assertEqual(set(candidates), {(0,), (1,), (2,)})
        self.assertEqual({key for key, row in candidates.items() if row['nondominated']}, {(0,), (1,)})
        self.assertEqual(metrics['num_nondominated'], 2)
        self.assertEqual(selection, report['candidates'][report['selected_index']]['selection'])
        for key, row in candidates.items():
            self.assertTrue(problem.is_feasible(key))
            np.testing.assert_array_equal(row['objectives'], report['items'][0][key[0]])
