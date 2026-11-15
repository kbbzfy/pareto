import unittest

import numpy as np
import torch

from scripts.train import RLTrainer
from src.environments.tsp_env import TSPEnv
from src.problems.tsp import TSPProblem
from src.utils.pareto import nondominated_mask


class TestTSPTraining(unittest.TestCase):
    def setUp(self):
        config = {
            'problem': {'type': 'tsp', 'size': 4},
            'model': {'input_dim': 4, 'hidden_dim': 8},
            'training': {
                'num_episodes': 1, 'buffer_size': 16, 'batch_size': 2,
                'gamma': 0.99, 'learning_rate': 0.001, 'entropy_weight': 0.01,
            },
        }
        self.env = TSPEnv(TSPProblem(config))
        self.trainer = RLTrainer(self.env, config)

    def test_replay_states_keep_separate_city_masks_at_different_steps(self):
        first = self.env.reset(regenerate=False)
        city = int(self.env.get_valid_actions()[0])
        second, _, _, _ = self.env.step(city)
        self.assertEqual(first['visited_mask'][city], 0)
        self.assertEqual(second['visited_mask'][city], 1)

        batch = torch.cat([self.trainer._process_state(state) for state in (first, second)])
        self.assertEqual(batch.shape, (2, 4, 4))
        scores, _, _ = self.trainer.model(batch)
        probabilities = self.trainer._action_distribution(scores, [first, second]).probs
        self.assertGreater(probabilities[0, city].item(), 0)
        self.assertEqual(probabilities[1, city].item(), 0)

    def test_invalid_city_keeps_state_and_test_returns_verified_closed_tour(self):
        state = self.env.reset(regenerate=False)
        with self.assertRaisesRegex(ValueError, '城市 ID'):
            self.env.step(state['current_pos'])
        self.assertEqual(self.env.episode_steps, 0)
        coordinates = self.env.coordinates.copy()

        metrics, tour, objectives = self.trainer.test()

        self.assertTrue(self.env.problem.is_feasible(tour))
        np.testing.assert_array_equal(self.env.coordinates, coordinates)
        expected = np.linalg.norm(np.diff(coordinates[tour], axis=0), axis=1).sum()
        self.assertAlmostEqual(metrics['tour_length'], expected)
        self.assertAlmostEqual(objectives[self.env.objective_names[0]], expected)


    def test_objectives_use_arrival_at_destination_and_have_no_random_columns(self):
        problem = TSPProblem({'problem': {
            'size': 4, 'n_objectives': 4, 'objective_names': ['D', 'T', 'CV', 'Cost'],
        }})
        problem.coordinates = np.column_stack((np.arange(4), np.zeros(4)))
        problem.distance_matrix = problem._compute_distance_matrix()
        tour = [0, 3, 1, 2, 0]
        segments = [3, 2, 1, 2]
        expected = [8, 2, np.std(segments) / 2, 3]
        np.testing.assert_allclose(problem.compute_objectives(tour), expected)
        np.testing.assert_array_equal(problem.compute_objectives(tour), problem.compute_objectives(tour))
        self.assertEqual(problem.objective_names[3], '最长路段')
        with self.assertRaisesRegex(ValueError, '1 至 4'):
            TSPProblem({'problem': {'n_objectives': 5}})

    def test_pareto_mask_honors_min_and_max_directions(self):
        values = [[5, 10], [4, 10], [6, 11], [4, 10], [2, 9]]
        np.testing.assert_array_equal(
            nondominated_mask(values, [-1, 1]), [False, True, True, True, True]
        )
