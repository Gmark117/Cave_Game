import unittest

from agents.exploration_policy import RandomDirectionPolicy


class RandomDirectionPolicyTests(unittest.TestCase):
    def test_seeded_policies_reproduce_the_same_choices(self) -> None:
        first = RandomDirectionPolicy(seed=19)
        second = RandomDirectionPolicy(seed=19)

        first_choices = [
            first.choose_direction((10, 20, 30)) for _index in range(8)
        ]
        second_choices = [
            second.choose_direction((10, 20, 30)) for _index in range(8)
        ]

        self.assertEqual(first_choices, second_choices)
        self.assertTrue(set(first_choices).issubset({10, 20, 30}))

    def test_empty_direction_set_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RandomDirectionPolicy(seed=1).choose_direction(())

    def test_weighted_choices_are_seeded_and_favor_stronger_headings(
        self,
    ) -> None:
        first = RandomDirectionPolicy(seed=19)
        second = RandomDirectionPolicy(seed=19)
        weights = {10: 1.0, 20: 8.0, 30: 1.0}

        first_choices = [
            first.choose_weighted_direction(weights) for _index in range(100)
        ]
        second_choices = [
            second.choose_weighted_direction(weights) for _index in range(100)
        ]

        self.assertEqual(first_choices, second_choices)
        self.assertGreater(first_choices.count(20), 60)

    def test_zero_weight_map_falls_back_to_uniform_choice(self) -> None:
        policy = RandomDirectionPolicy(seed=5)

        chosen = policy.choose_weighted_direction({10: 0.0, 20: -2.0})

        self.assertIn(chosen, {10, 20})


if __name__ == "__main__":
    unittest.main()
