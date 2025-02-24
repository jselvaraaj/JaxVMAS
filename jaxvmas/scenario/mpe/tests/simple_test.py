#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import jax.numpy as jnp

from jaxvmas.runner_script import run_heuristic
from jaxvmas.simulator.heuristic_policy import RandomPolicy


def test_run_heuristic():
    expected_reward = -272.2748107910156
    actual_reward = run_heuristic(
        scenario_name="simple",
        heuristic=RandomPolicy,
        n_envs=32,
        n_steps=200,
        render=False,
        save_render=False,
    )
    assert jnp.isclose(
        actual_reward, expected_reward, atol=1e-5
    ), f"Expected reward: {expected_reward}, but got: {actual_reward}"


def test_run_heuristic_with_render():
    expected_reward = -272.2748107910156
    actual_reward = run_heuristic(
        scenario_name="simple",
        heuristic=RandomPolicy,
        n_envs=32,
        n_steps=200,
        render=True,
        save_render=False,
    )
    assert jnp.isclose(
        actual_reward, expected_reward, atol=1e-5
    ), f"Expected reward: {expected_reward}, but got: {actual_reward}"


if __name__ == "__main__":
    from jaxvmas.simulator.heuristic_policy import RandomPolicy

    run_heuristic(
        scenario_name="simple",
        heuristic=RandomPolicy,
        n_envs=32,
        n_steps=200,
        render=False,
        save_render=False,
    )
