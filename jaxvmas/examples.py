#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import time
from typing import Type

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxvmas.make_env import make_env
from jaxvmas.simulator.environment.environment import Environment
from jaxvmas.simulator.heuristic_policy import BaseHeuristicPolicy, RandomPolicy
from jaxvmas.simulator.utils import save_video


def run_heuristic(
    scenario_name: str,
    heuristic: Type[BaseHeuristicPolicy] = RandomPolicy,
    n_steps: int = 200,
    n_envs: int = 32,
    env_kwargs: dict = None,
    render: bool = False,
    save_render: bool = False,
):
    assert not (save_render and not render), "To save the video you have to render it"
    if env_kwargs is None:
        env_kwargs = {}

    # Scenario specific variables
    policy = heuristic(continuous_action=True)
    key = jax.random.PRNGKey(0)

    key, key_env = jax.random.split(key)
    env: Environment = make_env(
        scenario=scenario_name,
        num_envs=n_envs,
        PRNG_key=key_env,
        continuous_actions=True,
        wrapper=None,
        # Environment specific variables
        **env_kwargs,
    )

    frame_list = []  # For creating a gif
    init_time = time.time()
    step = 0
    env, obs = env.reset(PRNG_key=key)
    total_reward = 0
    for _ in range(n_steps):
        key, key_step = jax.random.split(key)
        step += 1
        actions = [None] * len(obs)
        for i in range(len(obs)):
            actions[i] = policy.compute_action(
                obs[i], u_range=env.agents[i].u_range, key=key_step
            )
        jitted_step = eqx.filter_jit(env.step)
        env, (obs, rews, dones, info) = jitted_step(actions)
        rewards = jnp.stack(rews, axis=1)
        global_reward = rewards.mean(axis=1)
        mean_global_reward = global_reward.mean(axis=0)
        total_reward += mean_global_reward
        if render:
            env, rgb_array = env.render(
                mode="rgb_array",
                agent_index_focus=None,
                visualize_when_rgb=True,
            )
            frame_list.append(rgb_array)

    total_time = time.time() - init_time
    if render and save_render:
        save_video(scenario_name, frame_list, 1 / env.scenario.world.dt)

    print(
        f"It took: {total_time}s for {n_steps} steps of {n_envs} parallel environments\n"
        f"The average total reward was {total_reward}"
    )


if __name__ == "__main__":
    from jaxvmas.simulator.heuristic_policy import RandomPolicy

    run_heuristic(
        scenario_name="simple",
        heuristic=RandomPolicy,
        n_envs=32,
        n_steps=200,
        render=True,
        save_render=False,
    )
