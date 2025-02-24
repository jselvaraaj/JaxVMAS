#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import time
from typing import Type

import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from jaxvmas.make_env import make_env
from jaxvmas.simulator.environment.environment import Environment, RenderObject
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
) -> float:
    assert not (save_render and not render), "To save the video you have to render it"
    if env_kwargs is None:
        env_kwargs = {}

    # Scenario specific variables
    policy = heuristic(continuous_action=True)
    PRNG_key = jax.random.PRNGKey(0)

    PRNG_key, key_env = jax.random.split(PRNG_key)
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
    total_init_time = time.time()
    step = 0
    env, obs = env.reset(PRNG_key=PRNG_key)
    total_reward = 0
    if render:
        render_object = RenderObject()

        for _ in range(n_steps):
            PRNG_key, key_step = jax.random.split(PRNG_key)
            step += 1
            actions = [None] * len(obs)
            for i in range(len(obs)):
                key_step, key_step_i = jax.random.split(key_step)
                actions[i] = policy.compute_action(
                    obs[i], u_range=env.agents[i].u_range, key=key_step_i
                )
            jitted_step = eqx.filter_jit(env.step)
            PRNG_key, key_step_i = jax.random.split(PRNG_key)
            env, (obs, rews, dones, info) = jitted_step(
                PRNG_key=key_step_i, actions=actions
            )
            rewards = jnp.stack(rews, axis=1)
            global_reward = rewards.mean(axis=1)
            mean_global_reward = global_reward.mean(axis=0)
            total_reward += mean_global_reward
            render_object, rgb_array = env.render(
                render_object=render_object,
                mode="rgb_array",
                agent_index_focus=None,
                visualize_when_rgb=True,
            )
            frame_list.append(rgb_array)
    else:
        compile_init_time = time.time()
        init_state = (env, obs, PRNG_key, jnp.asarray(total_reward, dtype=jnp.float32))
        dynamic_init_state, static_state = eqx.partition(init_state, eqx.is_array)

        @chex.assert_max_traces(1)
        def step_fn(dynamic_carry, _):
            carry = eqx.combine(static_state, dynamic_carry)

            env, obs, PRNG_key, total_reward = carry
            PRNG_key, key_step = jax.random.split(PRNG_key)

            actions = [None] * len(obs)
            for i in range(len(obs)):
                key_step, key_step_i = jax.random.split(key_step)
                actions[i] = policy.compute_action(
                    obs[i], u_range=env.agents[i].u_range, key=key_step_i
                )

            # Step environment
            PRNG_key, key_step_i = jax.random.split(PRNG_key)
            env, (next_obs, rews, dones, info) = env.step(
                PRNG_key=key_step_i, actions=actions
            )
            rewards = jnp.stack(rews, axis=1)
            global_reward = rewards.mean(axis=1)
            mean_global_reward = global_reward.mean(axis=0)
            new_total_reward = total_reward + mean_global_reward

            carry = (env, next_obs, PRNG_key, new_total_reward)
            dynamic_carry, _ = eqx.partition(carry, eqx.is_array)
            return dynamic_carry, None

        # dry run to warm up jax
        _ = jax.lax.scan(step_fn, dynamic_init_state, None, length=n_steps)
        jax.block_until_ready(_)
        compile_time = time.time() - compile_init_time
        print(f"Compile time: {compile_time}s")

        simulation_init_time = time.time()
        print("Starting simulation")
        final_carry, _ = jax.lax.scan(step_fn, dynamic_init_state, None, length=n_steps)
        jax.block_until_ready(final_carry)
        env, obs, PRNG_key, total_reward = eqx.combine(static_state, final_carry)
        print("Simulation done", time.time() - simulation_init_time)
    total_time = time.time() - total_init_time
    if render and save_render:
        save_video(scenario_name, frame_list, 1 / env.scenario.world.dt)

    print(
        f"It took: {total_time}s for {n_steps} steps of {n_envs} parallel environments\n"
        f"The average total reward was {total_reward}"
    )
    return total_reward


if __name__ == "__main__":
    from jaxvmas.simulator.heuristic_policy import RandomPolicy

    run_heuristic(
        scenario_name="football",
        heuristic=RandomPolicy,
        n_envs=1,
        n_steps=200,
        render=False,
        save_render=False,
        env_kwargs={
            "control_two_agents": True,
            "n_blue_agents": 5,
            "n_red_agents": 5,
            "ai_blue_agents": False,
            "ai_red_agents": True,
            "ai_strength": 1.0,
            "ai_decision_strength": 1.0,
            "ai_precision_strength": 1.0,
            "n_traj_points": 8,
        },
    )
