#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

"""
JAX-based vectorized multi-agent environment implementation.
This version separates static (stored in self) and dynamic (stored in EnvironmentState)
variables and threads the RNG key through every functional call.
"""

import chex
import jax
import jax.numpy as jnp
from flax import struct
from jaxtyping import Array, Int

from jaxvmas.simulator.core import Agent, World
from jaxvmas.simulator.scenario import BaseScenario

# Type definitions for dimensions
batch = "batch"  # Batch dimension for vectorized environments
agents = "agents"  # Number of agents dimension
action = "action"  # Action dimension
obs = "obs"  # Observation dimension
comm = "comm"  # Communication dimension
physical = "physical"  # Physical action dimension
discrete = "discrete"  # Discrete action dimension


class EnvironmentState(struct.PyTreeNode):
    """Dynamic environment state (a PyTree) that is updated during episodes."""

    batch_size: int = struct.field(pytree_node=False)  # Static field (never changes)
    steps: Int[Array, f"{batch}"]
    world_state: World  # The world state maintained by the scenario
    rng: chex.PRNGKey  # RNG key

    @classmethod
    def create(
        cls, batch_size: int, world_state: World, seed: int = 0
    ) -> "EnvironmentState":
        """Create an initial environment state."""
        return cls(
            batch_size=batch_size,
            steps=jnp.zeros(batch_size, dtype=jnp.int32),
            world_state=world_state,
            rng=jax.random.PRNGKey(seed),
        )

    def reset(self, env_index: int | None = None) -> "EnvironmentState":
        """Reset step counters for the entire batch or a specific environment index."""
        if env_index is None:
            return self.replace(steps=jnp.zeros_like(self.steps))
        # Reset specific environment index only.
        mask = jnp.arange(self.batch_size) == env_index
        return self.replace(steps=jnp.where(mask, 0, self.steps))


class Environment:
    """JAX-based vectorized multi-agent environment.

    The static values (scenario, number of environments, configuration, etc.)
    are stored in the instance, while the dynamic values (world state, steps, RNG)
    are stored in an EnvironmentState instance that must be passed into step/reset.
    """

    def __init__(
        self,
        scenario: BaseScenario,
        num_envs: int = 32,
        max_steps: int | None = None,
        continuous_actions: bool = True,
        dict_spaces: bool = False,
        multidiscrete_actions: bool = False,
        clamp_actions: bool = False,
        grad_enabled: bool = False,
        terminated_truncated: bool = False,
        **kwargs,
    ):
        """
        Initialize the environment with static parameters.
        Note: The dynamic state (world state, steps, rng) should be created separately,
        e.g., via a helper like EnvironmentState.create(...) once the world is available.
        """
        if multidiscrete_actions:
            assert (
                not continuous_actions
            ), "Multidiscrete actions require discrete actions"

        # Static parameters.
        self.scenario = scenario
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.continuous_actions = continuous_actions
        self.dict_spaces = dict_spaces
        self.multidiscrete_actions = multidiscrete_actions
        self.clamp_actions = clamp_actions
        self.grad_enabled = grad_enabled
        self.terminated_truncated = terminated_truncated

    # Optional: helper to create the initial state.
    def init_state(self, seed: int = 0, **world_kwargs) -> EnvironmentState:
        """
        Create an initial dynamic state. For example, if your scenario
        provides an `env_make_world` method, you might use it here.
        """
        # This assumes that your scenario has a method for creating the world state.
        # Replace the following with your scenario's actual initializer.
        world_state = self.scenario.env_make_world(self.num_envs, **world_kwargs)
        return EnvironmentState.create(
            batch_size=self.num_envs, world_state=world_state, seed=seed
        )

    def reset(
        self,
        state: EnvironmentState,
        env_index: int | None = None,
        return_observations: bool = True,
        return_info: bool = False,
        return_dones: bool = False,
    ) -> tuple[EnvironmentState, list | dict | None]:
        """Reset the environment and return the new state and (optionally) observations/info."""
        # Reset the world state via the scenario.
        new_world_state = self.scenario.reset_world(state.world_state, env_index)
        new_state = state.replace(world_state=new_world_state).reset(env_index)

        result = self._get_from_scenario(
            new_state,
            get_observations=return_observations,
            get_infos=return_info,
            get_rewards=False,
            get_dones=return_dones,
        )
        # Return a single item if only one element is requested.
        return new_state, result[0] if result and len(result) == 1 else result

    def step(
        self, state: EnvironmentState, actions: list[Array] | dict[str, Array]
    ) -> tuple[EnvironmentState, list]:
        """
        Step the environment forward given actions.
        Returns a tuple containing the new state and a list of scenario outputs (e.g. observations, rewards, dones, infos).
        """
        # If actions are provided as a dict, convert to list (based on agent order).
        if isinstance(actions, dict):
            actions_list = []
            for agent in self.scenario.agents:
                try:
                    actions_list.append(actions[agent.name])
                except KeyError:
                    raise AssertionError(f"Agent '{agent.name}' not in action dict")
            actions = actions_list

        assert len(actions) == len(
            self.scenario.agents
        ), f"Expected {len(self.scenario.agents)} actions, got {len(actions)}"

        # Process the raw actions (while threading the RNG key).
        processed_actions, new_rng = self._process_actions(state, actions)
        # Advance the world state.
        new_world_state = self.scenario.step_world(state.world_state, processed_actions)
        new_state = state.replace(
            world_state=new_world_state,
            steps=state.steps + 1,
            rng=new_rng,
        )

        # Get and return the outputs from the scenario.
        scenario_outputs = self._get_from_scenario(
            new_state,
            get_observations=True,
            get_infos=True,
            get_rewards=True,
            get_dones=True,
        )
        return (new_state,) + tuple(scenario_outputs)

    def _get_from_scenario(
        self,
        state: EnvironmentState,
        get_observations: bool,
        get_rewards: bool,
        get_infos: bool,
        get_dones: bool,
        dict_agent_names: bool | None = None,
    ) -> list:
        """Extract information from the scenario (observations, rewards, dones, infos)."""
        if not any([get_observations, get_rewards, get_infos, get_dones]):
            return []

        if dict_agent_names is None:
            dict_agent_names = self.dict_spaces

        result = []

        if get_observations:
            obs = self.scenario.get_obs(state.world_state)
            if dict_agent_names:
                obs = {agent.name: o for agent, o in zip(self.scenario.agents, obs)}
            result.append(obs)

        if get_rewards:
            rewards = self.scenario.get_rewards(state.world_state)
            if dict_agent_names:
                rewards = {
                    agent.name: r for agent, r in zip(self.scenario.agents, rewards)
                }
            result.append(rewards)

        if get_dones:
            done = self.scenario.get_done(state.world_state)
            if self.max_steps:
                if self.terminated_truncated:
                    # If terminated_truncated, compute truncated separately.
                    truncated = state.steps >= self.max_steps
                    result.extend([done, truncated])
                else:
                    done = jnp.logical_or(done, state.steps >= self.max_steps)
                    result.append(done)
            else:
                result.append(done)

        if get_infos:
            infos = self.scenario.get_info(state.world_state)
            if dict_agent_names:
                infos = {agent.name: i for agent, i in zip(self.scenario.agents, infos)}
            result.append(infos)

        return result

    def _process_actions(
        self,
        state: EnvironmentState,
        actions: list[Array],
    ) -> tuple[list[Array], chex.PRNGKey]:
        """
        Process raw actions for all agents, ensuring proper shape and applying any noise/clamping.
        Threads and returns an updated RNG key.
        """
        rng = state.rng
        processed = []
        for agent_idx, action in enumerate(actions):
            if not isinstance(action, jnp.ndarray):
                action = jnp.array(action)
            # Ensure the action has a trailing dimension (e.g. (num_envs, action_dim))
            if len(action.shape) == 1:
                action = jnp.expand_dims(action, -1)

            # Validate action dimensions.
            expected_action_size = self._get_agent_action_size(
                self.scenario.agents[agent_idx]
            )
            assert (
                action.shape[0] == self.num_envs
            ), f"Actions must have batch dim {self.num_envs}"
            assert action.shape[1] == expected_action_size, (
                f"Agent {self.scenario.agents[agent_idx].name}: Expected action shape "
                f"(num_envs, {expected_action_size}), got {action.shape}"
            )

            # Process the action based on its type.
            agent = self.scenario.agents[agent_idx]
            if self.continuous_actions:
                processed_action, rng = self._process_continuous_action(
                    rng, action, agent
                )
            elif self.multidiscrete_actions:
                processed_action, rng = self._process_multidiscrete_action(
                    rng, action, agent
                )
            else:
                processed_action, rng = self._process_discrete_action(
                    rng, action, agent
                )

            processed.append(processed_action)

        return processed, rng

    def _get_agent_action_size(self, agent: Agent) -> int:
        """Determine the expected action size for the given agent."""
        if self.continuous_actions:
            return agent.action_size + (agent.world.dim_c if not agent.silent else 0)
        elif self.multidiscrete_actions:
            return agent.action_size + (
                1 if not agent.silent and agent.world.dim_c != 0 else 0
            )
        else:
            return 1

    def _process_continuous_action(
        self,
        rng: chex.PRNGKey,
        action: Array,
        agent: Agent,
    ) -> tuple[Array, chex.PRNGKey]:
        """Process continuous actions, including clamping and optional noise addition."""
        physical_action = action[..., : agent.action_size]

        if self.clamp_actions:
            physical_action = jnp.clip(
                physical_action, -agent.action_range, agent.action_range
            )

        if agent.action_noise > 0:
            rng, subkey = jax.random.split(rng)
            noise = (
                jax.random.normal(subkey, physical_action.shape) * agent.action_noise
            )
            physical_action = physical_action + noise

        return physical_action * agent.action_multiplier, rng

    def _process_discrete_action(
        self,
        rng: chex.PRNGKey,
        action: Array,
        agent: Agent,
    ) -> tuple[Array, chex.PRNGKey]:
        """Process discrete actions by converting them to one-hot and adding optional noise."""
        physical_action = jax.nn.one_hot(action, agent.action_size)

        if agent.action_noise > 0:
            rng, subkey = jax.random.split(rng)
            noise = (
                jax.random.normal(subkey, physical_action.shape) * agent.action_noise
            )
            physical_action = physical_action + noise

        return physical_action * agent.action_multiplier, rng

    def _process_multidiscrete_action(
        self,
        rng: chex.PRNGKey,
        action: Array,
        agent: Agent,
    ) -> tuple[Array, chex.PRNGKey]:
        """
        Process multidiscrete actions by converting each discrete action into one-hot and concatenating.
        """
        physical_actions = []
        for i, size in enumerate(agent.action_nvec):
            physical_action = jax.nn.one_hot(action[..., i], size)
            if agent.action_noise > 0:
                rng, subkey = jax.random.split(rng)
                noise = (
                    jax.random.normal(subkey, physical_action.shape)
                    * agent.action_noise
                )
                physical_action = physical_action + noise
            physical_actions.append(physical_action * agent.action_multiplier)

        return jnp.concatenate(physical_actions, axis=-1), rng

    def get_random_actions(
        self,
        state: EnvironmentState,
    ) -> tuple[EnvironmentState, list[Array] | dict[str, Array]]:
        """
        Generate random actions for all agents.
        Returns a tuple (new_state, actions) where new_state includes the updated RNG key.
        """
        rng = state.rng
        actions = []
        for agent in self.scenario.agents:
            rng, subkey = jax.random.split(rng)
            if self.continuous_actions:
                action = jax.random.uniform(
                    subkey,
                    shape=(self.num_envs, self._get_agent_action_size(agent)),
                    minval=-agent.action_range,
                    maxval=agent.action_range,
                )
            else:
                action = jax.random.randint(
                    subkey,
                    shape=(self.num_envs,),
                    minval=0,
                    maxval=self._get_agent_action_size(agent),
                )
            actions.append(action)

        new_state = state.replace(rng=rng)
        if self.dict_spaces:
            return new_state, {
                agent.name: act for agent, act in zip(self.scenario.agents, actions)
            }
        return new_state, actions

    def render(self, mode="human"):
        """Render the environment (to be implemented for your specific needs)."""
        raise NotImplementedError("Rendering not implemented for JAX environment")
