# Copyright (c) 2024.
# ProrokLab (https://www.proroklab.org/)
# All rights reserved.


import chex
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Bool, Float, PRNGKeyArray, Scalar, jaxtyped

from jaxvmas.equinox_utils import PyTreeNode
from jaxvmas.simulator.environment.environment import Environment
from jaxvmas.simulator.utils import (
    AGENT_INFO_TYPE,
    AGENT_OBS_TYPE,
    AGENT_REWARD_TYPE,
    INFO_TYPE,
    OBS_TYPE,
    REWARD_TYPE,
    extract_nested_with_index,
)

# Type definitions for dimensions

batch_axis_dim = "batch_axis_dim"
BATCHED_ARRAY_TYPE = Float[Array, f"{batch_axis_dim} ..."]

agents_dim = "agents"  # Number of agents dimension
action_dim = "action"  # Action dimension
obs_dim = "obs"  # Observation dimension


@jaxtyped(typechecker=beartype)
class EnvData(
    PyTreeNode,
):
    """Immutable container for environment step data."""

    obs: OBS_TYPE
    rews: REWARD_TYPE
    terminated: Bool[Array, f"{batch_axis_dim}"] | Bool[Scalar, ""]
    truncated: Bool[Array, f"{batch_axis_dim}"] | Bool[Scalar, ""]
    done: Bool[Array, f"{batch_axis_dim}"] | Bool[Scalar, ""]
    info: INFO_TYPE


@jaxtyped(typechecker=beartype)
class BaseJaxGymWrapper(PyTreeNode):
    """Base class for JAX-based gym environment wrappers."""

    env: Environment
    dict_spaces: bool
    vectorized: bool

    @classmethod
    @jaxtyped(typechecker=beartype)
    @chex.assert_max_traces(0)
    def create(
        cls,
        env: Environment,
        vectorized: bool,
    ):
        """Initialize the wrapper.

        Args:
            env: The JAX environment to wrap
            return_numpy: Whether to convert outputs to numpy arrays
            vectorized: Whether to return vectorized (batched) outputs
        """
        self = cls(env, env.dict_spaces, vectorized)
        return self

    @jaxtyped(typechecker=beartype)
    def _convert_output(
        self,
        data: AGENT_OBS_TYPE | AGENT_INFO_TYPE | AGENT_REWARD_TYPE | BATCHED_ARRAY_TYPE,
    ) -> AGENT_OBS_TYPE | AGENT_INFO_TYPE | AGENT_REWARD_TYPE | BATCHED_ARRAY_TYPE:
        """Convert output data based on vectorization settings.

        Args:
            data: The data to convert
            item: Whether to convert to a scalar value
        """
        if not self.vectorized:
            # Take first item if not vectorized
            data = extract_nested_with_index(data, index=0)
        return data

    @jaxtyped(typechecker=beartype)
    def _compress_infos(
        self,
        infos: (
            list[AGENT_INFO_TYPE] | dict[str, AGENT_INFO_TYPE] | tuple[AGENT_INFO_TYPE]
        ),
    ) -> dict[str, AGENT_INFO_TYPE]:
        """Compress info data into a dictionary format.

        Args:
            infos: Info data either as list or dict
        """
        if isinstance(infos, dict):
            return infos
        elif isinstance(infos, (list, tuple)):
            return {self.env.agents[i].name: info for i, info in enumerate(infos)}

    @jaxtyped(typechecker=beartype)
    def _convert_env_data(
        self,
        obs: OBS_TYPE | None = None,
        rews: REWARD_TYPE | None = None,
        info: INFO_TYPE | None = None,
        terminated: Bool[Array, f"{batch_axis_dim}"] | None = None,
        truncated: Bool[Array, f"{batch_axis_dim}"] | None = None,
        done: Bool[Array, f"{batch_axis_dim}"] | None = None,
    ) -> EnvData:
        """Convert environment data to appropriate format.

        Args:
            obs: Observations
            rews: Rewards
            info: Additional info
            terminated: Termination flags
            truncated: Truncation flags
            done: Done flags
        """
        if self.dict_spaces:
            for agent in obs.keys():
                if obs is not None:
                    obs[agent] = self._convert_output(obs[agent])
                if info is not None:
                    info[agent] = self._convert_output(info[agent])
                if rews is not None:
                    rews[agent] = self._convert_output(rews[agent])
            info = self._compress_infos(info) if info is not None else {}
            obs = obs if obs is not None else {}
            rews = rews if rews is not None else {}
        else:
            for i in range(len(self.env.agents)):
                if obs is not None:
                    obs[i] = self._convert_output(obs[i])
                if info is not None:
                    info[i] = self._convert_output(info[i])
                if rews is not None:
                    rews[i] = self._convert_output(rews[i])

            info = self._compress_infos(info) if info is not None else []
            obs = obs if obs is not None else []
            rews = rews if rews is not None else []

        terminated = (
            self._convert_output(terminated)
            if terminated is not None
            else jnp.zeros(self.env.num_envs, dtype=bool)
        )
        truncated = (
            self._convert_output(truncated)
            if truncated is not None
            else jnp.zeros(self.env.num_envs, dtype=bool)
        )
        done = (
            self._convert_output(done)
            if done is not None
            else jnp.zeros(self.env.num_envs, dtype=bool)
        )

        return EnvData(
            obs=obs,
            rews=rews,
            terminated=terminated,
            truncated=truncated,
            done=done,
            info=info,
        )

    @jaxtyped(typechecker=beartype)
    def _action_list_to_array(self, actions: list) -> list[BATCHED_ARRAY_TYPE]:
        """Convert action list to JAX arrays.

        Args:
            actions: List of actions for each agent
        """
        n_agents = self.env.n_agents
        assert (
            len(actions) == n_agents
        ), f"Expecting actions for {n_agents} agents, got {len(actions)} actions"

        return [
            (
                jnp.asarray(
                    act,
                ).reshape(self.env.num_envs, self.env.get_agent_action_size(agent))
                if not isinstance(act, Array)
                else act.reshape(
                    self.env.num_envs, self.env.get_agent_action_size(agent)
                )
            )
            for agent, act in zip(self.env.agents, actions)
        ]

    @jaxtyped(typechecker=beartype)
    def step(
        self, PRNG_key: PRNGKeyArray, action: BATCHED_ARRAY_TYPE
    ) -> tuple["BaseJaxGymWrapper", EnvData]:
        """Take a step in the environment."""
        raise NotImplementedError

    @jaxtyped(typechecker=beartype)
    def reset(
        self,
        *,
        options: dict | None = None,
    ) -> tuple["BaseJaxGymWrapper", tuple[OBS_TYPE, INFO_TYPE]]:
        """Reset the environment."""
        raise NotImplementedError

    @jaxtyped(typechecker=beartype)
    @chex.assert_max_traces(0)
    def render(
        self,
        agent_index_focus: int | None = None,
        visualize_when_rgb: bool = False,
        **kwargs,
    ) -> Array | None:
        """Render the environment."""
        raise NotImplementedError
