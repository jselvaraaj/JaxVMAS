# Copyright (c) 2024.
# ProrokLab (https://www.proroklab.org/)
# All rights reserved.

from typing import TypeVar

import jax.numpy as jnp
from jaxtyping import Array, Bool, PyTree

from jaxvmas.equinox_utils import PyTreeNode
from jaxvmas.simulator.environment.environment import Environment
from jaxvmas.simulator.utils import extract_nested_with_index

# Type definitions for dimensions
batch = "batch"  # Batch dimension for vectorized environments
agents = "agents"  # Number of agents dimension
action = "action"  # Action dimension
obs = "obs"  # Observation dimension

T = TypeVar("T")  # Generic type for nested structures


class EnvData(
    PyTreeNode,
):
    """Immutable container for environment step data."""

    obs: T
    rews: T
    terminated: Bool[Array, f"{batch}"]
    truncated: Bool[Array, f"{batch}"]
    done: Bool[Array, f"{batch}"]
    info: T


class BaseJaxGymWrapper(PyTreeNode):
    """Base class for JAX-based gym environment wrappers."""

    env: Environment
    dict_spaces: bool
    vectorized: bool

    @classmethod
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

    def _convert_output(self, data: Array, item: bool = False) -> Array:
        """Convert output data based on vectorization settings.

        Args:
            data: The data to convert
            item: Whether to convert to a scalar value
        """
        if not self.vectorized:
            # Take first item if not vectorized
            data = extract_nested_with_index(data, index=0)
            if item:
                return data
        return data

    def _compress_infos(self, infos: PyTree) -> dict:
        """Compress info data into a dictionary format.

        Args:
            infos: Info data either as list or dict
        """
        if isinstance(infos, dict):
            return infos
        elif isinstance(infos, (list, tuple)):
            return {self.env.agents[i].name: info for i, info in enumerate(infos)}
        else:
            raise ValueError(
                f"Expected list or dictionary for infos but got {type(infos)}"
            )

    def _convert_env_data(
        self,
        obs: PyTree | None = None,
        rews: PyTree | None = None,
        info: PyTree | None = None,
        terminated: Bool[Array, f"{batch}"] | None = None,
        truncated: Bool[Array, f"{batch}"] | None = None,
        done: Bool[Array, f"{batch}"] | None = None,
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
                    rews[agent] = self._convert_output(rews[agent], item=True)
        else:
            for i in range(len(self.env.agents)):
                if obs is not None:
                    obs[i] = self._convert_output(obs[i])
                if info is not None:
                    info[i] = self._convert_output(info[i])
                if rews is not None:
                    rews[i] = self._convert_output(rews[i], item=True)

        terminated = (
            self._convert_output(terminated, item=True)
            if terminated is not None
            else None
        )
        truncated = (
            self._convert_output(truncated, item=True)
            if truncated is not None
            else None
        )
        done = self._convert_output(done, item=True) if done is not None else None
        info = self._compress_infos(info) if info is not None else None

        return EnvData(
            obs=obs,
            rews=rews,
            terminated=terminated,
            truncated=truncated,
            done=done,
            info=info,
        )

    def _action_list_to_array(self, actions: list) -> list[Array]:
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

    def step(self, action: PyTree) -> tuple["BaseJaxGymWrapper", EnvData]:
        """Take a step in the environment."""
        raise NotImplementedError

    def reset(
        self,
        *,
        options: dict | None = None,
    ) -> tuple["BaseJaxGymWrapper", tuple[PyTree, dict]]:
        """Reset the environment."""
        raise NotImplementedError

    def render(
        self,
        agent_index_focus: int | None = None,
        visualize_when_rgb: bool = False,
        **kwargs,
    ) -> Array | None:
        """Render the environment."""
        raise NotImplementedError
