#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

"""
JAX-compatible Gymnasium wrapper for vectorized environment instances.
Ensures all operations are jittable and compatible with JAX transformations.
"""

from jaxtyping import Array, PyTree

from jaxvmas.simulator.environment.environment import Environment, EnvironmentState
from jaxvmas.simulator.environment.jaxgym.base import BaseJaxGymWrapper, EnvData

# Type definitions for dimensions
batch = "batch"  # Batch dimension for vectorized environments
agents = "agents"  # Number of agents dimension
action = "action"  # Action dimension
obs = "obs"  # Observation dimension


class JaxGymnasiumVecWrapper(BaseJaxGymWrapper):
    """JAX-compatible Gymnasium wrapper for vectorized environment instances."""

    def __init__(
        self,
        env: Environment,
        return_numpy: bool = True,
    ):
        """Initialize the wrapper.

        Args:
            env: The JAX environment to wrap
            return_numpy: Whether to convert outputs to numpy arrays
        """
        super().__init__(env=env, return_numpy=return_numpy, vectorized=True)

        assert (
            self._env.terminated_truncated
        ), "JaxGymnasiumVecWrapper requires termination and truncation flags. Set terminated_truncated=True in environment."

    def step(
        self, state: EnvironmentState, action: PyTree
    ) -> tuple[EnvironmentState, EnvData]:
        """Take a step in the environment.

        Args:
            state: Current environment state
            action: Action to take

        Returns:
            Tuple of (new state, step data)
        """
        # Convert action to expected format and step environment
        action = self._action_list_to_array(action)
        new_state, (obs, rews, terminated, truncated, info) = self._env.step(
            state, action
        )

        # Convert outputs to appropriate format
        env_data = self._convert_env_data(
            obs=obs,
            rews=rews,
            info=info,
            terminated=terminated,
            truncated=truncated,
        )

        return new_state, env_data

    def reset(
        self,
        state: EnvironmentState,
        *,
        options: dict | None = None,
    ) -> tuple[EnvironmentState, tuple[PyTree, dict]]:
        """Reset the environment.

        Args:
            state: Current environment state
            seed: Random seed
            options: Additional options for reset

        Returns:
            Tuple of (new state, (observations, info))
        """
        # Reset environment state
        new_state, (obs, info) = self._env.reset(
            state,
            return_observations=True,
            return_info=True,
        )

        # Convert outputs
        env_data = self._convert_env_data(obs=obs, info=info)
        return new_state, (env_data.obs, env_data.info)

    def render(
        self,
        state: EnvironmentState,
        agent_index_focus: int | None = None,
        visualize_when_rgb: bool = False,
        **kwargs,
    ) -> Array | None:
        """Render the environment.

        Args:
            state: Current environment state
            agent_index_focus: Index of agent to focus on
            visualize_when_rgb: Whether to visualize RGB output
            **kwargs: Additional rendering arguments

        Returns:
            Rendered output as JAX array
        """
        raise NotImplementedError("Rendering not implemented for JAX environment")
