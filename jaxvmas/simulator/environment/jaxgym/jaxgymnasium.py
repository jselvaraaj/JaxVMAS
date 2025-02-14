#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

"""
JAX-compatible Gymnasium wrapper for single environment instances.
Ensures all operations are jittable and compatible with JAX transformations.
"""


from jaxtyping import Array, PyTree

from jaxvmas.equinox_utils import dataclass_to_dict_first_layer
from jaxvmas.simulator.environment.environment import Environment
from jaxvmas.simulator.environment.jaxgym.base import BaseJaxGymWrapper, EnvData

# Type definitions for dimensions
batch = "batch"  # Batch dimension for vectorized environments
agents = "agents"  # Number of agents dimension
action = "action"  # Action dimension
obs = "obs"  # Observation dimension


class JaxGymnasiumWrapper(BaseJaxGymWrapper):
    """JAX-compatible Gymnasium wrapper for single environment instances."""

    render_mode: str

    @classmethod
    def create(
        cls,
        env: Environment,
        render_mode: str = "human",
    ):
        """Initialize the wrapper.

        Args:
            env: The JAX environment to wrap
            return_numpy: Whether to convert outputs to numpy arrays
        """

        assert (
            env.num_envs == 1
        ), "JaxGymnasiumWrapper only supports singleton environments. For vectorized environments, use JaxGymnasiumVecWrapper."

        assert (
            env.terminated_truncated
        ), "JaxGymnasiumWrapper requires termination and truncation flags. Set terminated_truncated=True in environment."

        base_wrapper = BaseJaxGymWrapper.create(env=env, vectorized=False)

        return cls(
            **dataclass_to_dict_first_layer(base_wrapper),
            render_mode=render_mode,
        )

    def step(self, action: list) -> tuple["JaxGymnasiumWrapper", EnvData]:
        """Take a step in the environment.

        Args:
            state: Current environment state
            action: Action to take

        Returns:
            Tuple of (new state, step data)
        """
        # Convert action to expected format and step environment
        action = self._action_list_to_array(action)
        env, (obs, rews, terminated, truncated, info) = self.env.step(action)
        self = self.replace(env=env)

        # Convert outputs to appropriate format
        env_data = self._convert_env_data(
            obs=obs,
            rews=rews,
            info=info,
            terminated=terminated,
            truncated=truncated,
        )

        return self, env_data

    def reset(
        self,
        *,
        options: dict | None = None,
    ) -> tuple["JaxGymnasiumWrapper", tuple[PyTree, dict]]:

        # Reset environment state
        env, (obs, info) = self.env.reset_at(
            index=0,
            return_observations=True,
            return_info=True,
        )
        self = self.replace(env=env)

        # Convert outputs
        env_data = self._convert_env_data(obs=obs, info=info)
        return self, (env_data.obs, env_data.info)

    def render(
        self,
        agent_index_focus: int | None = None,
        visualize_when_rgb: bool = False,
        **kwargs,
    ) -> Array | None:
        return self.env.render(
            mode=self.render_mode,
            env_index=0,
            agent_index_focus=agent_index_focus,
            visualize_when_rgb=visualize_when_rgb,
            **kwargs,
        )
