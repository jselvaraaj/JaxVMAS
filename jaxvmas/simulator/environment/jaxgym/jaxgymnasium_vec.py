#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

"""
JAX-compatible Gymnasium wrapper for vectorized environment instances.
Ensures all operations are jittable and compatible with JAX transformations.
"""

import jax
from jaxtyping import Array, PyTree

from jaxvmas.equinox_utils import dataclass_to_dict_first_layer
from jaxvmas.simulator.environment.environment import Environment
from jaxvmas.simulator.environment.jaxgym.base import BaseJaxGymWrapper, EnvData
from jaxvmas.simulator.environment.jaxgym.spaces import Space

# Type definitions for dimensions
batch = "batch"  # Batch dimension for vectorized environments
agents = "agents"  # Number of agents dimension
action = "action"  # Action dimension
obs = "obs"  # Observation dimension


class JaxGymnasiumVecWrapper(BaseJaxGymWrapper):
    """JAX-compatible Gymnasium wrapper for vectorized environment instances."""

    render_mode: str

    @classmethod
    def create(
        cls,
        env: Environment,
        render_mode: str = "human",
    ):
        base_wrapper = BaseJaxGymWrapper.create(env=env, vectorized=True)

        assert (
            env.terminated_truncated
        ), "JaxGymnasiumVecWrapper requires termination and truncation flags. Set terminated_truncated=True in environment."

        return cls(
            **dataclass_to_dict_first_layer(base_wrapper),
            render_mode=render_mode,
        )

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    @property
    def observation_space(self) -> Space:
        return self.env.observation_space

    @property
    def action_space(self) -> Space:
        return self.env.action_space

    def step(
        self, PRNG_key: Array, action: PyTree
    ) -> tuple["JaxGymnasiumVecWrapper", EnvData]:
        """Take a step in the environment.

        Args:
            action: Action to take

        Returns:
            Tuple of (new state, step data)
        """
        # Convert action to expected format and step environment
        action = self._action_list_to_array(action)
        PRNG_key, subkey = jax.random.split(PRNG_key)
        env, (obs, rews, terminated, truncated, info) = self.env.step(
            PRNG_key=subkey, actions=action
        )
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
        PRNG_key: Array,
        *,
        options: dict | None = None,
    ) -> tuple["JaxGymnasiumVecWrapper", tuple[PyTree, dict]]:

        # Reset environment state
        env, (obs, info) = self.env.reset(
            PRNG_key=PRNG_key,
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
            agent_index_focus=agent_index_focus,
            visualize_when_rgb=visualize_when_rgb,
            **kwargs,
        )
