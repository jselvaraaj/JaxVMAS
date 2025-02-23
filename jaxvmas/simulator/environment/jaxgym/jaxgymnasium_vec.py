#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

"""
JAX-compatible Gymnasium wrapper for vectorized environment instances.
Ensures all operations are jittable and compatible with JAX transformations.
"""

import chex
import jax
from beartype import beartype
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped

from jaxvmas.equinox_utils import dataclass_to_dict_first_layer
from jaxvmas.simulator.environment.environment import Environment, RenderObject
from jaxvmas.simulator.environment.jaxgym.base import BaseJaxGymWrapper, EnvData
from jaxvmas.simulator.environment.jaxgym.spaces import Space
from jaxvmas.simulator.utils import INFO_TYPE, OBS_TYPE

batch_axis_dim = "batch_axis_dim"
BATCHED_ARRAY_TYPE = Float[Array, f"{batch_axis_dim} ..."]


# Type definitions for dimensions
agents_dim = "agents"  # Number of agents dimension
action_dim = "action"  # Action dimension
obs_dim = "obs"  # Observation dimension


@jaxtyped(typechecker=beartype)
class JaxGymnasiumVecWrapper(BaseJaxGymWrapper):
    """JAX-compatible Gymnasium wrapper for vectorized environment instances."""

    render_mode: str

    @classmethod
    @jaxtyped(typechecker=beartype)
    @chex.assert_max_traces(0)
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

    @jaxtyped(typechecker=beartype)
    def step(
        self, PRNG_key: PRNGKeyArray, action: list[Array]
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

    @jaxtyped(typechecker=beartype)
    def reset(
        self,
        PRNG_key: PRNGKeyArray,
        *,
        options: dict | None = None,
    ) -> tuple["JaxGymnasiumVecWrapper", tuple[OBS_TYPE, INFO_TYPE]]:

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

    @jaxtyped(typechecker=beartype)
    @chex.assert_max_traces(0)
    def render(
        self,
        render_object: RenderObject,
        agent_index_focus: int | None = None,
        visualize_when_rgb: bool = False,
        **kwargs,
    ) -> tuple[RenderObject, Array]:
        return self.env.render(
            render_object=render_object,
            mode=self.render_mode,
            agent_index_focus=agent_index_focus,
            visualize_when_rgb=visualize_when_rgb,
            **kwargs,
        )
