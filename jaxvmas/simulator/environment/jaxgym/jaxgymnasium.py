#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

"""
JAX-compatible Gymnasium wrapper for single environment instances.
Ensures all operations are jittable and compatible with JAX transformations.
"""


import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, PRNGKeyArray, jaxtyped

from jaxvmas.equinox_utils import dataclass_to_dict_first_layer
from jaxvmas.simulator.environment.environment import Environment, RenderObject
from jaxvmas.simulator.environment.jaxgym.base import BaseJaxGymWrapper, EnvData
from jaxvmas.simulator.utils import INFO_TYPE, OBS_TYPE


@jaxtyped(typechecker=beartype)
class JaxGymnasiumWrapper(BaseJaxGymWrapper):
    """JAX-compatible Gymnasium wrapper for single environment instances."""

    render_mode: str

    @classmethod
    @jaxtyped(typechecker=beartype)
    @chex.assert_max_traces(0)
    def create(
        cls,
        env: Environment,
        render_mode: str = "human",
    ):
        """Initialize the wrapper.

        Args:
            env: The JAX environment to wrap
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

    @property
    def unwrapped(self) -> Environment:
        return self.env

    @eqx.filter_jit
    @jaxtyped(typechecker=beartype)
    def step(
        self, PRNG_key: PRNGKeyArray, action: list
    ) -> tuple["JaxGymnasiumWrapper", EnvData]:
        """Take a step in the environment.

        Args:
            state: Current environment state
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
            done=terminated,
        )

        return self, env_data

    @jaxtyped(typechecker=beartype)
    def reset(
        self,
        PRNG_key: PRNGKeyArray,
        *,
        options: dict | None = None,
    ) -> tuple["JaxGymnasiumWrapper", tuple[OBS_TYPE, INFO_TYPE]]:

        # Reset environment state
        env, (obs, info) = self.env.reset_at(
            PRNG_key=PRNG_key,
            index=jnp.asarray([0]),
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
    ) -> tuple[RenderObject, Array | None]:

        kwargs = {
            "mode": self.render_mode,
            "env_index": 0,
            "agent_index_focus": agent_index_focus,
            "visualize_when_rgb": visualize_when_rgb,
            **kwargs,
        }
        render_object, rgb_array = self.env.render(
            render_object=render_object, **kwargs
        )
        return render_object, rgb_array
