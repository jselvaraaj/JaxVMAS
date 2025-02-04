#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

dim_batch = "batch"
dim_features = "features"
dim_actions = "actions"


class BaseHeuristicPolicy(ABC):
    def __init__(self, continuous_action: bool):
        self.continuous_actions = continuous_action

    @abstractmethod
    def compute_action(
        self,
        observation: Float[Array, f"{dim_batch} {dim_features}"],
        u_range: float,
        key: jax.Array | None = None,
    ) -> Float[Array, f"{dim_batch} {dim_actions}"]:
        raise NotImplementedError


class RandomPolicy(BaseHeuristicPolicy):
    def compute_action(
        self,
        observation: Float[Array, f"{dim_batch} {dim_features}"],
        u_range: float,
        key: jax.Array | None = None,
    ) -> Float[Array, f"{dim_batch} {dim_actions}"]:
        if key is None:
            key = jax.random.PRNGKey(0)
        n_envs = observation.shape[0]
        key, subkey = jax.random.split(key)
        random_actions = jax.random.normal(subkey, (n_envs, 2))
        return jnp.clip(random_actions, -u_range, u_range)
