#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import jaxtyped

if TYPE_CHECKING:
    from jaxvmas.simulator.core.agent import Agent

from jaxvmas.simulator.dynamics.common import Dynamics
from jaxvmas.simulator.utils import JaxUtils, X


class Forward(Dynamics):
    @property
    def needed_action_size(self) -> int:
        return 1

    @jaxtyped(typechecker=beartype)
    def process_action(self, agent: "Agent") -> tuple["Forward", "Agent"]:
        force = jnp.zeros((agent.batch_dim, 2))
        force = force.at[:, X].set(agent.action.u[:, 0])
        force = JaxUtils.rotate_vector(force, agent.state.rot)
        agent = agent.replace(
            state=agent.state.replace(
                force=force,
            )
        )
        return self, agent
