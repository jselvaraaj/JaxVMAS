#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jaxvmas.simulator.core import Agent
from jaxvmas.simulator.dynamics.common import Dynamics


class HolonomicWithRotation(Dynamics):
    @property
    def needed_action_size(self) -> int:
        return 3

    def process_action(self, agent: "Agent") -> tuple["HolonomicWithRotation", "Agent"]:
        force = agent.action.u[:, :2]
        torque = agent.action.u[:, 2][..., None]
        agent = agent.replace(
            state=agent.state.replace(
                force=force,
                torque=torque,
            )
        )
        return self, agent
