#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jaxvmas.simulator.core import Agent
from jaxvmas.simulator.dynamics.common import Dynamics


class Holonomic(Dynamics):
    @property
    def needed_action_size(self) -> int:
        return 2

    def process_action(self, agent: "Agent") -> tuple["Holonomic", "Agent"]:
        force = agent.action.u[:, : self.needed_action_size]
        agent = agent.replace(
            state=agent.state.replace(
                force=force,
            )
        )
        return self, agent
