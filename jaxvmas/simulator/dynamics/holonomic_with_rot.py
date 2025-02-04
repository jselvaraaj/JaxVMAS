#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from jaxvmas.simulator.dynamics.common import Dynamics


class HolonomicWithRotation(Dynamics):
    @property
    def needed_action_size(self) -> int:
        return 3

    def process_action(self):
        self.agent.state.force = self.agent.action.u[:, :2]
        self.agent.state.torque = self.agent.action.u[:, 2].unsqueeze(-1)
