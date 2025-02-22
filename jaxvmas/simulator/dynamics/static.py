#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jaxvmas.simulator.core.agent import Agent
from beartype import beartype
from jaxtyping import jaxtyped

from jaxvmas.simulator.dynamics.common import Dynamics


class Static(Dynamics):
    @property
    def needed_action_size(self) -> int:
        return 0

    @jaxtyped(typechecker=beartype)
    def process_action(self, agent: "Agent") -> tuple["Static", "Agent"]:
        return self, agent
