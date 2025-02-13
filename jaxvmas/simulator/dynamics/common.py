#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import abc
from abc import ABC
from typing import TYPE_CHECKING

from jaxtyping import Array

from jaxvmas.equinox_utils import PyTreeNode

if TYPE_CHECKING:
    from jaxvmas.simulator.core import Agent


class Dynamics(PyTreeNode, ABC):

    def reset(self, index: Array | int | None = None) -> "Dynamics":
        return self

    def check_and_process_action(self, agent: "Agent") -> tuple["Dynamics", "Agent"]:
        action = agent.action.u
        if action.shape[1] < self.needed_action_size:
            raise ValueError(
                f"Agent action size {action.shape[1]} is less than the required dynamics action size {self.needed_action_size}"
            )
        agent = self.process_action(agent)
        return self, agent

    @property
    @abc.abstractmethod
    def needed_action_size(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def process_action(self, agent: "Agent") -> tuple["Dynamics", "Agent"]:
        raise NotImplementedError
