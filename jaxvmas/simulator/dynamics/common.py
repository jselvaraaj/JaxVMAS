#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from __future__ import annotations

import abc
from abc import ABC
from typing import TYPE_CHECKING

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Int, jaxtyped

from jaxvmas.equinox_utils import PyTreeNode

if TYPE_CHECKING:
    from jaxvmas.simulator.core.agent import Agent

batch_axis_dim = "batch_axis_dim"


class Dynamics(PyTreeNode, ABC):

    @jaxtyped(typechecker=beartype)
    def reset(
        self,
        index: Int[Array, f"{batch_axis_dim}"] | Int[Array, ""] = jnp.asarray(-1),
    ) -> "Dynamics":
        return self

    @jaxtyped(typechecker=beartype)
    def check_and_process_action(self, agent: "Agent") -> tuple["Dynamics", "Agent"]:
        action = agent.action.u
        if action.shape[1] < self.needed_action_size:
            raise ValueError(
                f"Agent action size {action.shape[1]} is less than the required dynamics action size {self.needed_action_size}"
            )
        self, agent = self.process_action(agent)
        return self, agent

    @property
    @abc.abstractmethod
    def needed_action_size(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def process_action(self, agent: "Agent") -> tuple["Dynamics", "Agent"]:
        raise NotImplementedError
