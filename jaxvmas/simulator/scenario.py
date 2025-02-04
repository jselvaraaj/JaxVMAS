#  Copyright (c) 2022-2025.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
from abc import ABC, abstractmethod

import jax.numpy as jnp
from jaxtyping import Array

from jaxvmas.simulator.core import Agent, World
from jaxvmas.simulator.rendering import Geom
from jaxvmas.simulator.utils import (
    AGENT_INFO_TYPE,
    AGENT_OBS_TYPE,
    AGENT_REWARD_TYPE,
    INITIAL_VIEWER_SIZE,
    VIEWER_DEFAULT_ZOOM,
)

# Type dimensions
batch = "batch"
pos = "pos"
comm = "comm"
action = "action"
info = "info"


class BaseScenario(ABC):
    """Base class for scenarios.

    This is the class that scenarios inherit from.

    The methods that are **compulsory to instantiate** are:

    - :class:`make_world`
    - :class:`reset_world_at`
    - :class:`observation`
    - :class:`reward`

    The methods that are **optional to instantiate** are:

    - :class:`info`
    - :class:`extra_render`
    - :class:`process_action`
    - :class:`pre_step`
    - :class:`post_step`

    """

    def __init__(self):
        """Do not override."""
        self._world = None
        self.viewer_size = INITIAL_VIEWER_SIZE
        """The size of the rendering viewer window. This can be changed in the :class:`~make_world` function. """
        self.viewer_zoom = VIEWER_DEFAULT_ZOOM
        """The zoom of the rendering camera (a lower value means more zoom). This can be changed in the :class:`~make_world` function. """
        self.render_origin = (0.0, 0.0)
        """The origin of the rendering camera when ``agent_index_to_focus`` is None in the ``render()`` arguments. This can be changed in the :class:`~make_world` function. """
        self.plot_grid = False
        """Whether to plot a grid in the scenario rendering background. This can be changed in the :class:`~make_world` function. """
        self.grid_spacing = 0.1
        """If :class:`~plot_grid`, the distance between lines in the background grid. This can be changed in the :class:`~make_world` function. """
        self.visualize_semidims = True
        """Whether to display boundaries in dimension-limited environment. This can be changed in the :class:`~make_world` function. """

    @property
    def world(self) -> World:
        """The :class:`~jaxvmas.simulator.core.World` associated with this scenario."""
        assert (
            self._world is not None
        ), "You first need to set `self._world` in the `make_world` method"
        return self._world

    def env_make_world(self, batch_dim: int, **kwargs) -> World:
        # Do not override
        self._world = self.make_world(batch_dim, **kwargs)
        return self._world

    def env_reset_world_at(self, env_index: int | None):
        # Do not override
        self.world.reset(env_index)
        self.reset_world_at(env_index)

    def env_process_action(self, agent: Agent):
        # Do not override
        if agent.action_script is not None:
            agent.action_callback(self.world)
        # Customizable action processor
        self.process_action(agent)
        agent.dynamics.check_and_process_action()

    @abstractmethod
    def make_world(self, batch_dim: int, **kwargs) -> World:
        """
        This function needs to be implemented when creating a scenario.
        In this function the user should instantiate the world and insert agents and landmarks in it.

        Args:
            batch_dim (int): the number of vectorized environments.
            kwargs (dict, optional): named arguments passed from environment creation

        Returns:
            :class:`~jaxvmas.simulator.core.World` : the :class:`~jaxvmas.simulator.core.World`
            instance which is automatically set in :class:`~world`.
        """
        raise NotImplementedError()

    @abstractmethod
    def reset_world_at(self, env_index: int | None):
        """Resets the world at the specified env_index.

        When a ``None`` index is passed, the world should make a vectorized (batched) reset.
        The ``entity.set_x()`` methods already have this logic integrated and will perform
        batched operations when index is ``None``.

        When this function is called, all entities have already had their state reset to zeros according to the ``env_index``.
        In this function you should change the values of the reset states according to your task.

        Args:
            env_index (int, optional): index of the environment to reset. If ``None`` a vectorized reset should be performed.
        """
        raise NotImplementedError()

    @abstractmethod
    def observation(self, agent: Agent) -> AGENT_OBS_TYPE:
        """This function computes the observations for ``agent`` in a vectorized way.

        The returned array should contain the observations for ``agent`` in all envs and should have
        shape ``(self.world.batch_dim, n_agent_obs)``, or be a dict with leaves following that shape.

        Args:
            agent (Agent): the agent to compute the observations for

        Returns:
             Union[Array, Dict[str, Array]]: the observation
        """
        raise NotImplementedError()

    @abstractmethod
    def reward(self, agent: Agent) -> AGENT_REWARD_TYPE:
        """This function computes the reward for ``agent`` in a vectorized way.

        The returned array should contain the reward for ``agent`` in all envs and should have
        shape ``(self.world.batch_dim)`` and dtype ``float32``.

        Args:
            agent (Agent): the agent to compute the reward for

        Returns:
             Array: reward array of shape ``(self.world.batch_dim)``
        """
        raise NotImplementedError()

    def done(self) -> Array:
        """This function computes the done flag for each env in a vectorized way.

        The returned array should contain the ``done`` for all envs and should have
        shape ``(n_envs)`` and dtype ``bool``.

        By default, this function returns all ``False`` s.

        The scenario can still be done if ``max_steps`` has been set at environment construction.

        Returns:
            Array: done array of shape ``(self.world.batch_dim)``
        """
        return jnp.zeros(self.world.batch_dim, dtype=bool)

    def info(self, agent: Agent) -> AGENT_INFO_TYPE:
        """This function computes the info dict for ``agent`` in a vectorized way.

        The returned dict should have a key for each info of interest and the corresponding value should
        be an array of shape ``(n_envs, info_size)``

        By default this function returns an empty dictionary.

        Args:
            agent (Agent): the agent to compute the info for

        Returns:
             Dict[str, Array]: the info
        """
        return {}

    def extra_render(self, env_index: int = 0) -> list[Geom]:
        """
        This function facilitates additional user/scenario-level rendering for a specific environment index.

        The returned list is a list of geometries. It is the user's responsibility to set attributes such as color,
        position and rotation.

        Args:
            env_index (int, optional): index of the environment to render. Defaults to ``0``.

        Returns: A list of geometries to render for the current time step.
        """
        return []

    def process_action(self, agent: Agent):
        """This function can be overridden to process the agent actions before the simulation step.

        For example here you can manage additional actions before passing them to the dynamics.

        Args:
            agent (Agent): the agent process the action of
        """
        return

    def pre_step(self):
        """This function can be overridden to perform any computation that has to happen before the simulation step.
        Its intended use is for computation that has to happen only once before the simulation step has occurred.
        """
        return

    def post_step(self):
        """This function can be overridden to perform any computation that has to happen after the simulation step.
        Its intended use is for computation that has to happen only once after the simulation step has occurred.
        """
        return
