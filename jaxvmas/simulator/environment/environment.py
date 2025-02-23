#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

"""
JAX-based vectorized multi-agent environment implementation.
This version separates static (stored in self) and dynamic (stored in EnvironmentState)
variables and threads the RNG key through every functional call.
"""


import math
from ctypes import byref

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Callable
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, jaxtyped

import jaxvmas
from jaxvmas.simulator.core.agent import Agent
from jaxvmas.simulator.core.jax_vectorized_object import JaxVectorizedObject
from jaxvmas.simulator.core.world import World
from jaxvmas.simulator.environment.jaxgym.spaces import (
    Box,
    Dict,
    Discrete,
    MultiDiscrete,
    Space,
    Tuple,
)
from jaxvmas.simulator.rendering import Image, TextLine, Viewer
from jaxvmas.simulator.scenario import BaseScenario
from jaxvmas.simulator.utils import (
    AGENT_OBS_TYPE,
    ALPHABET,
    OBS_TYPE,
    SCENARIO_PYTREE_TYPE,
    X,
    Y,
)

batch_axis_dim = "batch_axis_dim"
BATCHED_ARRAY_TYPE = (
    Float[Array, f"{batch_axis_dim} ..."]
    | Int[Array, f"{batch_axis_dim} ..."]
    | Bool[Array, f"{batch_axis_dim} ..."]
)


class RenderObject:
    def __init__(
        self,
        viewer: Viewer | None = None,
        headless: bool | None = None,
        visible_display: bool | None = None,
        text_lines: list[TextLine] | None = None,
    ):

        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "runtime.vectorized": True,
        }
        self.viewer = viewer
        self.headless = headless
        self.visible_display = visible_display
        self.text_lines = text_lines

    def _init_rendering(self, viewer_size, dim_c, agents) -> None:
        from jaxvmas.simulator import rendering

        viewer = rendering.Viewer(*viewer_size, visible=self.visible_display)
        text_lines = []
        idx = 0
        if dim_c > 0:
            for agent in agents:
                if not agent.silent:
                    text_line = rendering.TextLine(y=idx * 40)
                    viewer.geoms.append(text_line)
                    text_lines.append(text_line)
                    idx += 1

        self.viewer = viewer
        self.text_lines = text_lines

    def plot_boundary(self, x_semidim, y_semidim) -> None:
        # include boundaries in the rendering if the environment is dimension-limited
        if x_semidim is not None or y_semidim is not None:
            from jaxvmas.simulator.rendering import Line
            from jaxvmas.simulator.utils import Color

            # set a big value for the cases where the environment is dimension-limited only in one coordinate
            infinite_value = 100

            x_semi = x_semidim if x_semidim is not None else infinite_value
            y_semi = y_semidim if y_semidim is not None else infinite_value

            # set the color for the boundary line
            color = Color.GRAY.value

            # Define boundary points based on whether world semidims are provided
            if (
                x_semidim is not None and y_semidim is not None
            ) or y_semidim is not None:
                boundary_points = [
                    (-x_semi, y_semi),
                    (x_semi, y_semi),
                    (x_semi, -y_semi),
                    (-x_semi, -y_semi),
                ]
            else:
                boundary_points = [
                    (-x_semi, y_semi),
                    (-x_semi, -y_semi),
                    (x_semi, y_semi),
                    (x_semi, -y_semi),
                ]

            # Create lines by connecting points
            for i in range(
                0,
                len(boundary_points),
                (1 if (x_semidim is not None and y_semidim is not None) else 2),
            ):
                start = boundary_points[i]
                end = boundary_points[(i + 1) % len(boundary_points)]
                line = Line(start, end, width=0.7)
                line.set_color(*color)
                self.viewer.add_onetime(line)

    def _set_agent_comm_messages(
        self, env_index: int, dim_c: int, continuous_actions: bool, agents: list[Agent]
    ) -> "Environment":
        text_lines = [self.text_lines[i] for i in range(len(self.text_lines))]
        # Render comm messages
        if dim_c > 0:
            idx = 0
            for agent in agents:
                if not agent.silent:
                    assert (
                        agent.state.c is not None
                    ), "Agent has no comm state but it should"
                    if continuous_actions:
                        word = (
                            "["
                            + ",".join(
                                [f"{comm:.2f}" for comm in agent.state.c[env_index]]
                            )
                            + "]"
                        )
                    else:
                        word = ALPHABET[jnp.argmax(agent.state.c[env_index]).item()]

                    message = agent.name + " sends " + word + "   "
                    text_lines[idx].set_text(message)
                    idx += 1
        self.text_lines = text_lines

    def plot_function(
        self, f, precision, plot_range, cmap_range, cmap_alpha, cmap_name
    ) -> Image:
        from jaxvmas.simulator.rendering import render_function_util

        if plot_range is None:
            assert self.viewer.bounds is not None, "Set viewer bounds before plotting"
            x_min, x_max, y_min, y_max = self.viewer.bounds.tolist()
            plot_range = (
                [x_min - precision, x_max - precision],
                [
                    y_min - precision,
                    y_max + precision,
                ],
            )

        geom = render_function_util(
            f=f,
            precision=precision,
            plot_range=plot_range,
            cmap_range=cmap_range,
            cmap_alpha=cmap_alpha,
            cmap_name=cmap_name,
        )
        return geom


# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class Environment(JaxVectorizedObject):
    scenario: BaseScenario
    num_envs: int
    max_steps: int | float
    continuous_actions: bool
    dict_spaces: bool
    clamp_action: bool
    grad_enabled: bool
    terminated_truncated: bool
    multidiscrete_actions: bool
    action_space: Space
    observation_space: Space

    steps: Array

    @classmethod
    @jaxtyped(typechecker=beartype)
    @chex.assert_max_traces(0)
    def create(
        cls,
        scenario: BaseScenario,
        PRNG_key: PRNGKeyArray,
        num_envs: int = 32,
        max_steps: int | float = jnp.inf,
        continuous_actions: bool = True,
        dict_spaces: bool = False,
        multidiscrete_actions: bool = False,
        clamp_actions: bool = False,
        grad_enabled: bool = False,
        terminated_truncated: bool = False,
        **kwargs,
    ):
        if multidiscrete_actions:
            assert (
                not continuous_actions
            ), "When asking for multidiscrete_actions, make sure continuous_actions=False"

        scenario = scenario.env_make_world(num_envs, **kwargs)

        max_steps = max_steps
        continuous_actions = continuous_actions
        dict_spaces = dict_spaces
        clamp_action = clamp_actions
        grad_enabled = grad_enabled
        terminated_truncated = terminated_truncated

        steps = jnp.zeros(num_envs)
        self = cls(
            batch_dim=num_envs,
            scenario=scenario,
            num_envs=num_envs,
            max_steps=max_steps,
            continuous_actions=continuous_actions,
            dict_spaces=dict_spaces,
            clamp_action=clamp_action,
            grad_enabled=grad_enabled,
            terminated_truncated=terminated_truncated,
            multidiscrete_actions=multidiscrete_actions,
            action_space=None,
            observation_space=None,
            steps=steps,
        )

        PRNG_key, PRNG_key_reset = jax.random.split(PRNG_key)

        self, observations = self.reset(PRNG_key=PRNG_key_reset)

        # configure spaces
        multidiscrete_actions = multidiscrete_actions
        action_space = self.get_action_space()
        observation_space = self.get_observation_space(observations)

        self = self.replace(
            action_space=action_space,
            observation_space=observation_space,
        )

        return self

    @property
    def world(self) -> World:
        return self.scenario.world

    @property
    def n_agents(self) -> int:
        return len(self.world.policy_agents)

    @property
    def agents(self) -> list[Agent]:
        return self.world.policy_agents

    def replace(self, **kwargs) -> "Environment":
        if "world" in kwargs:
            world = kwargs.pop("world")
            kwargs["scenario"] = self.scenario.replace(world=world)
        elif "agents" in kwargs:
            agents = kwargs.pop("agents")
            kwargs["scenario"] = self.scenario.replace(
                world=self.world.replace(policy_agents=agents)
            )

        return super().replace(**kwargs)

    @jaxtyped(typechecker=beartype)
    def reset(
        self,
        PRNG_key: PRNGKeyArray,
        return_observations: bool = True,
        return_info: bool = False,
        return_dones: bool = False,
    ) -> tuple["Environment", SCENARIO_PYTREE_TYPE]:
        """
        Resets the environment in a vectorized way
        Returns observations for all envs and agents
        """
        # reset world
        scenario = self.scenario.env_reset_world_at(
            PRNG_key=PRNG_key, env_index=jnp.asarray(-1)
        )
        self = self.replace(scenario=scenario, steps=jnp.zeros(self.num_envs))

        result = self.get_from_scenario(
            get_observations=return_observations,
            get_infos=return_info,
            get_rewards=False,
            get_dones=return_dones,
        )
        return self, result[0] if result and len(result) == 1 else result

    @jaxtyped(typechecker=beartype)
    def reset_at(
        self,
        PRNG_key: PRNGKeyArray,
        index: Int[Array, f"{batch_axis_dim}"] | Int[Array, ""],
        return_observations: bool = True,
        return_info: bool = False,
        return_dones: bool = False,
    ) -> tuple["Environment", SCENARIO_PYTREE_TYPE]:
        """
        Resets the environment at index
        Returns observations for all agents in that environment
        """
        self._check_batch_index(index)
        scenario = self.scenario.env_reset_world_at(PRNG_key=PRNG_key, env_index=index)
        self = self.replace(scenario=scenario, steps=self.steps.at[index].set(0))

        result = self.get_from_scenario(
            get_observations=return_observations,
            get_infos=return_info,
            get_rewards=False,
            get_dones=return_dones,
        )

        return self, result[0] if result and len(result) == 1 else result

    @jaxtyped(typechecker=beartype)
    def get_from_scenario(
        self,
        get_observations: bool,
        get_rewards: bool,
        get_infos: bool,
        get_dones: bool,
        dict_agent_names: bool | None = None,
    ) -> SCENARIO_PYTREE_TYPE:
        if not get_infos and not get_dones and not get_rewards and not get_observations:
            return
        if dict_agent_names is None:
            dict_agent_names = self.dict_spaces

        obs = rewards = infos = terminated = truncated = dones = None

        if get_observations:
            obs = {} if dict_agent_names else []
        if get_rewards:
            rewards = {} if dict_agent_names else []
        if get_infos:
            infos = {} if dict_agent_names else []

        if get_rewards:
            for agent in self.agents:
                reward = self.scenario.reward(agent)
                if dict_agent_names:
                    rewards.update({agent.name: reward})
                else:
                    rewards.append(reward)
        if get_observations:
            for agent in self.agents:
                observation = self.scenario.observation(agent)
                if dict_agent_names:
                    obs.update({agent.name: observation})
                else:
                    obs.append(observation)
        if get_infos:
            for agent in self.agents:
                info = self.scenario.info(agent)
                if dict_agent_names:
                    infos.update({agent.name: info})
                else:
                    infos.append(info)

        if self.terminated_truncated:
            if get_dones:
                terminated, truncated = self.done()
            result = [obs, rewards, terminated, truncated, infos]
        else:
            if get_dones:
                dones = self.done()
            result = [obs, rewards, dones, infos]

        return [data for data in result if data is not None]

    @eqx.filter_jit
    @jaxtyped(typechecker=beartype)
    def step(
        self,
        PRNG_key: PRNGKeyArray,
        actions: list[BATCHED_ARRAY_TYPE] | dict[str, BATCHED_ARRAY_TYPE],
    ) -> tuple["Environment", SCENARIO_PYTREE_TYPE]:
        """Performs a vectorized step on all sub environments using `actions`.
        Args:
            actions: Is a list on len 'self.n_agents' of which each element is a torch.Tensor of shape
            '(self.num_envs, action_size_of_agent)'.
        Returns:
            obs: List on len 'self.n_agents' of which each element is a torch.Tensor
                 of shape '(self.num_envs, obs_size_of_agent)'
            rewards: List on len 'self.n_agents' of which each element is a torch.Tensor of shape '(self.num_envs)'
            dones: Tensor of len 'self.num_envs' of which each element is a bool
            infos : List on len 'self.n_agents' of which each element is a dictionary for which each key is a metric
                    and the value is a tensor of shape '(self.num_envs, metric_size_per_agent)'

        Examples:
            >>> import vmas
            >>> env = vmas.make_env(
            ...     scenario="waterfall",  # can be scenario name or BaseScenario class
            ...     num_envs=32,
            ...     device="cpu",  # Or "cuda" for GPU
            ...     continuous_actions=True,
            ...     max_steps=100,  # Defines the horizon. jnp.nan is infinite horizon.
            ...     seed=None,  # Seed of the environment
            ...     n_agents=3,  # Additional arguments you want to pass to the scenario
            ... )
            >>> obs = env.reset()
            >>> for _ in range(10):
            ...     obs, rews, dones, info = env.step(env.get_random_actions())
        """
        if isinstance(actions, dict):
            actions_dict = actions
            _actions = []
            for agent in self.agents:
                try:
                    _actions.append(actions_dict[agent.name])
                except KeyError:
                    raise AssertionError(
                        f"Agent '{agent.name}' not contained in action dict"
                    )
            assert (
                len(actions_dict) == self.n_agents
            ), f"Expecting actions for {self.n_agents}, got {len(actions_dict)} actions"
            actions = _actions

        assert (
            len(actions) == self.n_agents
        ), f"Expecting actions for {self.n_agents}, got {len(actions)} actions"
        for i in range(len(actions)):
            action = actions[i]
            if not isinstance(action, Array):
                action = jnp.asarray(action, dtype=jnp.float32)
            if len(action.shape) == 1:
                action = action[..., None]
            assert (
                action.shape[0] == self.num_envs
            ), f"Actions used in input of env must be of len {self.num_envs}, got {action.shape[0]}"
            assert action.shape[1] == self.get_agent_action_size(self.agents[i]), (
                f"Action for agent {self.agents[i].name} has shape {action.shape[1]},"
                f" but should have shape {self.get_agent_action_size(self.agents[i])}"
            )
            actions[i] = action

        # set action for each agent
        agents = []
        for i, agent in enumerate(self.agents):
            PRNG_key, subkey = jax.random.split(PRNG_key)
            self, agent = self._set_action(subkey, actions[i], agent)
            agents.append(agent)
        self = self.replace(agents=agents)

        agents = []
        print("starting env_process_action")
        # Scenarios can define a custom action processor. This step takes care also of scripted agents automatically
        for agent in self.world.agents:
            PRNG_key, subkey = jax.random.split(PRNG_key)
            scenario, agent = self.scenario.env_process_action(subkey, agent)
            self = self.replace(scenario=scenario)
            agents.append(agent)
            self = self.replace(agents=agents + self.world.agents[len(agents) :])
        print("ending env_process_action")
        # advance world state
        scenario = self.scenario.pre_step()
        self = self.replace(scenario=scenario, world=self.world.step())
        scenario = self.scenario.post_step()
        self = self.replace(scenario=scenario, steps=self.steps + 1)

        return self, self.get_from_scenario(
            get_observations=True,
            get_infos=True,
            get_rewards=True,
            get_dones=True,
        )

    @jaxtyped(typechecker=beartype)
    def done(
        self,
    ) -> BATCHED_ARRAY_TYPE | tuple[BATCHED_ARRAY_TYPE, BATCHED_ARRAY_TYPE]:
        terminated = self.scenario.done()

        truncated = jnp.where(
            jnp.isnan(self.max_steps),
            jnp.zeros_like(terminated),
            self.steps >= self.max_steps,
        )

        if self.terminated_truncated:
            return terminated, truncated
        else:
            return terminated + truncated

    @jaxtyped(typechecker=beartype)
    def get_action_space(self) -> Tuple | Dict:
        if not self.dict_spaces:
            return Tuple([self.get_agent_action_space(agent) for agent in self.agents])
        else:
            return Dict(
                {
                    agent.name: self.get_agent_action_space(agent)
                    for agent in self.agents
                }
            )

    @jaxtyped(typechecker=beartype)
    def get_observation_space(
        self, observations: OBS_TYPE | list[OBS_TYPE] | dict[str, OBS_TYPE]
    ) -> Tuple | Dict:
        if not self.dict_spaces:
            return Tuple(
                [
                    self.get_agent_observation_space(agent, observations[i])
                    for i, agent in enumerate(self.agents)
                ]
            )
        else:
            return Dict(
                {
                    agent.name: self.get_agent_observation_space(
                        agent, observations[agent.name]
                    )
                    for agent in self.agents
                }
            )

    @jaxtyped(typechecker=beartype)
    def get_agent_action_size(self, agent: Agent) -> int:
        if self.continuous_actions:
            return agent.action.action_size + (
                self.world.dim_c if not agent.silent else 0
            )
        elif self.multidiscrete_actions:
            return agent.action_size + (
                1 if not agent.silent and self.world.dim_c != 0 else 0
            )
        else:
            return 1

    @jaxtyped(typechecker=beartype)
    def get_agent_action_space(self, agent: Agent) -> Space:
        if self.continuous_actions:
            return Box(
                low=jnp.asarray(
                    (-agent.action.u_range_jax_array).tolist()
                    + [0] * (self.world.dim_c if not agent.silent else 0),
                    dtype=jnp.float32,
                ),
                high=jnp.asarray(
                    agent.action.u_range_jax_array.tolist()
                    + [1] * (self.world.dim_c if not agent.silent else 0),
                    dtype=jnp.float32,
                ),
                shape=(self.get_agent_action_size(agent),),
            )
        elif self.multidiscrete_actions:
            actions = agent.discrete_action_nvec + (
                [self.world.dim_c] if not agent.silent and self.world.dim_c != 0 else []
            )
            return MultiDiscrete(actions)
        else:
            return Discrete(
                math.prod(agent.discrete_action_nvec)
                * (
                    self.world.dim_c
                    if not agent.silent and self.world.dim_c != 0
                    else 1
                )
            )

    @jaxtyped(typechecker=beartype)
    def get_agent_observation_space(
        self, agent: Agent, obs: AGENT_OBS_TYPE | OBS_TYPE
    ) -> Space:
        if isinstance(obs, Array):
            return Box(
                low=-jnp.float32("inf"),
                high=jnp.float32("inf"),
                shape=obs.shape[1:],
                dtype=jnp.float32,
            )
        elif isinstance(obs, dict):
            return Dict(
                {
                    key: self.get_agent_observation_space(agent, value)
                    for key, value in obs.items()
                }
            )
        else:
            raise NotImplementedError(
                f"Invalid type of observation {obs} for agent {agent.name}"
            )

    # TODO: Use jax primitives for loops
    @jaxtyped(typechecker=beartype)
    def get_random_action(
        self, PRNG_key: PRNGKeyArray, agent: Agent
    ) -> BATCHED_ARRAY_TYPE:
        """Returns a random action for the given agent.

        Args:
            agent (Agent): The agent to get the action for

        Returns:
            torch.tensor: the random actions tensor with shape ``(agent.batch_dim, agent.action_size)``

        """
        if self.continuous_actions:
            actions = []
            for action_index in range(agent.action_size):
                PRNG_key, subkey = jax.random.split(PRNG_key)
                actions.append(
                    jax.random.uniform(
                        key=subkey,
                        shape=(agent.batch_dim,),
                        minval=-agent.action.u_range_jax_array[action_index],
                        maxval=agent.action.u_range_jax_array[action_index],
                    )
                )
            if self.world.dim_c != 0 and not agent.silent:
                # If the agent needs to communicate
                for _ in range(self.world.dim_c):
                    PRNG_key, subkey = jax.random.split(PRNG_key)
                    actions.append(
                        jax.random.uniform(
                            key=subkey,
                            shape=(agent.batch_dim,),
                            minval=0,
                            maxval=1,
                        )
                    )
            action = jnp.stack(actions, axis=-1)
        else:
            action_space = self.get_agent_action_space(agent)
            if self.multidiscrete_actions and isinstance(action_space, MultiDiscrete):
                PRNG_key, *subkey = jax.random.split(
                    PRNG_key, action_space.shape[0] + 1
                )
                actions = [
                    jax.random.randint(
                        key=subkey[action_index],
                        minval=0,
                        maxval=action_space.num_categories[action_index],
                        shape=(agent.batch_dim,),
                    )
                    for action_index in range(action_space.shape[0])
                ]
                action = jnp.stack(actions, axis=-1)
            else:
                if not isinstance(action_space, Discrete):
                    raise ValueError(
                        f"Agent {agent.name} does not have a discrete or multidiscrete action space"
                    )
                PRNG_key, subkey = jax.random.split(PRNG_key)
                action = jax.random.randint(
                    key=subkey,
                    minval=0,
                    maxval=action_space.n,
                    shape=(agent.batch_dim,),
                )
        return action

    @jaxtyped(typechecker=beartype)
    def get_random_actions(self, PRNG_key: PRNGKeyArray) -> list[BATCHED_ARRAY_TYPE]:
        """Returns random actions for all agents that you can feed to :class:`step`

        Returns:
            list[Array]: the random actions for the agents

        Examples:
            >>> import vmas
            >>> env = vmas.make_env(
            ...     scenario="waterfall",  # can be scenario name or BaseScenario class
            ...     num_envs=32,
            ...     device="cpu",  # Or "cuda" for GPU
            ...     continuous_actions=True,
            ...     max_steps=100,  # Defines the horizon. jnp.nan is infinite horizon.
            ...     seed=None,  # Seed of the environment
            ...     n_agents=3,  # Additional arguments you want to pass to the scenario
            ... )
            >>> obs = env.reset()
            >>> for _ in range(10):
            ...     obs, rews, dones, info = env.step(env.get_random_actions())

        """
        PRNG_key, *subkey = jax.random.split(PRNG_key, len(self.agents) + 1)
        return [
            self.get_random_action(subkey[i], self.agents[i])
            for i in range(len(self.agents))
        ]

    @jaxtyped(typechecker=beartype)
    def _check_discrete_action(
        self, action: Array, low: int, high: int, type: str
    ) -> None:
        assert jnp.all(
            (action >= jnp.asarray(low, dtype=jnp.float32))
            * (action < jnp.asarray(high, dtype=jnp.float32))
        ), f"Discrete {type} actions are out of bounds, allowed int range [{low},{high})"

    # set env action for a particular agent
    @jaxtyped(typechecker=beartype)
    def _set_action(
        self, PRNG_key: PRNGKeyArray, action: BATCHED_ARRAY_TYPE, agent: Agent
    ) -> tuple["Environment", Agent]:

        u = jnp.zeros(
            (self.num_envs, agent.action_size),
        )
        agent = agent.replace(action=agent.action.replace(u=u))

        assert action.shape[1] == self.get_agent_action_size(agent), (
            f"Agent {agent.name} has wrong action size, got {action.shape[1]}, "
            f"expected {self.get_agent_action_size(agent)}"
        )
        if self.clamp_action and self.continuous_actions:
            physical_action = action[..., : agent.action_size]
            a_range = jnp.broadcast_to(
                agent.action.u_range_jax_array[None],
                physical_action.shape,
            )
            physical_action = jnp.clip(physical_action, -a_range, a_range)

            if self.world.dim_c > 0 and not agent.silent:  # If comms
                comm_action = action[..., agent.action_size :]
                action = jnp.concatenate(
                    [physical_action, comm_action.clip(0, 1)], axis=-1
                )
            else:
                action = physical_action

        action_index = 0

        if self.continuous_actions:
            physical_action = action[:, action_index : action_index + agent.action_size]
            action_index += self.world.dim_p
            # assert not jnp.any(
            #     jnp.abs(physical_action) > agent.action.u_range_jax_array
            # ), f"Physical actions of agent {agent.name} are out of its range {agent.u_range}"

            physical_action = physical_action.astype(jnp.float32)
            agent = agent.replace(action=agent.action.replace(u=physical_action))

        else:
            if not self.multidiscrete_actions:
                # This bit of code translates the discrete action (taken from a space that
                # is the cartesian product of all action spaces) into a multi discrete action.
                # This is done by iteratively taking the modulo of the action and dividing by the
                # number of actions in the current action space, which treats the action as if
                # it was the "flat index" of the multi-discrete actions. E.g. if we have
                # nvec = [3,2], action 0 corresponds to the actions [0,0],
                # action 1 corresponds to the action [0,1], action 2 corresponds
                # to the action [1,0], action 3 corresponds to the action [1,1], etc.
                flat_action = action.squeeze(-1)
                actions = []
                nvec = list(agent.discrete_action_nvec) + (
                    [self.world.dim_c]
                    if not agent.silent and self.world.dim_c != 0
                    else []
                )
                for i in range(len(nvec)):
                    n = math.prod(nvec[i + 1 :])
                    actions.append(flat_action // n)
                    flat_action = flat_action % n
                action = jnp.stack(actions, axis=-1)

            # Now we have an action with shape [n_envs, action_size+comms_actions]
            for n in agent.discrete_action_nvec:
                physical_action = action[:, action_index]
                self._check_discrete_action(
                    physical_action[..., None],
                    low=0,
                    high=n,
                    type="physical",
                )
                u_max = agent.action.u_range_jax_array[action_index]
                # For odd n we want the first action to always map to u=0, so
                # we swap 0 values with the middle value, and shift the first
                # half of the remaining values by -1.
                if n % 2 != 0:
                    stay = physical_action == 0
                    decrement = (physical_action > 0) & (physical_action <= n // 2)
                    physical_action[stay] = n // 2
                    physical_action[decrement] -= 1
                # We know u must be in [-u_max, u_max], and we know action is
                # in [0, n-1]. Conversion steps: [0, n-1] -> [0, 1] -> [0, 2*u_max] -> [-u_max, u_max]
                # E.g. action 0 -> -u_max, action n-1 -> u_max, action 1 -> -u_max + 2*u_max/(n-1)
                u = u.at[:, action_index].set(
                    (physical_action / (n - 1)) * (2 * u_max) - u_max
                )
                agent = agent.replace(action=agent.action.replace(u=u))

                action_index += 1

        u = agent.action.u * agent.action.u_multiplier_jax_array
        agent = agent.replace(action=agent.action.replace(u=u))

        if agent.action.u_noise > 0:
            PRNG_key, subkey = jax.random.split(PRNG_key)
            noise = (
                jax.random.normal(
                    key=subkey,
                    shape=agent.action.u.shape,
                )
                * agent.u_noise
            )
            agent = agent.replace(action=agent.action.replace(u=agent.action.u + noise))
        if self.world.dim_c > 0 and not agent.silent:
            if not self.continuous_actions:
                comm_action = action[:, action_index:]
                self._check_discrete_action(
                    comm_action, 0, self.world.dim_c, "communication"
                )
                comm_action = comm_action.astype(jnp.int32)
                c = jnp.zeros(
                    (self.num_envs, self.world.dim_c),
                    dtype=jnp.float32,
                )
                c = c.at[:, comm_action].set(1)
                agent = agent.replace(action=agent.action.replace(c=c))
            else:
                comm_action = action[:, action_index:]
                # assert not jnp.any(comm_action > 1) and not jnp.any(
                #     comm_action < 0
                # ), "Comm actions are out of range [0,1]"
                agent = agent.replace(action=agent.action.replace(c=comm_action))
            if agent.c_noise > 0:
                PRNG_key, subkey = jax.random.split(PRNG_key)
                noise = (
                    jax.random.normal(
                        key=subkey,
                        shape=agent.action.c.shape,
                    )
                    * agent.c_noise
                )
                agent = agent.replace(
                    action=agent.action.replace(c=agent.action.c + noise)
                )
        return self, agent

    @jaxtyped(typechecker=beartype)
    @chex.assert_max_traces(0)
    def render(
        self,
        render_object: RenderObject,
        mode="human",
        env_index: Int[Array, f"{batch_axis_dim}"] | Int[Array, ""] = jnp.asarray(0),
        agent_index_focus: int | None = None,
        visualize_when_rgb: bool = False,
        plot_position_function: Callable = None,
        plot_position_function_precision: float = 0.01,
        plot_position_function_range: (
            float
            | tuple[float, float]
            | tuple[tuple[float, float], tuple[float, float]]
            | None
        ) = None,
        plot_position_function_cmap_range: tuple[float, float] | None = None,
        plot_position_function_cmap_alpha: float = 1.0,
        plot_position_function_cmap_name: str | None = "viridis",
    ) -> tuple[RenderObject, Array]:
        """
        Render function for environment using pyglet

        On servers use mode="rgb_array" and set
        ```
        export DISPLAY=':99.0'
        Xvfb :99 -screen 0 1400x900x24 > /dev/null 2>&1 &
        ```

        :param mode: One of human or rgb_array
        :param env_index: Index of the environment to render
        :param agent_index_focus: If specified the camera will stay on the agent with this index.
                                  If None, the camera will stay in the center and zoom out to contain all agents
        :param visualize_when_rgb: Also run human visualization when mode=="rgb_array"
        :param plot_position_function: A function to plot under the rendering.
        The function takes a numpy array with shape (n_points, 2), which represents a set of x,y values to evaluate f over and plot it
        It should output either an array with shape (n_points, 1) which will be plotted as a colormap
        or an array with shape (n_points, 4), which will be plotted as RGBA values
        :param plot_position_function_precision: The precision to use for plotting the function
        :param plot_position_function_range: The position range to plot the function in.
        If float, the range for x and y is (-function_range, function_range)
        If Tuple[float, float], the range for x is (-function_range[0], function_range[0]) and y is (-function_range[1], function_range[1])
        If Tuple[Tuple[float, float], Tuple[float, float]], the first tuple is the x range and the second tuple is the y range
        :param plot_position_function_cmap_range: The range of the cmap in case plot_position_function outputs a single value
        :param plot_position_function_cmap_alpha: The alpha of the cmap in case plot_position_function outputs a single value
        :return: Rgb array or None, depending on the mode
        """
        self._check_batch_index(env_index)
        assert (
            mode in render_object.metadata["render.modes"]
        ), f"Invalid mode {mode} received, allowed modes: {render_object.metadata['render.modes']}"
        if agent_index_focus is not None:
            assert 0 <= agent_index_focus < self.n_agents, (
                f"Agent focus in rendering should be a valid agent index"
                f" between 0 and {self.n_agents}, got {agent_index_focus}"
            )
        shared_viewer = agent_index_focus is None
        aspect_ratio = self.scenario.viewer_size[X] / self.scenario.viewer_size[Y]

        headless = mode == "rgb_array" and not visualize_when_rgb
        # First time rendering
        if render_object.visible_display is None:
            render_object.visible_display = not headless
            render_object.headless = headless
        # All other times headless should be the same
        else:
            assert render_object.visible_display is not headless

        # First time rendering
        if render_object.viewer is None:
            try:
                import pyglet
            except ImportError:
                raise ImportError(
                    "Cannot import pyg;et: you can install pyglet directly via 'pip install pyglet'."
                )

            try:
                # Try to use EGL
                pyglet.lib.load_library("EGL")

                # Only if we have GPUs
                from pyglet.libs.egl import egl, eglext

                num_devices = egl.EGLint()
                eglext.eglQueryDevicesEXT(0, None, byref(num_devices))
                assert num_devices.value > 0

            except (ImportError, AssertionError):
                render_object.headless = False
            pyglet.options["headless"] = render_object.headless

            render_object._init_rendering(
                self.scenario.viewer_size, self.world.dim_c, self.world.agents
            )

        if self.scenario.viewer_zoom <= 0:
            raise ValueError("Scenario viewer zoom must be > 0")
        zoom = self.scenario.viewer_zoom

        if aspect_ratio < 1:
            cam_range = jnp.asarray([zoom, zoom / aspect_ratio])
        else:
            cam_range = jnp.asarray([zoom * aspect_ratio, zoom])

        if shared_viewer:
            # zoom out to fit everyone
            all_poses = jnp.stack(
                [agent.state.pos[env_index] for agent in self.world.agents],
                axis=0,
            )
            max_agent_radius = max(
                [agent.shape.circumscribed_radius() for agent in self.world.agents]
            )
            viewer_size_fit = (
                jnp.stack(
                    [
                        jnp.max(
                            jnp.abs(all_poses[:, X] - self.scenario.render_origin[X])
                        ),
                        jnp.max(
                            jnp.abs(all_poses[:, Y] - self.scenario.render_origin[Y])
                        ),
                    ]
                )
                + 2 * max_agent_radius
            )

            viewer_size = jnp.maximum(
                viewer_size_fit / cam_range,
                jnp.asarray(zoom),
            )
            cam_range *= jnp.max(viewer_size)
            render_object.viewer.set_bounds(
                -cam_range[X] + self.scenario.render_origin[X],
                cam_range[X] + self.scenario.render_origin[X],
                -cam_range[Y] + self.scenario.render_origin[Y],
                cam_range[Y] + self.scenario.render_origin[Y],
            )
        else:
            # update bounds to center around agent
            pos = self.agents[agent_index_focus].state.pos[env_index]
            render_object.viewer.set_bounds(
                pos[X] - cam_range[X],
                pos[X] + cam_range[X],
                pos[Y] - cam_range[Y],
                pos[Y] + cam_range[Y],
            )

        # Render
        if self.scenario.visualize_semidims:
            render_object.plot_boundary(self.world.x_semidim, self.world.y_semidim)

        render_object._set_agent_comm_messages(
            env_index,
            self.world.dim_c,
            self.continuous_actions,
            self.world.agents,
        )

        if plot_position_function is not None:
            render_object.viewer.add_onetime(
                render_object.plot_function(
                    plot_position_function,
                    precision=plot_position_function_precision,
                    plot_range=plot_position_function_range,
                    cmap_range=plot_position_function_cmap_range,
                    cmap_alpha=plot_position_function_cmap_alpha,
                    cmap_name=plot_position_function_cmap_name,
                )
            )

        from jaxvmas.simulator.rendering import Grid

        if self.scenario.plot_grid:
            grid = Grid(spacing=self.scenario.grid_spacing)
            grid.set_color(*jaxvmas.simulator.utils.Color.BLACK.value, alpha=0.3)
            render_object.viewer.add_onetime(grid)

        render_object.viewer.add_onetime_list(self.scenario.extra_render(env_index))

        for entity in self.world.entities:
            render_object.viewer.add_onetime_list(entity.render(env_index=env_index))

        # render to display or array
        return render_object, render_object.viewer.render(
            return_rgb_array=mode == "rgb_array"
        )
