#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import jax.numpy as jnp
from flax import struct
from jaxtyping import Array, Float

from jaxvmas.simulator.core import Agent, Entity, World
from jaxvmas.simulator.dynamics.holonomic import Holonomic
from jaxvmas.simulator.scenario import BaseScenario
from jaxvmas.simulator.utils import ScenarioUtils

# Type dimensions
batch = "batch"
n_agents = "n_agents"
n_targets = "n_targets"


@struct.dataclass
class NavigationState:
    """Dynamic state for Navigation scenario."""

    agent_targets: Float[Array, f"{batch} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1))
    )
    target_reached: Float[Array, f"{batch} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1))
    )


class Navigation(BaseScenario):
    """
    Scenario where agents must navigate to their assigned targets while avoiding collisions.
    Each agent has a specific target and must reach it while avoiding other agents.
    """

    def __init__(self):
        super().__init__()
        self.n_agents = 4
        self.collision_penalty = 1.0
        self.agent_size = 0.05
        self.target_size = 0.05
        self.target_reached_distance = 0.15
        self.state = None

    def make_world(self, batch_dim: int, **kwargs) -> World:
        world = World(batch_dim=batch_dim, dim_p=2)

        # Add agents
        for i in range(self.n_agents):
            agent = Agent(name=f"agent_{i}", dynamics=Holonomic())
            agent.color = jnp.array([0.35, 0.35, 0.85])
            agent.collision_penalty = True
            agent.size = self.agent_size
            world.add_agent(agent)

        # Add targets (landmarks)
        for i in range(self.n_agents):
            target = Entity(name=f"target_{i}")
            target.collide = False
            target.movable = False
            target.size = self.target_size
            target.color = jnp.array([0.25, 0.85, 0.25])
            world.add_landmark(target)

        # Initialize scenario state
        self.state = NavigationState(
            agent_targets=jnp.zeros((batch_dim, self.n_agents)),
            target_reached=jnp.zeros((batch_dim, self.n_agents)),
        )

        return world

    def reset_world_at(self, env_index: int | None):
        # Random agent positions
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents,
            self.world,
            env_index,
            min_dist_between_entities=0.15,
            x_bounds=(-1, 1),
            y_bounds=(-1, 1),
        )

        # Random target positions
        ScenarioUtils.spawn_entities_randomly(
            self.world.landmarks,
            self.world,
            env_index,
            min_dist_between_entities=0.15,
            x_bounds=(-1, 1),
            y_bounds=(-1, 1),
        )

        # Reset target assignments and reached status
        batch_size = self.world.batch_dim if env_index is None else 1
        self.state = NavigationState(
            agent_targets=jnp.zeros((batch_size, self.n_agents)),
            target_reached=jnp.zeros((batch_size, self.n_agents)),
        )

    def reward(self, agent: Agent) -> Float[Array, f"{batch}"]:
        agent_idx = self.world.agents.index(agent)
        agent_pos = agent.state.pos
        target_pos = self.world.landmarks[agent_idx].state.pos

        # Distance to target
        dist_to_target = jnp.linalg.norm(agent_pos - target_pos, axis=-1)

        # Base reward is negative distance to target
        reward = -dist_to_target

        # Bonus for reaching target
        target_reached = dist_to_target < self.target_reached_distance
        reward = jnp.where(target_reached, reward + 1.0, reward)

        # Update target reached status
        self.state = self.state.replace(
            target_reached=self.state.target_reached.at[:, agent_idx].set(
                target_reached
            )
        )

        # Penalty for collisions
        if agent.collision_penalty:
            for other in self.world.agents:
                if other is agent:
                    continue
                collision_dist = agent.size + other.size
                dist = jnp.linalg.norm(agent.state.pos - other.state.pos, axis=-1)
                reward = jnp.where(
                    dist < collision_dist, reward - self.collision_penalty, reward
                )

        return reward

    def observation(self, agent: Agent) -> Float[Array, f"{batch} ..."]:
        agent_idx = self.world.agents.index(agent)

        # Get target position relative to agent
        target_pos = self.world.landmarks[agent_idx].state.pos - agent.state.pos

        # Get positions of other agents relative to this agent
        other_pos = []
        for other in self.world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.pos - agent.state.pos)

        # Stack observations
        obs = jnp.concatenate(
            [agent.state.pos]  # Own position
            + [agent.state.vel]  # Own velocity
            + [target_pos]  # Relative position to target
            + other_pos,  # Relative positions to other agents
            axis=-1,
        )
        return obs

    def done(self) -> Float[Array, f"{batch}"]:
        # Episode is done when all agents have reached their targets
        return jnp.all(self.state.target_reached, axis=-1)
