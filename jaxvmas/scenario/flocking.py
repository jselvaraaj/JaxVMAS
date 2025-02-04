#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import jax
import jax.numpy as jnp
from flax import struct
from jaxtyping import Array, Float

from jaxvmas.simulator.core import Agent, World
from jaxvmas.simulator.dynamics.holonomic import Holonomic
from jaxvmas.simulator.scenario import BaseScenario
from jaxvmas.simulator.utils import ScenarioUtils

# Type dimensions
batch = "batch"
n_agents = "n_agents"


@struct.dataclass
class FlockingState:
    """Dynamic state for Flocking scenario."""

    desired_velocities: Float[Array, f"{batch} {n_agents} 2"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1, 2))
    )


class Flocking(BaseScenario):
    """
    Scenario where agents must move together as a flock.
    Agents are rewarded for:
    1. Moving in the same direction (alignment)
    2. Staying close to each other (cohesion)
    3. Not getting too close (separation)
    """

    def __init__(self):
        super().__init__()
        self.n_agents = 4
        self.agent_size = 0.05
        self.desired_velocity = 0.5
        self.collision_penalty = 1.0
        self.cohesion_distance = 0.5
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

        # Initialize scenario state with random desired velocities
        key = kwargs.get("key", jax.random.PRNGKey(0))
        angles = jax.random.uniform(key, (batch_dim, 1), minval=0, maxval=2 * jnp.pi)
        desired_vels = jnp.stack(
            [
                jnp.cos(angles) * self.desired_velocity,
                jnp.sin(angles) * self.desired_velocity,
            ],
            axis=-1,
        )
        desired_vels = jnp.repeat(desired_vels, self.n_agents, axis=1)

        self.state = FlockingState(desired_velocities=desired_vels)

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

        # Reset velocities to zero
        for agent in self.world.agents:
            agent.state.vel = jnp.zeros_like(agent.state.vel)

    def reward(self, agent: Agent) -> Float[Array, f"{batch}"]:
        agent_idx = self.world.agents.index(agent)
        reward = jnp.zeros(self.world.batch_dim)

        # Alignment reward: velocity matches flock's average velocity
        all_vels = jnp.stack(
            [a.state.vel for a in self.world.agents]
        )  # [n_agents, batch, 2]
        avg_vel = jnp.mean(all_vels, axis=0)  # [batch, 2]
        alignment_reward = -jnp.linalg.norm(agent.state.vel - avg_vel, axis=-1)
        reward += alignment_reward

        # Cohesion reward: stay close to other agents
        all_pos = jnp.stack(
            [a.state.pos for a in self.world.agents]
        )  # [n_agents, batch, 2]
        avg_pos = jnp.mean(all_pos, axis=0)  # [batch, 2]
        dist_to_center = jnp.linalg.norm(agent.state.pos - avg_pos, axis=-1)
        cohesion_reward = -jnp.where(
            dist_to_center > self.cohesion_distance,
            dist_to_center - self.cohesion_distance,
            0.0,
        )
        reward += cohesion_reward

        # Separation reward: don't get too close to other agents
        if agent.collision_penalty:
            for other in self.world.agents:
                if other is agent:
                    continue
                collision_dist = agent.size + other.size
                dist = jnp.linalg.norm(agent.state.pos - other.state.pos, axis=-1)
                reward = jnp.where(
                    dist < collision_dist, reward - self.collision_penalty, reward
                )

        # Velocity matching reward: try to maintain desired velocity
        vel_diff = jnp.linalg.norm(
            agent.state.vel - self.state.desired_velocities[:, agent_idx], axis=-1
        )
        reward -= vel_diff

        return reward

    def observation(self, agent: Agent) -> Float[Array, f"{batch} ..."]:
        agent_idx = self.world.agents.index(agent)
        # Get positions and velocities of other agents relative to this agent
        other_pos = []
        other_vel = []
        for other in self.world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.pos - agent.state.pos)
            other_vel.append(other.state.vel - agent.state.vel)

        # Stack observations
        obs = jnp.concatenate(
            [agent.state.pos]  # Own position
            + [agent.state.vel]  # Own velocity
            + other_pos  # Relative positions to other agents
            + other_vel,  # Relative velocities to other agents
            axis=-1,
        )
        return obs
