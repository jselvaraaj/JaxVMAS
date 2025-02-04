#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import jax.numpy as jnp
from flax import struct
from jaxtyping import Array, Float

from jaxvmas.simulator.core import Agent, Entity, World
from jaxvmas.simulator.dynamics.holonomic import Holonomic
from jaxvmas.simulator.scenario import BaseScenario

# Type dimensions
batch = "batch"
n_agents = "n_agents"
n_droplets = "n_droplets"


@struct.dataclass
class WaterfallState:
    """Dynamic state for Waterfall scenario."""

    agent_distances: Float[Array, f"{batch} {n_agents} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1, 1))
    )
    droplet_distances: Float[Array, f"{batch} {n_agents} {n_droplets}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1, 1))
    )
    target_distances: Float[Array, f"{batch} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1))
    )
    targets_reached: Float[Array, f"{batch} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1))
    )
    total_reached: Float[Array, f"{batch}"] = struct.field(
        default_factory=lambda: jnp.zeros((1,))
    )
    progress: Float[Array, f"{batch} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1))
    )


class Waterfall(BaseScenario):
    """
    Scenario where agents must navigate through a waterfall of moving droplets
    to reach their targets. Agents must avoid collisions with water droplets
    while efficiently finding paths to their goals.
    """

    def __init__(self):
        super().__init__()
        self.n_agents = 3
        self.n_droplets = 20
        self.agent_size = 0.05
        self.droplet_size = 0.03
        self.target_size = 0.05
        self.collision_penalty = 1.0
        self.droplet_penalty = 2.0
        self.arena_size = 1.0
        self.target_reward = 10.0
        self.progress_reward_scale = 0.1
        self.target_threshold = 0.1
        self.droplet_speed = 0.1
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

        # Add targets
        for i in range(self.n_agents):
            target = Entity(name=f"target_{i}")
            target.collide = False
            target.movable = False
            target.size = self.target_size
            target.color = jnp.array([0.25, 0.85, 0.25])
            world.add_landmark(target)

        # Add water droplets
        for i in range(self.n_droplets):
            droplet = Entity(name=f"droplet_{i}")
            droplet.collide = True
            droplet.movable = True
            droplet.size = self.droplet_size
            droplet.color = jnp.array([0.25, 0.25, 0.85])
            world.add_landmark(droplet)

        # Initialize scenario state
        self.state = WaterfallState(
            agent_distances=jnp.zeros((batch_dim, self.n_agents, self.n_agents)),
            droplet_distances=jnp.zeros((batch_dim, self.n_agents, self.n_droplets)),
            target_distances=jnp.zeros((batch_dim, self.n_agents)),
            targets_reached=jnp.zeros((batch_dim, self.n_agents)),
            total_reached=jnp.zeros(batch_dim),
            progress=jnp.zeros((batch_dim, self.n_agents)),
        )

        return world

    def reset_world_at(self, env_index: int | None):
        batch_size = self.world.batch_dim if env_index is None else 1

        # Place agents on the left side
        for i, agent in enumerate(self.world.agents):
            y_offset = 0.4 * (i - (self.n_agents - 1) / 2)
            pos = jnp.array([-self.arena_size + 0.1, y_offset])
            agent.state.pos = jnp.where(
                env_index is None,
                jnp.tile(pos[None, :], (batch_size, 1)),
                pos,
            )
            agent.state.vel = jnp.zeros_like(agent.state.vel)

        # Place targets on the right side
        for i, target in enumerate(self.world.landmarks[: self.n_agents]):
            y_offset = 0.4 * (i - (self.n_agents - 1) / 2)
            pos = jnp.array([self.arena_size - 0.1, y_offset])
            target.state.pos = jnp.where(
                env_index is None,
                jnp.tile(pos[None, :], (batch_size, 1)),
                pos,
            )

        # Place water droplets in a grid pattern at the top
        for i, droplet in enumerate(self.world.landmarks[self.n_agents :]):
            x = -self.arena_size + 2 * self.arena_size * (i / self.n_droplets)
            y = self.arena_size
            pos = jnp.array([x, y])
            droplet.state.pos = jnp.where(
                env_index is None,
                jnp.tile(pos[None, :], (batch_size, 1)),
                pos,
            )
            # Set initial downward velocity for droplets
            vel = jnp.array([0.0, -self.droplet_speed])
            droplet.state.vel = jnp.where(
                env_index is None,
                jnp.tile(vel[None, :], (batch_size, 1)),
                vel,
            )

        # Reset state
        self.state = WaterfallState(
            agent_distances=jnp.zeros((batch_size, self.n_agents, self.n_agents)),
            droplet_distances=jnp.zeros((batch_size, self.n_agents, self.n_droplets)),
            target_distances=jnp.zeros((batch_size, self.n_agents)),
            targets_reached=jnp.zeros((batch_size, self.n_agents)),
            total_reached=jnp.zeros(batch_size),
            progress=jnp.zeros((batch_size, self.n_agents)),
        )

    def reward(self, agent: Agent) -> Float[Array, f"{batch}"]:
        reward = jnp.zeros(self.world.batch_dim)
        agent_idx = self.world.agents.index(agent)

        # Update distances between agents
        for i, agent_i in enumerate(self.world.agents):
            for j, agent_j in enumerate(self.world.agents):
                if i != j:
                    dist = jnp.linalg.norm(
                        agent_i.state.pos - agent_j.state.pos, axis=-1
                    )
                    self.state = self.state.replace(
                        agent_distances=self.state.agent_distances.at[:, i, j].set(dist)
                    )

        # Update distances to droplets
        for i, droplet in enumerate(self.world.landmarks[self.n_agents :]):
            dist = jnp.linalg.norm(agent.state.pos - droplet.state.pos, axis=-1)
            self.state = self.state.replace(
                droplet_distances=self.state.droplet_distances.at[:, agent_idx, i].set(
                    dist
                )
            )
            # Apply droplet collision penalty
            collision_dist = agent.size + droplet.size
            reward = jnp.where(
                dist < collision_dist,
                reward - self.droplet_penalty,
                reward,
            )

        # Update distances to targets and check for reaching
        target = self.world.landmarks[agent_idx]
        dist_to_target = jnp.linalg.norm(agent.state.pos - target.state.pos, axis=-1)
        self.state = self.state.replace(
            target_distances=self.state.target_distances.at[:, agent_idx].set(
                dist_to_target
            )
        )

        # Check if target is reached
        reached_target = dist_to_target < self.target_threshold
        was_reached = self.state.targets_reached[:, agent_idx] > 0

        # Update target reached status
        self.state = self.state.replace(
            targets_reached=self.state.targets_reached.at[:, agent_idx].set(
                jnp.where(reached_target, 1.0, self.state.targets_reached[:, agent_idx])
            )
        )

        # Update total reached count
        self.state = self.state.replace(
            total_reached=jnp.sum(self.state.targets_reached, axis=-1)
        )

        # Calculate progress (x-position progress towards target)
        old_progress = self.state.progress[:, agent_idx]
        new_progress = agent.state.pos[:, 0] + self.arena_size
        progress_delta = new_progress - old_progress

        # Update progress
        self.state = self.state.replace(
            progress=self.state.progress.at[:, agent_idx].set(new_progress)
        )

        # Reward components
        # 1. Target reaching reward
        reward = jnp.where(
            reached_target & ~was_reached,
            reward + self.target_reward,
            reward,
        )

        # 2. Progress reward
        reward += self.progress_reward_scale * progress_delta

        # 3. Agent collision penalties
        if agent.collision_penalty:
            for other in self.world.agents:
                if other is agent:
                    continue
                collision_dist = agent.size + other.size
                dist = jnp.linalg.norm(agent.state.pos - other.state.pos, axis=-1)
                reward = jnp.where(
                    dist < collision_dist,
                    reward - self.collision_penalty,
                    reward,
                )

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
            other_vel.append(other.state.vel)

        # Get target position relative to agent
        target = self.world.landmarks[agent_idx]
        target_pos = target.state.pos - agent.state.pos

        # Get droplet positions and velocities relative to agent
        droplet_pos = []
        droplet_vel = []
        for droplet in self.world.landmarks[self.n_agents :]:
            droplet_pos.append(droplet.state.pos - agent.state.pos)
            droplet_vel.append(droplet.state.vel)

        # Stack observations
        obs = jnp.concatenate(
            [agent.state.pos]  # Own position
            + [agent.state.vel]  # Own velocity
            + [target_pos]  # Target position
            + droplet_pos  # Droplet positions
            + droplet_vel  # Droplet velocities
            + other_pos  # Other agents' positions
            + other_vel,  # Other agents' velocities
            axis=-1,
        )
        return obs

    def done(self) -> Float[Array, f"{batch}"]:
        # Episode ends when all agents reach their targets
        return self.state.total_reached >= self.n_agents

    def update_world(self):
        # Reset droplets that fall below the bottom of the arena
        for droplet in self.world.landmarks[self.n_agents :]:
            # Check if droplet is below bottom
            is_below = droplet.state.pos[:, 1] < -self.arena_size
            # Reset position to top with random x coordinate
            new_x = jnp.where(
                is_below,
                -self.arena_size + 2 * self.arena_size * jnp.random.uniform(),
                droplet.state.pos[:, 0],
            )
            new_y = jnp.where(
                is_below,
                self.arena_size,
                droplet.state.pos[:, 1],
            )
            droplet.state.pos = jnp.stack([new_x, new_y], axis=-1)
            # Reset velocity to initial downward velocity
            droplet.state.vel = jnp.where(
                is_below[:, None],
                jnp.array([0.0, -self.droplet_speed]),
                droplet.state.vel,
            )
