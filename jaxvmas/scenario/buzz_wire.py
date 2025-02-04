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


@struct.dataclass
class BuzzWireState:
    """Dynamic state for BuzzWire scenario."""

    agent_distances: Float[Array, f"{batch} {n_agents} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1, 1))
    )
    wire_distances: Float[Array, f"{batch} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1))
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
    wire_touched: Float[Array, f"{batch} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1))
    )


class BuzzWire(BaseScenario):
    """
    Scenario where agents must navigate along a curved wire without touching it.
    Similar to the classic buzz wire game, agents must reach their targets while
    maintaining a safe distance from the wire.
    """

    def __init__(self):
        super().__init__()
        self.n_agents = 3
        self.agent_size = 0.05
        self.target_size = 0.05
        self.wire_thickness = 0.02
        self.collision_penalty = 1.0
        self.wire_penalty = 2.0  # Extra penalty for touching the wire
        self.arena_size = 1.0
        self.target_reward = 10.0
        self.progress_reward_scale = 0.1
        self.target_threshold = 0.1  # Distance threshold for reaching target
        self.wire_threshold = 0.1  # Safe distance from wire
        self.n_wire_points = 10  # Number of points defining the wire curve
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

        # Add targets (as landmarks)
        for i in range(self.n_agents):
            target = Entity(name=f"target_{i}")
            target.collide = False
            target.movable = False
            target.size = self.target_size
            target.color = jnp.array([0.25, 0.85, 0.25])  # Green targets
            world.add_landmark(target)

        # Add wire segments (as landmarks)
        for i in range(self.n_wire_points):
            wire = Entity(name=f"wire_{i}")
            wire.collide = True
            wire.movable = False
            wire.size = self.wire_thickness
            wire.color = jnp.array([0.85, 0.35, 0.35])  # Red wire
            world.add_landmark(wire)

        # Initialize scenario state
        self.state = BuzzWireState(
            agent_distances=jnp.zeros((batch_dim, self.n_agents, self.n_agents)),
            wire_distances=jnp.zeros((batch_dim, self.n_agents)),
            target_distances=jnp.zeros((batch_dim, self.n_agents)),
            targets_reached=jnp.zeros((batch_dim, self.n_agents)),
            total_reached=jnp.zeros(batch_dim),
            progress=jnp.zeros((batch_dim, self.n_agents)),
            wire_touched=jnp.zeros((batch_dim, self.n_agents)),
        )

        return world

    def reset_world_at(self, env_index: int | None):
        batch_size = self.world.batch_dim if env_index is None else 1

        # Generate wire path (curved path from left to right)
        t = jnp.linspace(0, 1, self.n_wire_points)
        wire_x = -self.arena_size + 2 * self.arena_size * t
        wire_y = 0.5 * jnp.sin(2 * jnp.pi * t)  # Sinusoidal path

        # Place wire segments
        for i in range(self.n_wire_points):
            wire = self.world.landmarks[self.n_agents + i]  # Offset by targets
            pos = jnp.stack([wire_x[i], wire_y[i]], axis=0)
            wire.state.pos = jnp.where(
                env_index is None,
                jnp.tile(pos[None, :], (batch_size, 1)),
                pos,
            )

        # Place agents at start (left side)
        for i, agent in enumerate(self.world.agents):
            y_offset = 0.2 * (i - (self.n_agents - 1) / 2)  # Stagger vertically
            pos = jnp.array([-self.arena_size + 0.1, wire_y[0] + y_offset])
            agent.state.pos = jnp.where(
                env_index is None,
                jnp.tile(pos[None, :], (batch_size, 1)),
                pos,
            )
            agent.state.vel = jnp.zeros_like(agent.state.vel)

        # Place targets at end (right side)
        for i, target in enumerate(self.world.landmarks[: self.n_agents]):
            y_offset = 0.2 * (i - (self.n_agents - 1) / 2)  # Stagger vertically
            pos = jnp.array([self.arena_size - 0.1, wire_y[-1] + y_offset])
            target.state.pos = jnp.where(
                env_index is None,
                jnp.tile(pos[None, :], (batch_size, 1)),
                pos,
            )

        # Reset state
        self.state = BuzzWireState(
            agent_distances=jnp.zeros((batch_size, self.n_agents, self.n_agents)),
            wire_distances=jnp.zeros((batch_size, self.n_agents)),
            target_distances=jnp.zeros((batch_size, self.n_agents)),
            targets_reached=jnp.zeros((batch_size, self.n_agents)),
            total_reached=jnp.zeros(batch_size),
            progress=jnp.zeros((batch_size, self.n_agents)),
            wire_touched=jnp.zeros((batch_size, self.n_agents)),
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

        # Update distances to wire segments and check for wire touches
        min_wire_dist = jnp.inf * jnp.ones(self.world.batch_dim)
        for i in range(self.n_wire_points):
            wire = self.world.landmarks[self.n_agents + i]
            dist = jnp.linalg.norm(agent.state.pos - wire.state.pos, axis=-1)
            min_wire_dist = jnp.minimum(min_wire_dist, dist)

        self.state = self.state.replace(
            wire_distances=self.state.wire_distances.at[:, agent_idx].set(
                min_wire_dist
            ),
            wire_touched=self.state.wire_touched.at[:, agent_idx].set(
                min_wire_dist < self.wire_threshold
            ),
        )

        # Update distances to targets and check for reaching targets
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
        new_progress = agent.state.pos[:, 0] + self.arena_size  # Normalize to [0, 2]
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

        # 3. Wire touch penalty
        reward = jnp.where(
            min_wire_dist < self.wire_threshold,
            reward - self.wire_penalty,
            reward,
        )

        # 4. Collision penalties
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

        # Get wire positions relative to agent
        wire_pos = []
        for i in range(self.n_wire_points):
            wire = self.world.landmarks[self.n_agents + i]
            wire_pos.append(wire.state.pos - agent.state.pos)

        # Stack observations
        obs = jnp.concatenate(
            [agent.state.pos]  # Own position
            + [agent.state.vel]  # Own velocity
            + [target_pos]  # Target position
            + wire_pos  # Wire segment positions
            + other_pos  # Other agents' positions
            + other_vel  # Other agents' velocities
            + [self.state.wire_touched],  # Wire touch status
            axis=-1,
        )
        return obs

    def done(self) -> Float[Array, f"{batch}"]:
        # Episode ends when all agents reach their targets
        return self.state.total_reached >= self.n_agents
