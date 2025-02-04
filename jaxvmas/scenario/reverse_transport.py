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
n_objects = "n_objects"


@struct.dataclass
class ReverseTransportState:
    """Dynamic state for ReverseTransport scenario."""

    agent_distances: Float[Array, f"{batch} {n_agents} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1, 1))
    )
    object_distances: Float[Array, f"{batch} {n_agents} {n_objects}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1, 1))
    )
    target_distances: Float[Array, f"{batch} {n_objects}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1))
    )
    objects_delivered: Float[Array, f"{batch} {n_objects}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1))
    )
    total_delivered: Float[Array, f"{batch}"] = struct.field(
        default_factory=lambda: jnp.zeros((1,))
    )
    progress: Float[Array, f"{batch} {n_objects}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1))
    )


class ReverseTransport(BaseScenario):
    """
    Scenario where agents must transport objects with reversed dynamics to targets.
    Objects have negative mass, causing them to move in the opposite direction
    of applied forces. Agents must coordinate to control object movement.
    """

    def __init__(self):
        super().__init__()
        self.n_agents = 4
        self.n_objects = 2
        self.agent_size = 0.05
        self.object_size = 0.08
        self.target_size = 0.10
        self.collision_penalty = 1.0
        self.arena_size = 1.0
        self.delivery_reward = 10.0
        self.progress_reward_scale = 0.1
        self.delivery_threshold = 0.1
        self.object_mass = -1.0  # Negative mass for reverse dynamics
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

        # Add objects to transport
        for i in range(self.n_objects):
            obj = Entity(name=f"object_{i}")
            obj.collide = True
            obj.movable = True
            obj.size = self.object_size
            obj.mass = self.object_mass
            obj.color = jnp.array([0.85, 0.35, 0.35])
            world.add_landmark(obj)

        # Add targets
        for i in range(self.n_objects):
            target = Entity(name=f"target_{i}")
            target.collide = False
            target.movable = False
            target.size = self.target_size
            target.color = jnp.array([0.25, 0.85, 0.25])
            world.add_landmark(target)

        # Initialize scenario state
        self.state = ReverseTransportState(
            agent_distances=jnp.zeros((batch_dim, self.n_agents, self.n_agents)),
            object_distances=jnp.zeros((batch_dim, self.n_agents, self.n_objects)),
            target_distances=jnp.zeros((batch_dim, self.n_objects)),
            objects_delivered=jnp.zeros((batch_dim, self.n_objects)),
            total_delivered=jnp.zeros(batch_dim),
            progress=jnp.zeros((batch_dim, self.n_objects)),
        )

        return world

    def reset_world_at(self, env_index: int | None):
        batch_size = self.world.batch_dim if env_index is None else 1

        # Place agents in a circle around the center
        for i, agent in enumerate(self.world.agents):
            angle = 2 * jnp.pi * i / self.n_agents
            pos = jnp.array(
                [
                    0.5 * jnp.cos(angle),
                    0.5 * jnp.sin(angle),
                ]
            )
            agent.state.pos = jnp.where(
                env_index is None,
                jnp.tile(pos[None, :], (batch_size, 1)),
                pos,
            )
            agent.state.vel = jnp.zeros_like(agent.state.vel)

        # Place objects randomly on the left side
        for i, obj in enumerate(self.world.landmarks[: self.n_objects]):
            y_offset = 0.4 * (i - (self.n_objects - 1) / 2)
            pos = jnp.array([-self.arena_size + 0.2, y_offset])
            obj.state.pos = jnp.where(
                env_index is None,
                jnp.tile(pos[None, :], (batch_size, 1)),
                pos,
            )
            obj.state.vel = jnp.zeros_like(obj.state.vel)

        # Place targets on the right side
        for i, target in enumerate(self.world.landmarks[self.n_objects :]):
            y_offset = 0.4 * (i - (self.n_objects - 1) / 2)
            pos = jnp.array([self.arena_size - 0.2, y_offset])
            target.state.pos = jnp.where(
                env_index is None,
                jnp.tile(pos[None, :], (batch_size, 1)),
                pos,
            )

        # Reset state
        self.state = ReverseTransportState(
            agent_distances=jnp.zeros((batch_size, self.n_agents, self.n_agents)),
            object_distances=jnp.zeros((batch_size, self.n_agents, self.n_objects)),
            target_distances=jnp.zeros((batch_size, self.n_objects)),
            objects_delivered=jnp.zeros((batch_size, self.n_objects)),
            total_delivered=jnp.zeros(batch_size),
            progress=jnp.zeros((batch_size, self.n_objects)),
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

        # Update distances to objects
        for i, obj in enumerate(self.world.landmarks[: self.n_objects]):
            dist = jnp.linalg.norm(agent.state.pos - obj.state.pos, axis=-1)
            self.state = self.state.replace(
                object_distances=self.state.object_distances.at[:, agent_idx, i].set(
                    dist
                )
            )

        # Update distances between objects and targets
        for i in range(self.n_objects):
            obj = self.world.landmarks[i]
            target = self.world.landmarks[self.n_objects + i]
            dist = jnp.linalg.norm(obj.state.pos - target.state.pos, axis=-1)
            self.state = self.state.replace(
                target_distances=self.state.target_distances.at[:, i].set(dist)
            )

            # Check if object is delivered
            delivered = dist < self.delivery_threshold
            was_delivered = self.state.objects_delivered[:, i] > 0

            # Update delivery status
            self.state = self.state.replace(
                objects_delivered=self.state.objects_delivered.at[:, i].set(
                    jnp.where(delivered, 1.0, self.state.objects_delivered[:, i])
                )
            )

            # Calculate progress (x-position progress towards target)
            old_progress = self.state.progress[:, i]
            new_progress = obj.state.pos[:, 0] + self.arena_size
            progress_delta = new_progress - old_progress

            # Update progress
            self.state = self.state.replace(
                progress=self.state.progress.at[:, i].set(new_progress)
            )

            # Add rewards for this object
            # 1. Delivery reward
            reward = jnp.where(
                delivered & ~was_delivered,
                reward + self.delivery_reward,
                reward,
            )

            # 2. Progress reward
            reward += self.progress_reward_scale * progress_delta

        # Update total delivered count
        self.state = self.state.replace(
            total_delivered=jnp.sum(self.state.objects_delivered, axis=-1)
        )

        # Apply collision penalties
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

        # Get object positions and velocities relative to agent
        object_pos = []
        object_vel = []
        for obj in self.world.landmarks[: self.n_objects]:
            object_pos.append(obj.state.pos - agent.state.pos)
            object_vel.append(obj.state.vel)

        # Get target positions relative to agent
        target_pos = []
        for target in self.world.landmarks[self.n_objects :]:
            target_pos.append(target.state.pos - agent.state.pos)

        # Stack observations
        obs = jnp.concatenate(
            [agent.state.pos]  # Own position
            + [agent.state.vel]  # Own velocity
            + object_pos  # Object positions
            + object_vel  # Object velocities
            + target_pos  # Target positions
            + other_pos  # Other agents' positions
            + other_vel  # Other agents' velocities
            + [self.state.objects_delivered],  # Delivery status
            axis=-1,
        )
        return obs

    def done(self) -> Float[Array, f"{batch}"]:
        # Episode ends when all objects are delivered
        return self.state.total_delivered >= self.n_objects
