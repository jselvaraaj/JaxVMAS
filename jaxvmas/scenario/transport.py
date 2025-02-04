#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import jax
import jax.numpy as jnp
from flax import struct
from jaxtyping import Array, Float

from jaxvmas.simulator.core import Agent, Entity, World
from jaxvmas.simulator.dynamics.holonomic import Holonomic
from jaxvmas.simulator.scenario import BaseScenario

# Type dimensions
batch = "batch"
n_agents = "n_agents"
n_boxes = "n_boxes"


@struct.dataclass
class TransportState:
    """Dynamic state for Transport scenario."""

    agent_box_distances: Float[Array, f"{batch} {n_agents} {n_boxes}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1, 1))
    )
    box_target_distances: Float[Array, f"{batch} {n_boxes}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1))
    )
    boxes_delivered: Float[Array, f"{batch} {n_boxes}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1))
    )
    total_delivered: Float[Array, f"{batch}"] = struct.field(
        default_factory=lambda: jnp.zeros((1,))
    )


class Transport(BaseScenario):
    """
    Scenario where agents must cooperatively transport boxes to target locations.
    Boxes are heavy and require multiple agents to move efficiently.
    """

    def __init__(self):
        super().__init__()
        self.n_agents = 4
        self.n_boxes = 2
        self.agent_size = 0.05
        self.box_size = 0.15
        self.target_size = 0.15
        self.collision_penalty = 1.0
        self.box_mass = 2.0  # Heavy boxes require cooperation
        self.delivery_distance = 0.1
        self.delivery_reward = 10.0
        self.min_spawn_distance = 0.5  # Minimum distance between boxes and targets
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

        # Add boxes (as entities)
        for i in range(self.n_boxes):
            box = Entity(name=f"box_{i}")
            box.collide = True
            box.movable = True
            box.mass = self.box_mass
            box.size = self.box_size
            box.color = jnp.array([0.85, 0.85, 0.35])  # Yellow boxes
            world.add_entity(box)

        # Add targets (as landmarks)
        for i in range(self.n_boxes):
            target = Entity(name=f"target_{i}")
            target.collide = False
            target.movable = False
            target.size = self.target_size
            target.color = jnp.array([0.25, 0.85, 0.25])  # Green targets
            world.add_landmark(target)

        # Initialize scenario state
        self.state = TransportState(
            agent_box_distances=jnp.zeros((batch_dim, self.n_agents, self.n_boxes)),
            box_target_distances=jnp.zeros((batch_dim, self.n_boxes)),
            boxes_delivered=jnp.zeros((batch_dim, self.n_boxes)),
            total_delivered=jnp.zeros(batch_dim),
        )

        return world

    def reset_world_at(self, env_index: int | None):
        batch_size = self.world.batch_dim if env_index is None else 1

        # Place boxes in random positions on left side
        for i, box in enumerate(self.world.entities):
            pos = jax.random.uniform(
                jax.random.PRNGKey(i),
                (batch_size if env_index is None else 1, 2),
                minval=jnp.array([-1.0, -0.8]),
                maxval=jnp.array([-0.8, 0.8]),
            )
            box.state.pos = jnp.where(
                env_index is None,
                pos,
                pos[0],
            )
            box.state.vel = jnp.zeros_like(box.state.vel)

        # Place targets in random positions on right side
        for i, target in enumerate(self.world.landmarks):
            valid_position = False
            attempts = 0
            while not valid_position and attempts < 100:
                pos = jax.random.uniform(
                    jax.random.PRNGKey(i + self.n_boxes + attempts),
                    (batch_size if env_index is None else 1, 2),
                    minval=jnp.array([0.8, -0.8]),
                    maxval=jnp.array([1.0, 0.8]),
                )

                # Check distance from other targets
                valid_position = True
                for j in range(i):
                    other_target = self.world.landmarks[j]
                    dist = jnp.linalg.norm(pos - other_target.state.pos, axis=-1)
                    if jnp.any(dist < self.min_spawn_distance):
                        valid_position = False
                        break
                attempts += 1

            target.state.pos = jnp.where(
                env_index is None,
                pos,
                pos[0],
            )

        # Place agents around boxes
        for i, agent in enumerate(self.world.agents):
            box_idx = i % self.n_boxes
            box = self.world.entities[box_idx]
            angle = (i // self.n_boxes) * (2 * jnp.pi / (self.n_agents // self.n_boxes))
            radius = 0.2

            x_pos = box.state.pos[:, 0] + radius * jnp.cos(angle)
            y_pos = box.state.pos[:, 1] + radius * jnp.sin(angle)

            agent.state.pos = jnp.where(
                env_index is None,
                jnp.stack([x_pos, y_pos], axis=-1),
                jnp.array([x_pos[0], y_pos[0]]),
            )
            agent.state.vel = jnp.zeros_like(agent.state.vel)

        # Reset state
        self.state = TransportState(
            agent_box_distances=jnp.zeros((batch_size, self.n_agents, self.n_boxes)),
            box_target_distances=jnp.zeros((batch_size, self.n_boxes)),
            boxes_delivered=jnp.zeros((batch_size, self.n_boxes)),
            total_delivered=jnp.zeros(batch_size),
        )

    def reward(self, agent: Agent) -> Float[Array, f"{batch}"]:
        reward = jnp.zeros(self.world.batch_dim)
        agent_idx = self.world.agents.index(agent)

        # Calculate distances between agents and boxes
        for i, box in enumerate(self.world.entities):
            # Distance to box
            dist_to_box = jnp.linalg.norm(agent.state.pos - box.state.pos, axis=-1)
            self.state = self.state.replace(
                agent_box_distances=self.state.agent_box_distances.at[
                    :, agent_idx, i
                ].set(dist_to_box)
            )

            # Distance from box to target
            target = self.world.landmarks[i]
            box_to_target = jnp.linalg.norm(box.state.pos - target.state.pos, axis=-1)
            self.state = self.state.replace(
                box_target_distances=self.state.box_target_distances.at[:, i].set(
                    box_to_target
                )
            )

            # Check for delivery
            delivered = box_to_target < self.delivery_distance
            was_delivered = self.state.boxes_delivered[:, i] > 0

            # Update delivery status and give rewards
            new_delivery = delivered & ~was_delivered
            reward = jnp.where(new_delivery, reward + self.delivery_reward, reward)

            # Update box delivery status
            self.state = self.state.replace(
                boxes_delivered=self.state.boxes_delivered.at[:, i].set(
                    jnp.where(delivered, 1.0, self.state.boxes_delivered[:, i])
                )
            )

            # Reward for being close to undelivered boxes
            if not was_delivered:
                reward = jnp.where(
                    dist_to_box < self.box_size + self.agent_size,
                    reward + 0.1,
                    reward,
                )

            # Small reward for moving boxes closer to targets
            reward -= 0.1 * box_to_target

        # Update total deliveries
        self.state = self.state.replace(
            total_delivered=jnp.sum(self.state.boxes_delivered, axis=-1)
        )

        # Collision penalties
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
        # Get positions of other agents relative to this agent
        other_pos = []
        other_vel = []
        for other in self.world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.pos - agent.state.pos)
            other_vel.append(other.state.vel)

        # Get positions and velocities of boxes relative to agent
        box_pos = []
        box_vel = []
        for box in self.world.entities:
            box_pos.append(box.state.pos - agent.state.pos)
            box_vel.append(box.state.vel)

        # Get positions of targets relative to agent
        target_pos = []
        for target in self.world.landmarks:
            target_pos.append(target.state.pos - agent.state.pos)

        # Stack observations
        obs = jnp.concatenate(
            [agent.state.pos]  # Own position
            + [agent.state.vel]  # Own velocity
            + other_pos  # Other agents' positions
            + other_vel  # Other agents' velocities
            + box_pos  # Box positions
            + box_vel  # Box velocities
            + target_pos  # Target positions
            + [self.state.boxes_delivered],  # Delivery status
            axis=-1,
        )
        return obs

    def done(self) -> Float[Array, f"{batch}"]:
        # Episode ends when all boxes are delivered
        return self.state.total_delivered >= self.n_boxes
