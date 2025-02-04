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
n_landmarks = "n_landmarks"


@struct.dataclass
class SimpleState:
    """Dynamic state for Simple scenario."""

    agent_distances: Float[Array, f"{batch} {n_agents} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1, 1))
    )
    landmark_distances: Float[Array, f"{batch} {n_agents} {n_landmarks}"] = (
        struct.field(default_factory=lambda: jnp.zeros((1, 1, 1)))
    )


class Simple(BaseScenario):
    """
    Simple scenario where agents must navigate to landmarks while avoiding
    collisions with each other.
    """

    def __init__(self):
        super().__init__()
        self.n_agents = 3
        self.n_landmarks = 3
        self.agent_size = 0.05
        self.landmark_size = 0.05
        self.collision_penalty = 1.0
        self.arena_size = 1.0
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

        # Add landmarks
        for i in range(self.n_landmarks):
            landmark = Entity(name=f"landmark_{i}")
            landmark.collide = True
            landmark.movable = False
            landmark.size = self.landmark_size
            landmark.color = jnp.array([0.25, 0.25, 0.25])
            world.add_landmark(landmark)

        # Initialize scenario state
        self.state = SimpleState(
            agent_distances=jnp.zeros((batch_dim, self.n_agents, self.n_agents)),
            landmark_distances=jnp.zeros((batch_dim, self.n_agents, self.n_landmarks)),
        )

        return world

    def reset_world_at(self, env_index: int | None):
        batch_size = self.world.batch_dim if env_index is None else 1

        # Place agents randomly
        for agent in self.world.agents:
            pos = jnp.array(
                [
                    jnp.random.uniform(-self.arena_size + 0.1, self.arena_size - 0.1),
                    jnp.random.uniform(-self.arena_size + 0.1, self.arena_size - 0.1),
                ]
            )
            agent.state.pos = jnp.where(
                env_index is None,
                jnp.tile(pos[None, :], (batch_size, 1)),
                pos,
            )
            agent.state.vel = jnp.zeros_like(agent.state.vel)

        # Place landmarks randomly
        for landmark in self.world.landmarks:
            pos = jnp.array(
                [
                    jnp.random.uniform(-self.arena_size + 0.1, self.arena_size - 0.1),
                    jnp.random.uniform(-self.arena_size + 0.1, self.arena_size - 0.1),
                ]
            )
            landmark.state.pos = jnp.where(
                env_index is None,
                jnp.tile(pos[None, :], (batch_size, 1)),
                pos,
            )

        # Reset state
        self.state = SimpleState(
            agent_distances=jnp.zeros((batch_size, self.n_agents, self.n_agents)),
            landmark_distances=jnp.zeros((batch_size, self.n_agents, self.n_landmarks)),
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

        # Update distances to landmarks
        for i, landmark in enumerate(self.world.landmarks):
            dist = jnp.linalg.norm(agent.state.pos - landmark.state.pos, axis=-1)
            self.state = self.state.replace(
                landmark_distances=self.state.landmark_distances.at[
                    :, agent_idx, i
                ].set(dist)
            )

        # Reward is negative sum of distances to landmarks
        reward -= jnp.sum(self.state.landmark_distances[:, agent_idx], axis=-1)

        # Collision penalties
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
        # Get positions and velocities of other agents relative to this agent
        other_pos = []
        other_vel = []
        for other in self.world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.pos - agent.state.pos)
            other_vel.append(other.state.vel)

        # Get landmark positions relative to agent
        landmark_pos = []
        for landmark in self.world.landmarks:
            landmark_pos.append(landmark.state.pos - agent.state.pos)

        # Stack observations
        obs = jnp.concatenate(
            [agent.state.pos]  # Own position
            + [agent.state.vel]  # Own velocity
            + other_pos  # Other agents' positions
            + other_vel  # Other agents' velocities
            + landmark_pos,  # Landmark positions
            axis=-1,
        )

        return obs

    def done(self) -> Float[Array, f"{batch}"]:
        # Episode never ends
        return jnp.zeros(self.world.batch_dim, dtype=bool)
