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
class SimpleReferenceState:
    """Dynamic state for Simple Reference scenario."""

    agent_distances: Float[Array, f"{batch} {n_agents} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1, 1))
    )
    landmark_distances: Float[Array, f"{batch} {n_agents} {n_landmarks}"] = (
        struct.field(default_factory=lambda: jnp.zeros((1, 1, 1)))
    )
    target_landmarks: Float[Array, f"{batch} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1))
    )
    color_references: Float[Array, f"{batch} {n_agents} 3"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1, 3))
    )


class SimpleReference(BaseScenario):
    """
    Simple reference scenario where agents must coordinate to reach specific
    landmarks based on color references.
    """

    def __init__(self):
        super().__init__()
        self.n_agents = 3
        self.n_landmarks = 3
        self.agent_size = 0.05
        self.landmark_size = 0.05
        self.collision_penalty = 1.0
        self.arena_size = 1.0
        self.reference_reward = 1.0
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

        # Add landmarks with different colors
        colors = [
            jnp.array([0.85, 0.35, 0.35]),  # Red
            jnp.array([0.35, 0.85, 0.35]),  # Green
            jnp.array([0.35, 0.35, 0.85]),  # Blue
        ]
        for i in range(self.n_landmarks):
            landmark = Entity(name=f"landmark_{i}")
            landmark.collide = True
            landmark.movable = False
            landmark.size = self.landmark_size
            landmark.color = colors[i]
            world.add_landmark(landmark)

        # Initialize scenario state
        self.state = SimpleReferenceState(
            agent_distances=jnp.zeros((batch_dim, self.n_agents, self.n_agents)),
            landmark_distances=jnp.zeros((batch_dim, self.n_agents, self.n_landmarks)),
            target_landmarks=jnp.zeros((batch_dim, self.n_agents)),
            color_references=jnp.zeros((batch_dim, self.n_agents, 3)),
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

        # Assign random target landmarks to agents
        target_landmarks = jnp.floor(
            jnp.random.uniform(0, self.n_landmarks, (batch_size, self.n_agents))
        )

        # Get color references for each agent's target
        color_references = jnp.zeros((batch_size, self.n_agents, 3))
        for i in range(self.n_agents):
            for j, landmark in enumerate(self.world.landmarks):
                color_references = color_references.at[:, i].set(
                    jnp.where(
                        target_landmarks[:, i] == j,
                        landmark.color,
                        color_references[:, i],
                    )
                )

        # Reset state
        self.state = SimpleReferenceState(
            agent_distances=jnp.zeros((batch_size, self.n_agents, self.n_agents)),
            landmark_distances=jnp.zeros((batch_size, self.n_agents, self.n_landmarks)),
            target_landmarks=target_landmarks,
            color_references=color_references,
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

        # Reward based on distance to target landmark
        target_idx = self.state.target_landmarks[:, agent_idx].astype(jnp.int32)
        target_dist = self.state.landmark_distances[
            jnp.arange(self.world.batch_dim), agent_idx, target_idx
        ]
        reward -= target_dist

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
        agent_idx = self.world.agents.index(agent)

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

        # Get landmark colors
        landmark_colors = []
        for landmark in self.world.landmarks:
            landmark_colors.append(landmark.color)

        # Stack observations
        obs = jnp.concatenate(
            [agent.state.pos]  # Own position
            + [agent.state.vel]  # Own velocity
            + [self.state.color_references[:, agent_idx]]  # Target color reference
            + other_pos  # Other agents' positions
            + other_vel  # Other agents' velocities
            + landmark_pos  # Landmark positions
            + landmark_colors,  # Landmark colors
            axis=-1,
        )

        return obs

    def done(self) -> Float[Array, f"{batch}"]:
        # Episode never ends
        return jnp.zeros(self.world.batch_dim, dtype=bool)
