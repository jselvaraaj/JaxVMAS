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
msg_length = "msg_length"


@struct.dataclass
class SimpleSpeakerListenerState:
    """Dynamic state for Simple Speaker Listener scenario."""

    agent_distances: Float[Array, f"{batch} {n_agents} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1, 1))
    )
    landmark_distances: Float[Array, f"{batch} {n_agents} {n_landmarks}"] = (
        struct.field(default_factory=lambda: jnp.zeros((1, 1, 1)))
    )
    target_landmark: Float[Array, f"{batch}"] = struct.field(
        default_factory=lambda: jnp.zeros((1,))
    )
    message: Float[Array, f"{batch} {msg_length}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1))
    )


class SimpleSpeakerListener(BaseScenario):
    """
    Simple speaker listener scenario where a speaker must communicate
    information about a target landmark to a listener.
    """

    def __init__(self):
        super().__init__()
        self.n_agents = 2  # Speaker and listener
        self.n_landmarks = 3
        self.msg_length = 3  # Length of communication message
        self.agent_size = 0.05
        self.landmark_size = 0.05
        self.collision_penalty = 1.0
        self.arena_size = 1.0
        self.state = None

    def make_world(self, batch_dim: int, **kwargs) -> World:
        world = World(batch_dim=batch_dim, dim_p=2)

        # Add speaker (no movement)
        speaker = Agent(name="speaker", dynamics=Holonomic())
        speaker.color = jnp.array([0.35, 0.85, 0.35])  # Green for speaker
        speaker.collision_penalty = False
        speaker.size = self.agent_size
        speaker.movable = False
        speaker.silent = False
        speaker.speaker = True
        world.add_agent(speaker)

        # Add listener
        listener = Agent(name="listener", dynamics=Holonomic())
        listener.color = jnp.array([0.35, 0.35, 0.85])  # Blue for listener
        listener.collision_penalty = True
        listener.size = self.agent_size
        listener.movable = True
        listener.silent = True
        listener.speaker = False
        world.add_agent(listener)

        # Add landmarks
        for i in range(self.n_landmarks):
            landmark = Entity(name=f"landmark_{i}")
            landmark.collide = True
            landmark.movable = False
            landmark.size = self.landmark_size
            landmark.color = jnp.array([0.25, 0.25, 0.25])
            world.add_landmark(landmark)

        # Initialize scenario state
        self.state = SimpleSpeakerListenerState(
            agent_distances=jnp.zeros((batch_dim, self.n_agents, self.n_agents)),
            landmark_distances=jnp.zeros((batch_dim, self.n_agents, self.n_landmarks)),
            target_landmark=jnp.zeros(batch_dim),
            message=jnp.zeros((batch_dim, self.msg_length)),
        )

        return world

    def reset_world_at(self, env_index: int | None):
        batch_size = self.world.batch_dim if env_index is None else 1

        # Place speaker at origin (doesn't move)
        speaker = self.world.agents[0]
        speaker.state.pos = jnp.zeros_like(speaker.state.pos)
        speaker.state.vel = jnp.zeros_like(speaker.state.vel)

        # Place listener randomly
        listener = self.world.agents[1]
        pos = jnp.array(
            [
                jnp.random.uniform(-self.arena_size + 0.1, self.arena_size - 0.1),
                jnp.random.uniform(-self.arena_size + 0.1, self.arena_size - 0.1),
            ]
        )
        listener.state.pos = jnp.where(
            env_index is None,
            jnp.tile(pos[None, :], (batch_size, 1)),
            pos,
        )
        listener.state.vel = jnp.zeros_like(listener.state.vel)

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

        # Choose random target landmark
        target_landmark = jnp.floor(
            jnp.random.uniform(0, self.n_landmarks, (batch_size,))
        )

        # Reset state
        self.state = SimpleSpeakerListenerState(
            agent_distances=jnp.zeros((batch_size, self.n_agents, self.n_agents)),
            landmark_distances=jnp.zeros((batch_size, self.n_agents, self.n_landmarks)),
            target_landmark=target_landmark,
            message=jnp.zeros((batch_size, self.msg_length)),
        )

    def reward(self, agent: Agent) -> Float[Array, f"{batch}"]:
        reward = jnp.zeros(self.world.batch_dim)
        agent_idx = self.world.agents.index(agent)
        is_speaker = agent_idx == 0

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

        if not is_speaker:  # Listener
            # Reward based on distance to target landmark
            target_idx = self.state.target_landmark.astype(jnp.int32)
            target_dist = self.state.landmark_distances[
                jnp.arange(self.world.batch_dim), agent_idx, target_idx
            ]
            reward -= target_dist

            # Collision penalties
            if agent.collision_penalty:
                for landmark in self.world.landmarks:
                    collision_dist = agent.size + landmark.size
                    dist = jnp.linalg.norm(
                        agent.state.pos - landmark.state.pos, axis=-1
                    )
                    reward = jnp.where(
                        dist < collision_dist,
                        reward - self.collision_penalty,
                        reward,
                    )

        return reward

    def observation(self, agent: Agent) -> Float[Array, f"{batch} ..."]:
        agent_idx = self.world.agents.index(agent)
        is_speaker = agent_idx == 0

        if is_speaker:
            # Speaker observes target landmark index
            obs = jnp.concatenate(
                [self.state.target_landmark[:, None]],  # Target landmark index
                axis=-1,
            )
        else:  # Listener
            # Get landmark positions relative to listener
            landmark_pos = []
            for landmark in self.world.landmarks:
                landmark_pos.append(landmark.state.pos - agent.state.pos)

            # Stack observations
            obs = jnp.concatenate(
                [agent.state.pos]  # Own position
                + [agent.state.vel]  # Own velocity
                + [self.state.message]  # Speaker's message
                + landmark_pos,  # Landmark positions
                axis=-1,
            )

        return obs

    def done(self) -> Float[Array, f"{batch}"]:
        # Episode never ends
        return jnp.zeros(self.world.batch_dim, dtype=bool)
