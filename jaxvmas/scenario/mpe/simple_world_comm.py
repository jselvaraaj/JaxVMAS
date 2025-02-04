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
class SimpleWorldCommState:
    """Dynamic state for Simple World Comm scenario."""

    agent_distances: Float[Array, f"{batch} {n_agents} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1, 1))
    )
    landmark_distances: Float[Array, f"{batch} {n_agents} {n_landmarks}"] = (
        struct.field(default_factory=lambda: jnp.zeros((1, 1, 1)))
    )
    target_landmarks: Float[Array, f"{batch} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1))
    )
    messages: Float[Array, f"{batch} {n_agents} {msg_length}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1, 1))
    )


class SimpleWorldComm(BaseScenario):
    """
    Simple world comm scenario that combines communication and adversarial
    dynamics. Agents must coordinate through communication while avoiding
    adversaries.
    """

    def __init__(self):
        super().__init__()
        self.n_speakers = 2
        self.n_listeners = 2
        self.n_adversaries = 2
        self.n_agents = self.n_speakers + self.n_listeners + self.n_adversaries
        self.n_landmarks = 3
        self.msg_length = 4
        self.agent_size = 0.05
        self.landmark_size = 0.05
        self.collision_penalty = 1.0
        self.arena_size = 1.0
        self.target_reward = 10.0
        self.adversary_reward = 5.0
        self.comm_reward = 2.0
        self.state = None

    def make_world(self, batch_dim: int, **kwargs) -> World:
        world = World(batch_dim=batch_dim, dim_p=2)

        # Add speakers
        for i in range(self.n_speakers):
            agent = Agent(name=f"speaker_{i}", dynamics=Holonomic())
            agent.color = jnp.array([0.35, 0.85, 0.35])  # Green for speakers
            agent.collision_penalty = True
            agent.size = self.agent_size
            agent.speaker = True
            agent.listener = False
            agent.adversary = False
            world.add_agent(agent)

        # Add listeners
        for i in range(self.n_listeners):
            agent = Agent(name=f"listener_{i}", dynamics=Holonomic())
            agent.color = jnp.array([0.35, 0.35, 0.85])  # Blue for listeners
            agent.collision_penalty = True
            agent.size = self.agent_size
            agent.speaker = False
            agent.listener = True
            agent.adversary = False
            world.add_agent(agent)

        # Add adversaries
        for i in range(self.n_adversaries):
            agent = Agent(name=f"adversary_{i}", dynamics=Holonomic())
            agent.color = jnp.array([0.85, 0.35, 0.35])  # Red for adversaries
            agent.collision_penalty = True
            agent.size = self.agent_size
            agent.speaker = False
            agent.listener = False
            agent.adversary = True
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
        self.state = SimpleWorldCommState(
            agent_distances=jnp.zeros((batch_dim, self.n_agents, self.n_agents)),
            landmark_distances=jnp.zeros((batch_dim, self.n_agents, self.n_landmarks)),
            target_landmarks=jnp.zeros((batch_dim, self.n_agents)),
            messages=jnp.zeros((batch_dim, self.n_agents, self.msg_length)),
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

        # Assign random target landmarks to listeners
        target_landmarks = jnp.zeros((batch_size, self.n_agents))
        for i in range(self.n_speakers, self.n_speakers + self.n_listeners):
            target_landmarks = target_landmarks.at[:, i].set(
                jnp.floor(jnp.random.uniform(0, self.n_landmarks, (batch_size,)))
            )

        # Reset state
        self.state = SimpleWorldCommState(
            agent_distances=jnp.zeros((batch_size, self.n_agents, self.n_agents)),
            landmark_distances=jnp.zeros((batch_size, self.n_agents, self.n_landmarks)),
            target_landmarks=target_landmarks,
            messages=jnp.zeros((batch_size, self.n_agents, self.msg_length)),
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

        if agent.speaker:
            # Speaker rewards
            # Reward for helping listeners reach their targets
            for i in range(self.n_speakers, self.n_speakers + self.n_listeners):
                target_idx = self.state.target_landmarks[:, i].astype(jnp.int32)
                target_dist = self.state.landmark_distances[
                    jnp.arange(self.world.batch_dim), i, target_idx
                ]
                reward -= target_dist  # Want to minimize listener's distance to target

        elif agent.listener:
            # Listener rewards
            # Reward for reaching target landmark
            target_idx = self.state.target_landmarks[:, agent_idx].astype(jnp.int32)
            target_dist = self.state.landmark_distances[
                jnp.arange(self.world.batch_dim), agent_idx, target_idx
            ]
            reward -= target_dist

            # Penalty for being close to adversaries
            for i in range(self.n_speakers + self.n_listeners, self.n_agents):
                dist = self.state.agent_distances[:, agent_idx, i]
                reward += 0.1 * dist  # Want to maximize distance to adversaries

        else:  # adversary
            # Adversary rewards
            # Reward for being close to listeners
            for i in range(self.n_speakers, self.n_speakers + self.n_listeners):
                dist = self.state.agent_distances[:, agent_idx, i]
                reward -= 0.1 * dist  # Want to minimize distance to listeners

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

        if agent.speaker:
            # Speaker observes target landmarks for listeners
            target_indicators = []
            for i in range(self.n_speakers, self.n_speakers + self.n_listeners):
                indicator = jnp.zeros((self.world.batch_dim, self.n_landmarks))
                target_idx = self.state.target_landmarks[:, i].astype(jnp.int32)
                indicator = indicator.at[
                    jnp.arange(self.world.batch_dim), target_idx
                ].set(1.0)
                target_indicators.append(indicator)

            # Stack observations
            obs = jnp.concatenate(
                [agent.state.pos]  # Own position
                + [agent.state.vel]  # Own velocity
                + other_pos  # Other agents' positions
                + other_vel  # Other agents' velocities
                + landmark_pos  # Landmark positions
                + target_indicators,  # Target indicators for each listener
                axis=-1,
            )
        elif agent.listener:
            # Listener observes messages from speakers
            messages = []
            for i in range(self.n_speakers):
                messages.append(self.state.messages[:, i])

            # Stack observations
            obs = jnp.concatenate(
                [agent.state.pos]  # Own position
                + [agent.state.vel]  # Own velocity
                + other_pos  # Other agents' positions
                + other_vel  # Other agents' velocities
                + landmark_pos  # Landmark positions
                + messages,  # Messages from speakers
                axis=-1,
            )
        else:  # adversary
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
