#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import jax.numpy as jnp
from flax import struct
from jaxtyping import Array, Float

from jaxvmas.simulator.core import Agent, World
from jaxvmas.simulator.dynamics.holonomic import Holonomic
from jaxvmas.simulator.scenario import BaseScenario

# Type dimensions
batch = "batch"
n_agents = "n_agents"
msg_length = "msg_length"


@struct.dataclass
class SimpleCryptoState:
    """Dynamic state for Simple Crypto scenario."""

    agent_distances: Float[Array, f"{batch} {n_agents} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1, 1))
    )
    message: Float[Array, f"{batch} {msg_length}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1))
    )
    speaker_encoding: Float[Array, f"{batch} {msg_length}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1))
    )
    listener_decoding: Float[Array, f"{batch} {msg_length}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1))
    )
    adversary_decoding: Float[Array, f"{batch} {msg_length}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1))
    )


class SimpleCrypto(BaseScenario):
    """
    Simple crypto scenario where a speaker must communicate a message to a
    listener while preventing an adversary from intercepting it.
    """

    def __init__(self):
        super().__init__()
        self.msg_length = 4  # Length of the message
        self.n_agents = 3  # Speaker, listener, adversary
        self.agent_size = 0.05
        self.collision_penalty = 1.0
        self.arena_size = 1.0
        self.state = None

    def make_world(self, batch_dim: int, **kwargs) -> World:
        world = World(batch_dim=batch_dim, dim_p=2)

        # Add speaker
        speaker = Agent(name="speaker", dynamics=Holonomic())
        speaker.color = jnp.array([0.35, 0.85, 0.35])  # Green for speaker
        speaker.collision_penalty = True
        speaker.size = self.agent_size
        speaker.speaker = True
        speaker.listener = False
        speaker.adversary = False
        world.add_agent(speaker)

        # Add listener
        listener = Agent(name="listener", dynamics=Holonomic())
        listener.color = jnp.array([0.35, 0.35, 0.85])  # Blue for listener
        listener.collision_penalty = True
        listener.size = self.agent_size
        listener.speaker = False
        listener.listener = True
        listener.adversary = False
        world.add_agent(listener)

        # Add adversary
        adversary = Agent(name="adversary", dynamics=Holonomic())
        adversary.color = jnp.array([0.85, 0.35, 0.35])  # Red for adversary
        adversary.collision_penalty = True
        adversary.size = self.agent_size
        adversary.speaker = False
        adversary.listener = False
        adversary.adversary = True
        world.add_agent(adversary)

        # Initialize scenario state
        self.state = SimpleCryptoState(
            agent_distances=jnp.zeros((batch_dim, self.n_agents, self.n_agents)),
            message=jnp.zeros((batch_dim, self.msg_length)),
            speaker_encoding=jnp.zeros((batch_dim, self.msg_length)),
            listener_decoding=jnp.zeros((batch_dim, self.msg_length)),
            adversary_decoding=jnp.zeros((batch_dim, self.msg_length)),
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

        # Generate random message
        message = jnp.random.uniform(0, 1, (batch_size, self.msg_length))

        # Reset state
        self.state = SimpleCryptoState(
            agent_distances=jnp.zeros((batch_size, self.n_agents, self.n_agents)),
            message=message,
            speaker_encoding=jnp.zeros((batch_size, self.msg_length)),
            listener_decoding=jnp.zeros((batch_size, self.msg_length)),
            adversary_decoding=jnp.zeros((batch_size, self.msg_length)),
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

        # Calculate message reconstruction errors
        listener_error = jnp.mean(
            jnp.square(self.state.message - self.state.listener_decoding),
            axis=-1,
        )
        adversary_error = jnp.mean(
            jnp.square(self.state.message - self.state.adversary_decoding),
            axis=-1,
        )

        if agent.speaker:
            # Speaker is rewarded for listener success and adversary failure
            reward += -listener_error + adversary_error
        elif agent.listener:
            # Listener is rewarded for successful message reconstruction
            reward += -listener_error
        else:  # adversary
            # Adversary is rewarded for successful message interception
            reward += -adversary_error

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

        if agent.speaker:
            # Speaker observes the true message and its encoding
            obs = jnp.concatenate(
                [agent.state.pos]  # Own position
                + [agent.state.vel]  # Own velocity
                + other_pos  # Other agents' positions
                + other_vel  # Other agents' velocities
                + [self.state.message]  # True message
                + [self.state.speaker_encoding],  # Speaker's encoding
                axis=-1,
            )
        elif agent.listener:
            # Listener observes the speaker's encoding and its own decoding
            obs = jnp.concatenate(
                [agent.state.pos]  # Own position
                + [agent.state.vel]  # Own velocity
                + other_pos  # Other agents' positions
                + other_vel  # Other agents' velocities
                + [self.state.speaker_encoding]  # Speaker's encoding
                + [self.state.listener_decoding],  # Listener's decoding
                axis=-1,
            )
        else:  # adversary
            # Adversary observes the speaker's encoding and its own decoding
            obs = jnp.concatenate(
                [agent.state.pos]  # Own position
                + [agent.state.vel]  # Own velocity
                + other_pos  # Other agents' positions
                + other_vel  # Other agents' velocities
                + [self.state.speaker_encoding]  # Speaker's encoding
                + [self.state.adversary_decoding],  # Adversary's decoding
                axis=-1,
            )

        return obs

    def done(self) -> Float[Array, f"{batch}"]:
        # Episode never ends
        return jnp.zeros(self.world.batch_dim, dtype=bool)
