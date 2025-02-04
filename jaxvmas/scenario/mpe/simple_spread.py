#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.


import jax.numpy as jnp
from flax import struct
from jaxtyping import Array, Float

from jaxvmas.simulator.core import Agent, Entity, World
from jaxvmas.simulator.dynamics.holonomic import Holonomic
from jaxvmas.simulator.scenario import BaseScenario
from jaxvmas.simulator.utils import ScenarioUtils

# Type dimensions
batch = "batch"
n_agents = "n_agents"
n_landmarks = "n_landmarks"


@struct.dataclass
class SimpleSpreadState:
    """Dynamic state for SimpleSpread scenario."""

    agent_goals: Float[Array, f"{batch} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1))
    )


class SimpleSpread(BaseScenario):
    """
    Scenario where N agents must cooperate to cover N landmarks.
    Agents are rewarded based on how close they are to landmarks,
    but they are penalized if they collide with other agents.
    """

    def __init__(self):
        super().__init__()
        self.n_agents = 3  # Number of agents
        self.n_landmarks = 3  # Number of landmarks to cover
        self.collision_penalty = 1.0  # Penalty for agent collisions
        self.agent_size = 0.05  # Size of agent
        self.landmark_size = 0.05  # Size of landmarks
        self.state = None  # Will hold SimpleSpreadState

    def make_world(self, batch_dim: int, **kwargs) -> World:
        # Create world
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
            landmark.collide = False
            landmark.movable = False
            landmark.size = self.landmark_size
            landmark.color = jnp.array([0.25, 0.25, 0.25])
            world.add_landmark(landmark)

        # Initialize scenario state
        self.state = SimpleSpreadState(
            agent_goals=jnp.zeros((batch_dim, self.n_agents))
        )

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

        # Random landmark positions
        ScenarioUtils.spawn_entities_randomly(
            self.world.landmarks,
            self.world,
            env_index,
            min_dist_between_entities=0.15,
            x_bounds=(-1, 1),
            y_bounds=(-1, 1),
        )

    def reward(self, agent: Agent) -> Float[Array, f"{batch}"]:
        # Agents are rewarded based on minimum agent distance to each landmark
        # Shaped reward = -min(agent_distances) for each landmark
        agent_pos = agent.state.pos  # [batch, 2]

        # Calculate distances to all landmarks
        landmark_positions = jnp.stack(
            [l.state.pos for l in self.world.landmarks]
        )  # [n_landmarks, batch, 2]
        landmark_positions = jnp.transpose(
            landmark_positions, (1, 0, 2)
        )  # [batch, n_landmarks, 2]

        # Calculate distances from agent to each landmark
        dists = jnp.linalg.norm(
            landmark_positions - agent_pos[:, None, :], axis=-1  # Broadcasting
        )  # [batch, n_landmarks]

        # Reward is negative minimum distance to any landmark
        reward = -jnp.min(dists, axis=-1)  # [batch]

        # Penalty for collisions with other agents
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
        # Get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in self.world.landmarks:  # world.entities:
            entity_pos.append(entity.state.pos - agent.state.pos)

        # Get positions of all other agents in this agent's reference frame
        other_pos = []
        for other in self.world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.pos - agent.state.pos)

        # Stack all observations
        obs = jnp.concatenate(
            [agent.state.pos]  # Own position
            + [agent.state.vel]  # Own velocity
            + entity_pos  # Relative positions to landmarks
            + other_pos,  # Relative positions to other agents
            axis=-1,
        )
        return obs
