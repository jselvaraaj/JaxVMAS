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


@struct.dataclass
class BallPassageState:
    """Dynamic state for BallPassage scenario."""

    ball_touched: Float[Array, f"{batch}"] = struct.field(
        default_factory=lambda: jnp.zeros((1,))
    )
    ball_delivered: Float[Array, f"{batch}"] = struct.field(
        default_factory=lambda: jnp.zeros((1,))
    )
    agent_ball_distances: Float[Array, f"{batch} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1))
    )


class BallPassage(BaseScenario):
    """
    Scenario where agents must guide a ball through a narrow passage to reach a target.
    The passage creates a bottleneck that requires careful coordination.
    """

    def __init__(self):
        super().__init__()
        self.n_agents = 3
        self.agent_size = 0.05
        self.ball_size = 0.075
        self.target_size = 0.1
        self.passage_width = 0.3
        self.wall_thickness = 0.05
        self.collision_penalty = 1.0
        self.ball_mass = 0.5
        self.touch_distance = 0.1
        self.delivery_distance = 0.1
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

        # Add ball (as entity)
        ball = Entity(name="ball")
        ball.collide = True
        ball.movable = True
        ball.mass = self.ball_mass
        ball.size = self.ball_size
        ball.color = jnp.array([0.85, 0.35, 0.35])
        world.add_entity(ball)

        # Add target (as landmark)
        target = Entity(name="target")
        target.collide = False
        target.movable = False
        target.size = self.target_size
        target.color = jnp.array([0.25, 0.85, 0.25])
        world.add_landmark(target)

        # Add walls (as landmarks)
        # Top wall
        wall_top = Entity(name="wall_top")
        wall_top.collide = True
        wall_top.movable = False
        wall_top.size = self.wall_thickness
        wall_top.color = jnp.array([0.25, 0.25, 0.25])
        wall_top.state.pos = jnp.array(
            [0.0, self.passage_width / 2 + self.wall_thickness / 2]
        )
        world.add_landmark(wall_top)

        # Bottom wall
        wall_bottom = Entity(name="wall_bottom")
        wall_bottom.collide = True
        wall_bottom.movable = False
        wall_bottom.size = self.wall_thickness
        wall_bottom.color = jnp.array([0.25, 0.25, 0.25])
        wall_bottom.state.pos = jnp.array(
            [0.0, -self.passage_width / 2 - self.wall_thickness / 2]
        )
        world.add_landmark(wall_bottom)

        # Initialize scenario state
        self.state = BallPassageState(
            ball_touched=jnp.zeros(batch_dim),
            ball_delivered=jnp.zeros(batch_dim),
            agent_ball_distances=jnp.zeros((batch_dim, self.n_agents)),
        )

        return world

    def reset_world_at(self, env_index: int | None):
        # Random agent positions on the left side
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents,
            self.world,
            env_index,
            min_dist_between_entities=0.15,
            x_bounds=(-1, -0.2),
            y_bounds=(-1, 1),
        )

        # Ball starts on the left side
        ball = self.world.entities[0]  # First entity is the ball
        ScenarioUtils.spawn_entities_randomly(
            [ball],
            self.world,
            env_index,
            min_dist_between_entities=0.15,
            x_bounds=(-0.8, -0.4),
            y_bounds=(-0.3, 0.3),
        )

        # Target is on the right side
        target = self.world.landmarks[0]  # First landmark is the target
        ScenarioUtils.spawn_entities_randomly(
            [target],
            self.world,
            env_index,
            min_dist_between_entities=0.15,
            x_bounds=(0.4, 0.8),
            y_bounds=(-0.3, 0.3),
        )

        # Reset state
        batch_size = self.world.batch_dim if env_index is None else 1
        self.state = BallPassageState(
            ball_touched=jnp.zeros(batch_size),
            ball_delivered=jnp.zeros(batch_size),
            agent_ball_distances=jnp.zeros((batch_size, self.n_agents)),
        )

    def reward(self, agent: Agent) -> Float[Array, f"{batch}"]:
        agent_idx = self.world.agents.index(agent)
        reward = jnp.zeros(self.world.batch_dim)

        # Get ball and target
        ball = self.world.entities[0]
        target = self.world.landmarks[0]

        # Distance to ball
        dist_to_ball = jnp.linalg.norm(agent.state.pos - ball.state.pos, axis=-1)
        self.state = self.state.replace(
            agent_ball_distances=self.state.agent_ball_distances.at[:, agent_idx].set(
                dist_to_ball
            )
        )

        # Ball touching reward
        touching = dist_to_ball < self.touch_distance
        reward = jnp.where(touching, reward + 0.1, reward)

        # Update ball touched status
        self.state = self.state.replace(
            ball_touched=jnp.logical_or(touching, self.state.ball_touched)
        )

        # Distance between ball and target
        ball_target_dist = jnp.linalg.norm(ball.state.pos - target.state.pos, axis=-1)

        # Ball delivery reward
        delivered = ball_target_dist < self.delivery_distance
        reward = jnp.where(delivered, reward + 1.0, reward)

        # Update ball delivered status
        self.state = self.state.replace(ball_delivered=delivered)

        # Shaped reward based on ball-target distance
        reward -= 0.1 * ball_target_dist

        # Collision penalties
        if agent.collision_penalty:
            # Agent-agent collisions
            for other in self.world.agents:
                if other is agent:
                    continue
                collision_dist = agent.size + other.size
                dist = jnp.linalg.norm(agent.state.pos - other.state.pos, axis=-1)
                reward = jnp.where(
                    dist < collision_dist, reward - self.collision_penalty, reward
                )

            # Agent-wall collisions
            for wall in self.world.landmarks[1:]:  # Skip target, check walls
                collision_dist = agent.size + wall.size
                dist = jnp.linalg.norm(agent.state.pos - wall.state.pos, axis=-1)
                reward = jnp.where(
                    dist < collision_dist, reward - self.collision_penalty, reward
                )

        return reward

    def observation(self, agent: Agent) -> Float[Array, f"{batch} ..."]:
        agent_idx = self.world.agents.index(agent)

        # Get ball and target positions relative to agent
        ball = self.world.entities[0]
        target = self.world.landmarks[0]
        ball_pos = ball.state.pos - agent.state.pos
        target_pos = target.state.pos - agent.state.pos

        # Get wall positions relative to agent
        wall_pos = []
        for wall in self.world.landmarks[1:]:  # Skip target, get walls
            wall_pos.append(wall.state.pos - agent.state.pos)

        # Get positions of other agents relative to this agent
        other_pos = []
        for other in self.world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.pos - agent.state.pos)

        # Stack observations
        obs = jnp.concatenate(
            [agent.state.pos]  # Own position
            + [agent.state.vel]  # Own velocity
            + [ball_pos]  # Relative position to ball
            + [ball.state.vel]  # Ball velocity
            + [target_pos]  # Relative position to target
            + wall_pos  # Relative positions to walls
            + other_pos,  # Relative positions to other agents
            axis=-1,
        )
        return obs

    def done(self) -> Float[Array, f"{batch}"]:
        # Episode is done when ball is delivered
        return self.state.ball_delivered
