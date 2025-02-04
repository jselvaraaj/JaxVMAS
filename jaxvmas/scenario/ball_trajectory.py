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


@struct.dataclass
class BallTrajectoryState:
    """Dynamic state for BallTrajectory scenario."""

    agent_ball_distances: Float[Array, f"{batch} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1))
    )
    ball_target_distance: Float[Array, f"{batch}"] = struct.field(
        default_factory=lambda: jnp.zeros((1,))
    )
    ball_touched: Float[Array, f"{batch}"] = struct.field(
        default_factory=lambda: jnp.zeros((1,))
    )
    target_reached: Float[Array, f"{batch}"] = struct.field(
        default_factory=lambda: jnp.zeros((1,))
    )


class BallTrajectory(BaseScenario):
    """
    Scenario where agents must guide a ball along a specific trajectory to reach a target.
    The ball has momentum and agents must carefully control its movement.
    """

    def __init__(self):
        super().__init__()
        self.n_agents = 2
        self.agent_size = 0.05
        self.ball_size = 0.03
        self.target_size = 0.1
        self.collision_penalty = 1.0
        self.ball_touch_distance = 0.1
        self.target_reach_distance = 0.1
        self.ball_mass = 0.3
        self.target_reward = 10.0
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
        ball.color = jnp.array([1.0, 1.0, 1.0])  # White ball
        world.add_entity(ball)

        # Add target (as landmark)
        target = Entity(name="target")
        target.collide = False
        target.movable = False
        target.size = self.target_size
        target.color = jnp.array([0.25, 0.85, 0.25])
        world.add_landmark(target)

        # Initialize scenario state
        self.state = BallTrajectoryState(
            agent_ball_distances=jnp.zeros((batch_dim, self.n_agents)),
            ball_target_distance=jnp.zeros(batch_dim),
            ball_touched=jnp.zeros(batch_dim),
            target_reached=jnp.zeros(batch_dim),
        )

        return world

    def reset_world_at(self, env_index: int | None):
        batch_size = self.world.batch_dim if env_index is None else 1

        # Place ball at starting position
        ball = self.world.entities[0]
        ball.state.pos = jnp.where(
            env_index is None,
            jnp.tile(jnp.array([-0.8, 0.0]), (batch_size, 1)),
            jnp.array([-0.8, 0.0]),
        )
        ball.state.vel = jnp.zeros_like(ball.state.vel)

        # Place target at random position on right side
        target = self.world.landmarks[0]
        target_x = jax.random.uniform(
            jax.random.PRNGKey(0),
            (batch_size if env_index is None else 1,),
            minval=0.5,
            maxval=0.8,
        )
        target_y = jax.random.uniform(
            jax.random.PRNGKey(1),
            (batch_size if env_index is None else 1,),
            minval=-0.5,
            maxval=0.5,
        )
        target.state.pos = jnp.where(
            env_index is None,
            jnp.stack([target_x, target_y], axis=-1),
            jnp.array([target_x[0], target_y[0]]),
        )

        # Place agents around the ball
        for i, agent in enumerate(self.world.agents):
            angle = i * (2 * jnp.pi / self.n_agents)
            radius = 0.2
            x_pos = ball.state.pos[:, 0] + radius * jnp.cos(angle)
            y_pos = ball.state.pos[:, 1] + radius * jnp.sin(angle)

            agent.state.pos = jnp.where(
                env_index is None,
                jnp.stack([x_pos, y_pos], axis=-1),
                jnp.array([x_pos[0], y_pos[0]]),
            )
            agent.state.vel = jnp.zeros_like(agent.state.vel)

        # Reset state
        self.state = BallTrajectoryState(
            agent_ball_distances=jnp.zeros((batch_size, self.n_agents)),
            ball_target_distance=jnp.zeros(batch_size),
            ball_touched=jnp.zeros(batch_size),
            target_reached=jnp.zeros(batch_size),
        )

    def reward(self, agent: Agent) -> Float[Array, f"{batch}"]:
        reward = jnp.zeros(self.world.batch_dim)

        # Get ball and target
        ball = self.world.entities[0]
        target = self.world.landmarks[0]

        # Calculate distances
        dist_to_ball = jnp.linalg.norm(agent.state.pos - ball.state.pos, axis=-1)
        ball_to_target = jnp.linalg.norm(ball.state.pos - target.state.pos, axis=-1)

        # Update state
        agent_idx = self.world.agents.index(agent)
        self.state = self.state.replace(
            agent_ball_distances=self.state.agent_ball_distances.at[:, agent_idx].set(
                dist_to_ball
            ),
            ball_target_distance=ball_to_target,
        )

        # Ball control reward
        touching_ball = dist_to_ball < self.ball_touch_distance
        reward = jnp.where(touching_ball, reward + 0.1, reward)

        # Update ball touched status
        self.state = self.state.replace(
            ball_touched=jnp.logical_or(touching_ball, self.state.ball_touched)
        )

        # Target reaching reward
        target_reached = ball_to_target < self.target_reach_distance
        reward = jnp.where(target_reached, reward + self.target_reward, reward)

        # Update target reached status
        self.state = self.state.replace(target_reached=target_reached)

        # Reward for moving ball closer to target
        reward -= 0.1 * ball_to_target  # Small reward for progress

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

        # Get ball and target states relative to agent
        ball = self.world.entities[0]
        target = self.world.landmarks[0]

        ball_pos = ball.state.pos - agent.state.pos
        ball_vel = ball.state.vel
        target_pos = target.state.pos - agent.state.pos

        # Get positions and velocities of other agents relative to this agent
        other_pos = []
        other_vel = []
        for other in self.world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.pos - agent.state.pos)
            other_vel.append(other.state.vel)

        # Stack observations
        obs = jnp.concatenate(
            [agent.state.pos]  # Own position
            + [agent.state.vel]  # Own velocity
            + [ball_pos]  # Ball position
            + [ball_vel]  # Ball velocity
            + [target_pos]  # Target position
            + other_pos  # Other agents' positions
            + other_vel  # Other agents' velocities
            + [jnp.array([self.state.target_reached])]  # Target reached status
            + [jnp.array([self.state.ball_touched])],  # Ball touched status
            axis=-1,
        )
        return obs

    def done(self) -> Float[Array, f"{batch}"]:
        # Episode ends when target is reached
        return self.state.target_reached
