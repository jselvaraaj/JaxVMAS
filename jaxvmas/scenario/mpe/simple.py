import chex
import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Int, PRNGKeyArray, jaxtyped

from jaxvmas.interactive_rendering import render_interactively
from jaxvmas.simulator.core import Agent, Landmark, World
from jaxvmas.simulator.scenario import BaseScenario
from jaxvmas.simulator.utils import Color, ScenarioUtils

batch_axis_dim = "batch_axis_dim"


@jaxtyped(typechecker=beartype)
class Scenario(BaseScenario):
    """A simple scenario with a single agent and a single landmark."""

    @jaxtyped(typechecker=beartype)
    @chex.assert_max_traces(0)
    def make_world(self, batch_dim: int, **kwargs):
        ScenarioUtils.check_kwargs_consumed(kwargs)
        # Make world
        world = World.create(batch_dim=batch_dim)

        # Add agents
        for i in range(1):
            agent = Agent.create(
                name=f"agent_{i}",
                collide=False,
                color=Color.GRAY,
            )
            world = world.add_agent(agent)
        # Add landmarks
        for i in range(1):
            landmark = Landmark.create(
                name=f"landmark {i}",
                collide=False,
                color=Color.RED,
            )
            world = world.add_landmark(landmark)

        self = self.replace(world=world)
        return self

    @jaxtyped(typechecker=beartype)
    def reset_world_at(
        self,
        PRNG_key: PRNGKeyArray,
        env_index: Int[Array, f"{batch_axis_dim}"] | Int[Array, ""] = jnp.asarray(-1),
    ):
        agents = []
        for agent in self.world.agents:
            PRNG_key, PRNG_key_agent = jax.random.split(PRNG_key)
            agent = agent.set_pos(
                jax.lax.cond(
                    env_index == -1,
                    lambda: jax.random.uniform(
                        key=PRNG_key_agent,
                        shape=(self.world.batch_dim, self.world.dim_p),
                        minval=-1.0,
                        maxval=1.0,
                    ),
                    lambda: jnp.broadcast_to(
                        jax.random.uniform(
                            key=PRNG_key_agent,
                            shape=(self.world.dim_p,),
                            minval=-1.0,
                            maxval=1.0,
                        ),
                        (self.world.batch_dim, self.world.dim_p),
                    ),
                ),
                batch_index=env_index,
            )
            agents.append(agent)
        world = self.world.replace(agents=agents)
        self = self.replace(world=world)

        landmarks = []
        for landmark in self.world.landmarks:
            PRNG_key, PRNG_key_landmark = jax.random.split(PRNG_key)
            landmark = landmark.set_pos(
                jax.lax.cond(
                    env_index == -1,
                    lambda: jax.random.uniform(
                        key=PRNG_key_landmark,
                        shape=(self.world.batch_dim, self.world.dim_p),
                        minval=-1.0,
                        maxval=1.0,
                    ),
                    lambda: jnp.broadcast_to(
                        jax.random.uniform(
                            key=PRNG_key_landmark,
                            shape=(self.world.dim_p,),
                            minval=-1.0,
                            maxval=1.0,
                        ),
                        (
                            self.world.batch_dim,
                            self.world.dim_p,
                        ),  # This is fine since we we will only use the value from the env_index
                    ),
                ),
                batch_index=env_index,
            )
            landmarks.append(landmark)
        world = self.world.replace(landmarks=landmarks)
        self = self.replace(world=world)

        return self

    @jaxtyped(typechecker=beartype)
    def reward(self, agent: Agent):
        dist2 = jnp.sum(
            jnp.square(agent.state.pos - self.world.landmarks[0].state.pos),
            axis=-1,
        )
        return -dist2

    @jaxtyped(typechecker=beartype)
    def observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in self.world.landmarks:
            entity_pos.append(entity.state.pos - agent.state.pos)
        return jnp.concatenate([agent.state.vel, *entity_pos], axis=-1)


if __name__ == "__main__":
    render_interactively(__file__)
