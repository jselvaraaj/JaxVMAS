import jax
import jax.numpy as jnp

from jaxvmas.simulator.core import Agent, Landmark, World
from jaxvmas.simulator.scenario import BaseScenario
from jaxvmas.simulator.utils import Color, ScenarioUtils


class SimpleScenario(BaseScenario):
    def make_world(self, batch_dim: int, **kwargs):
        ScenarioUtils.check_kwargs_consumed(kwargs)
        # Make world
        world = World.create(batch_dim=batch_dim)
        # Add agents
        for i in range(1):
            agent = Agent.create(
                batch_dim=batch_dim,
                name=f"agent_{i}",
                dim_p=world.dim_p,
                dim_c=world.dim_c,
                collide=False,
                color=Color.GRAY,
            )
            world = world.add_agent(agent)
        # Add landmarks
        for i in range(1):
            landmark = Landmark.create(
                batch_dim=batch_dim,
                name=f"landmark {i}",
                collide=False,
                color=Color.RED,
            )
            world = world.add_landmark(landmark)

        return world

    def reset_world_at(self, seed=0, env_index: int = None):
        agents = []
        for agent in self.world.agents:
            agent = agent.set_pos(
                jax.random.uniform(
                    key=jax.random.PRNGKey(seed),
                    shape=(
                        (self.world.dim_p,)
                        if env_index is not None
                        else (self.world.batch_dim, self.world.dim_p)
                    ),
                    minval=-1.0,
                    maxval=1.0,
                ),
                batch_index=env_index,
            )
            agents.append(agent)
        world = self.world.replace(agents=agents)
        self = self.replace(world=world)

        landmarks = []
        for landmark in self.world.landmarks:
            landmark = landmark.set_pos(
                jax.random.uniform(
                    key=jax.random.PRNGKey(seed),
                    shape=(
                        (self.world.dim_p,)
                        if env_index is not None
                        else (self.world.batch_dim, self.world.dim_p)
                    ),
                    minval=-1.0,
                    maxval=1.0,
                ),
                batch_index=env_index,
            )
            landmarks.append(landmark)
        world = self.world.replace(landmarks=landmarks)
        self = self.replace(world=world)

        return self

    def reward(self, agent: Agent):
        dist2 = jnp.sum(
            jnp.square(agent.state.pos - self.world.landmarks[0].state.pos),
            axis=-1,
        )
        return -dist2

    def observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in self.world.landmarks:
            entity_pos.append(entity.state.pos - agent.state.pos)
        return jnp.concatenate([agent.state.vel, *entity_pos], axis=-1)
