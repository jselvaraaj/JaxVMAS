import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float, Int

from jaxvmas.simulator.core.agent import Agent
from jaxvmas.simulator.core.landmark import Landmark
from jaxvmas.simulator.core.world import World
from jaxvmas.simulator.rendering import Geom, Line
from jaxvmas.simulator.scenario import BaseScenario, pos
from jaxvmas.simulator.utils import INITIAL_VIEWER_SIZE, VIEWER_DEFAULT_ZOOM

# Define dimensions for type hints
dim1 = "dim1"
dim2 = "dim2"
batch_axis_dim = "batch_axis_dim"
env_index_dim = "env_index_dim"


class MockScenario(BaseScenario):
    """A simple scenario for testing."""

    def make_world(self, batch_dim: int, **kwargs):
        world = World.create(batch_dim=batch_dim, **kwargs)
        agent = Agent.create(name="test_agent")
        landmark = Landmark.create(name="test_landmark")
        world = world.add_agent(agent)
        world = world.add_landmark(landmark)
        self = self.replace(world=world)
        return self

    def reset_world_at(
        self, PRNG_key: Array, env_index: Int[Array, f"{env_index_dim}"] | None
    ) -> "MockScenario":
        agent = self.world.agents[0]
        if env_index is None:
            agent = agent.set_pos(jnp.ones((self.world.batch_dim, 2)))
        else:
            agent = agent.set_pos(jnp.ones(2), batch_index=env_index)
        self = self.replace(world=self.world.replace(agents=[agent]))
        return self

    def observation(self, agent: Agent) -> Float[Array, f"{batch_axis_dim} {pos}"]:
        return agent.state.pos

    def reward(self, agent: Agent) -> Float[Array, f"{batch_axis_dim}"]:
        return jnp.zeros(self.world.batch_dim)

    def info(self, agent: Agent) -> dict[str, Array]:
        return {"test_info": jnp.ones(self.world.batch_dim)}

    def extra_render(self, env_index: int = 0) -> list[Geom]:
        return [Line(start=(0, 0), end=(1, 1))]

    def process_action(self, agent: Agent) -> tuple["MockScenario", Agent]:
        return self, agent

    def pre_step(self) -> "MockScenario":
        return self

    def post_step(self) -> "MockScenario":
        return self


class TestBaseScenario:
    @pytest.fixture
    def scenario(self):
        scenario = MockScenario.create()
        scenario = scenario.env_make_world(batch_dim=2, dim_c=2, dim_p=2)
        return scenario

    def test_create(self):
        scenario = MockScenario.create()
        assert scenario.world is None
        assert scenario.viewer_size == INITIAL_VIEWER_SIZE
        assert scenario.viewer_zoom == VIEWER_DEFAULT_ZOOM
        assert scenario.render_origin == (0.0, 0.0)
        assert scenario.plot_grid is False
        assert scenario.grid_spacing == 0.1
        assert scenario.visualize_semidims is True

    def test_make_world(self, scenario: MockScenario):
        assert isinstance(scenario.world, World)
        assert len(scenario.world.agents) == 1
        assert len(scenario.world.landmarks) == 1
        assert scenario.world.batch_dim == 2

    def test_reset_world(self, scenario: MockScenario):
        PRNG_key = jax.random.PRNGKey(0)
        # Test reset with specific index
        scenario = scenario.env_reset_world_at(
            PRNG_key=PRNG_key, env_index=jnp.asarray([0])
        )
        assert jnp.all(scenario.world.agents[0].state.pos[0] == 1.0)
        assert not jnp.all(scenario.world.agents[0].state.pos[1] == 1.0)

        # Test reset all environments
        scenario = scenario.env_reset_world_at(PRNG_key=PRNG_key, env_index=None)
        assert jnp.all(scenario.world.agents[0].state.pos == 1.0)

    def test_observation(self, scenario: MockScenario):
        obs = scenario.observation(scenario.world.agents[0])
        assert obs.shape == (2, 2)
        assert jnp.array_equal(obs, scenario.world.agents[0].state.pos)

    def test_reward(self, scenario: MockScenario):
        reward = scenario.reward(scenario.world.agents[0])
        assert reward.shape == (2,)
        assert jnp.all(reward == 0)

    def test_info(self, scenario: MockScenario):
        info = scenario.info(scenario.world.agents[0])
        assert "test_info" in info
        assert info["test_info"].shape == (2,)
        assert jnp.all(info["test_info"] == 1)

    def test_extra_render(self, scenario: MockScenario):
        geoms = scenario.extra_render()
        assert len(geoms) == 1
        assert isinstance(geoms[0], Line)

    def test_process_action(self, scenario: MockScenario):
        agent = scenario.world.agents[0]
        new_scenario, new_agent = scenario.process_action(agent)
        assert isinstance(new_scenario, MockScenario)
        assert isinstance(new_agent, Agent)

    def test_pre_step(self, scenario: MockScenario):
        new_scenario = scenario.pre_step()
        assert isinstance(new_scenario, MockScenario)

    def test_post_step(self, scenario: MockScenario):
        new_scenario = scenario.post_step()
        assert isinstance(new_scenario, MockScenario)

    def test_done(self, scenario: MockScenario):
        done = scenario.done()
        assert done.shape == (2,)
        assert not jnp.any(done)

    def test_is_jittable(self, scenario: MockScenario):
        # Test jit compatibility of observation
        @eqx.filter_jit
        def get_observation(scenario: MockScenario, agent: Agent):
            return scenario.observation(agent)

        obs = get_observation(scenario, scenario.world.agents[0])
        assert obs.shape == (2, 2)

        # Test jit compatibility of reward
        @eqx.filter_jit
        def get_reward(scenario: MockScenario, agent: Agent):
            return scenario.reward(agent)

        reward = get_reward(scenario, scenario.world.agents[0])
        assert reward.shape == (2,)

        # Test jit compatibility of info
        @eqx.filter_jit
        def get_info(scenario: MockScenario, agent: Agent):
            return scenario.info(agent)

        info = get_info(scenario, scenario.world.agents[0])
        assert "test_info" in info

        # Test jit compatibility of reset
        @eqx.filter_jit
        def reset_scenario(
            scenario: MockScenario, PRNG_key: Array, env_index: Int[Array, ""] | None
        ):
            return scenario.env_reset_world_at(PRNG_key=PRNG_key, env_index=env_index)

        PRNG_key = jax.random.PRNGKey(0)
        reset_scen = reset_scenario(scenario, PRNG_key, jnp.asarray([0]))
        assert jnp.all(reset_scen.world.agents[0].state.pos[0] == 1.0)

        # Test jit compatibility of process_action
        @eqx.filter_jit
        def process_agent_action(scenario: MockScenario, agent: Agent):
            return scenario.process_action(agent)

        new_scen, new_agent = process_agent_action(scenario, scenario.world.agents[0])
        assert isinstance(new_scen, MockScenario)
        assert isinstance(new_agent, Agent)

        # Test jit compatibility of pre/post step
        @eqx.filter_jit
        def step_hooks(scenario: MockScenario):
            scenario = scenario.pre_step()
            scenario = scenario.post_step()
            return scenario

        stepped_scen = step_hooks(scenario)
        assert isinstance(stepped_scen, MockScenario)

        # Test jit compatibility of done
        @eqx.filter_jit
        def get_done(scenario: MockScenario):
            return scenario.done()

        done = get_done(scenario)
        assert done.shape == (2,)
