import equinox as eqx
import jax.numpy as jnp
import pytest

from jaxvmas.simulator.core.agent import Agent
from jaxvmas.simulator.sensors import Sensor


class TestAgent:
    @pytest.fixture
    def basic_agent(self):
        # Create basic agent with minimal configuration
        return Agent.create(
            batch_dim=2,
            name="test_agent",
            dim_c=3,
            dim_p=2,
            movable=True,
            rotatable=True,
        )

    def test_create_basic(self, basic_agent: Agent):
        # Test basic properties
        assert basic_agent.name == "test_agent"
        assert basic_agent.movable is True
        assert basic_agent.rotatable is True
        assert basic_agent.silent is True
        assert basic_agent.adversary is False
        assert basic_agent.alpha == 0.5
        assert basic_agent.action_size == 2  # Default for Holonomic dynamics
        assert basic_agent.discrete_action_nvec == [
            3,
            3,
        ]  # Default 3-way discretization

    def test_create_with_custom_config(self):
        # Test creation with custom configuration
        agent = Agent.create(
            batch_dim=2,
            name="custom_agent",
            dim_c=4,
            dim_p=2,
            obs_range=10.0,
            obs_noise=0.1,
            f_range=5.0,
            max_f=10.0,
            silent=False,
            action_size=3,
            discrete_action_nvec=[3, 4, 5],
        )

        assert agent.obs_range == 10.0
        assert agent.obs_noise == 0.1
        assert agent.f_range == 5.0
        assert agent.max_f == 10.0
        assert agent.silent is False
        assert agent.action_size == 3
        assert agent.discrete_action_nvec == [3, 4, 5]

    def test_invalid_configurations(self):

        class MockSensor(Sensor):
            @classmethod
            def create(cls):
                return cls()

            def measure(self):
                return jnp.zeros((2,))

            def render(self):
                return []

        Agent.create(
            batch_dim=2,
            name="invalid",
            dim_c=3,
            dim_p=2,
            obs_range=0.0,
            sensors=[MockSensor.create()],
        )

        # Test inconsistent action_size and discrete_action_nvec
        with pytest.raises(ValueError):
            Agent.create(
                batch_dim=2,
                name="invalid",
                dim_c=3,
                dim_p=2,
                action_size=2,
                discrete_action_nvec=[3, 3, 3],
            )

    def test_reset_and_spawn(self, basic_agent: Agent):
        # Test spawn
        spawned_agent = basic_agent._spawn(dim_c=4, dim_p=2)
        assert spawned_agent.state.pos.shape == (2, 2)
        assert spawned_agent.state.vel.shape == (2, 2)

        # Test spawn with silent agent
        silent_agent = basic_agent.replace(silent=True)
        spawned_silent = silent_agent._spawn(dim_c=4, dim_p=2)
        assert jnp.allclose(
            spawned_silent.state.c, jnp.full_like(spawned_silent.state.c, jnp.nan)
        )

        # Test reset
        reset_agent = basic_agent._reset(env_index=0)
        assert jnp.all(reset_agent.state.pos[0] == 0)
        assert jnp.all(reset_agent.state.vel[0] == 0)
        assert jnp.all(reset_agent.action.u[0] == 0)

    def test_render(self, basic_agent: Agent):
        # Test basic rendering
        geoms = basic_agent.render(env_index=0)
        assert len(geoms) > 0  # Should have at least one geometry

        # Test rendering with action visualization
        agent_with_render = basic_agent.replace(render_action=True)
        agent_with_render = agent_with_render.replace(
            state=agent_with_render.state.replace(
                force=jnp.ones((2, 2))  # Non-zero force
            )
        )
        geoms_with_action = agent_with_render.render(env_index=0)
        assert len(geoms_with_action) > len(
            geoms
        )  # Should have additional line for force

    def test_is_jittable(self, basic_agent: Agent):
        @eqx.filter_jit
        def f(agent):
            return agent._spawn(dim_c=3, dim_p=2)

        f(basic_agent)

        @eqx.filter_jit
        def f2(agent):
            return agent._reset(env_index=0)

        f2(basic_agent)
