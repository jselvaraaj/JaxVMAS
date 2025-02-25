from typing import Sequence

import chex
import jax.numpy as jnp
import pytest

from jaxvmas.simulator.core.agent import Agent
from jaxvmas.simulator.core.shapes import Box, Shape, Sphere
from jaxvmas.simulator.dynamics.common import Dynamics
from jaxvmas.simulator.dynamics.holonomic import Holonomic
from jaxvmas.simulator.sensors import Sensor
from jaxvmas.simulator.utils import Color


class MockSensor(Sensor):
    @classmethod
    def create(cls):
        return cls()

    def measure(self):
        return jnp.zeros((2,))

    def render(self):
        return []


class TestAgent:
    @pytest.fixture
    def basic_agent(self):
        agent = Agent.create(
            name="test_agent",
            movable=True,
            rotatable=True,
        )
        agent = agent._spawn(
            id=1,
            batch_dim=2,
            dim_c=3,
            dim_p=2,
        )
        chex.block_until_chexify_assertions_complete()
        return agent

    @pytest.mark.parametrize(
        "shape,color,alpha",
        [
            (Sphere(radius=0.5), Color.RED, 0.7),
            (Box(width=1.0), Color.GREEN, 0.3),
            (None, Color.BLUE, 0.5),
        ],
    )
    def test_create_with_different_shapes(
        self, shape: Shape, color: Color, alpha: float
    ):
        agent = Agent.create(
            name="shape_test",
            shape=shape,
            color=color,
            alpha=alpha,
        )
        agent = agent._spawn(
            id=2,
            batch_dim=2,
            dim_c=3,
            dim_p=2,
        )
        chex.block_until_chexify_assertions_complete()

        assert agent.color == color
        assert agent.alpha == alpha
        if shape:
            assert isinstance(agent.shape, type(shape))

    @pytest.mark.parametrize(
        "dynamics,expected_action_size",
        [
            (Holonomic(), 2),
            (Holonomic(), 2),
        ],
    )
    def test_different_dynamics(self, dynamics: Dynamics, expected_action_size: int):
        agent = Agent.create(
            name="dynamics_test",
            dynamics=dynamics,
        )
        agent = agent._spawn(
            id=2,
            batch_dim=2,
            dim_c=3,
            dim_p=2,
        )
        chex.block_until_chexify_assertions_complete()
        assert isinstance(agent.dynamics, type(dynamics))
        assert agent.action_size == expected_action_size

    def test_sensor_integration(self):
        sensors = [MockSensor.create(), MockSensor.create()]
        agent = Agent.create(
            name="sensor_test",
            sensors=sensors,
        )
        agent = agent._spawn(
            id=2,
            batch_dim=2,
            dim_c=3,
            dim_p=2,
        )
        assert len(agent.sensors) == 2
        assert all(isinstance(s, MockSensor) for s in agent.sensors)

    def test_spawn_behavior(self, basic_agent: Agent):
        # Test spawn with different dimensions
        basic_agent = Agent.create(name="one")
        basic_agent = basic_agent.replace(silent=False)
        spawned = basic_agent._spawn(id=2, batch_dim=2, dim_c=5, dim_p=2)
        assert spawned.state.c.shape == (2, 5)
        assert spawned.state.pos.shape == (2, 2)

        # Test spawn with silent agent and zero comm dimension
        basic_agent = Agent.create(name="one")
        silent_agent = basic_agent.replace(silent=True)
        spawned_silent = silent_agent._spawn(id=0, batch_dim=2, dim_c=0, dim_p=2)
        assert spawned_silent.state.c.shape == (2, 0)

    @pytest.mark.parametrize(
        "u_range,u_multiplier,u_noise",
        [
            (1.0, 1.0, 0.0),
            ([1.0, 2.0], [0.5, 1.0], [0.1, 0.1]),
        ],
    )
    def test_action_configurations(
        self,
        u_range: float | Sequence[float],
        u_multiplier: float | Sequence[float],
        u_noise: float | Sequence[float],
    ):
        agent = Agent.create(
            name="action_test",
            u_range=u_range,
            u_multiplier=u_multiplier,
            u_noise=u_noise,
        )

        agent = agent._spawn(
            id=2,
            batch_dim=2,
            dim_c=3,
            dim_p=2,
        )
        assert agent.action.u_range == u_range
        assert agent.action.u_multiplier == u_multiplier
        assert agent.action.u_noise == u_noise

    def test_render_with_action(self):
        agent = Agent.create(
            name="render_test",
            render_action=True,
            shape=Sphere(radius=0.5),
        )
        agent = agent._spawn(
            id=2,
            batch_dim=2,
            dim_c=3,
            dim_p=2,
        )

        # Add non-zero force to test force rendering
        agent = agent.replace(
            state=agent.state.replace(force=jnp.array([[1.0, 1.0], [1.0, 1.0]]))
        )

        geoms = agent.render(env_index=0)
        assert len(geoms) > 1  # Should have shape + force line

    def test_invalid_discrete_actions(self):
        # Test invalid discrete action configuration
        with pytest.raises(ValueError):
            Agent.create(
                name="invalid",
                discrete_action_nvec=[1, 2],  # Invalid: must be > 1
            )

    def test_reset_state(self, basic_agent: Agent):
        # Modify agent state
        modified_agent = basic_agent.replace(
            state=basic_agent.state.replace(
                pos=jnp.ones((2, 2)),
                vel=jnp.ones((2, 2)),
            ),
            action=basic_agent.action.replace(
                u=jnp.ones((2, 2)),
            ),
        )

        # Test reset
        reset_agent = modified_agent._reset(env_index=jnp.asarray([0]))
        assert jnp.all(reset_agent.state.pos[0] == 0)
        assert jnp.all(reset_agent.state.vel[0] == 0)
        assert jnp.all(reset_agent.action.u[0] == 0)
        # Other batch elements should remain unchanged
        assert jnp.all(reset_agent.state.pos[1] == 1)
