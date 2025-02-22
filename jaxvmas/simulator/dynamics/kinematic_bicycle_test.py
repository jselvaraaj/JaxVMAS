import equinox as eqx
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from jaxvmas.simulator.core.agent import Agent
from jaxvmas.simulator.dynamics.kinematic_bicycle import KinematicBicycle


class TestKinematicBicycleDynamics:
    @pytest.fixture
    def basic_dynamics(self):
        return KinematicBicycle.create(
            width=1.0,
            l_f=1.0,  # Front axle to COM distance
            l_r=1.0,  # Rear axle to COM distance
            max_steering_angle=jnp.pi / 4,  # 45 degrees
            dt=0.1,
            integration="rk4",
        )

    @pytest.fixture
    def basic_agent(self):
        agent = Agent.create(
            name="test_agent",
            action_size=3,  # Larger than needed_action_size for testing
        )
        agent = agent._spawn(id=1, batch_dim=2, dim_c=0, dim_p=2)
        return agent

    def test_create(self):
        # Test creation with default integration
        dynamics = KinematicBicycle.create(
            width=1.0, l_f=1.0, l_r=1.0, max_steering_angle=jnp.pi / 4, dt=0.1
        )
        assert isinstance(dynamics, KinematicBicycle)
        assert dynamics.needed_action_size == 2
        assert dynamics.integration == "rk4"
        assert dynamics.dt == 0.1
        assert dynamics.width == 1.0
        assert dynamics.l_f == 1.0
        assert dynamics.l_r == 1.0
        assert dynamics.max_steering_angle == jnp.pi / 4

        # Test creation with euler integration
        dynamics = KinematicBicycle.create(
            width=1.0,
            l_f=1.0,
            l_r=1.0,
            max_steering_angle=jnp.pi / 4,
            dt=0.1,
            integration="euler",
        )
        assert dynamics.integration == "euler"

        # Test invalid integration method
        with pytest.raises(AssertionError):
            KinematicBicycle.create(
                width=1.0,
                l_f=1.0,
                l_r=1.0,
                max_steering_angle=jnp.pi / 4,
                dt=0.1,
                integration="invalid",
            )

    def test_reset(self, basic_dynamics: KinematicBicycle):
        # Test reset without index
        reset_dynamics = basic_dynamics.reset()
        assert isinstance(reset_dynamics, KinematicBicycle)

        # Test reset with index
        reset_dynamics_index = basic_dynamics.reset(index=0)
        assert isinstance(reset_dynamics_index, KinematicBicycle)

        # Test reset with array index
        reset_dynamics_array = basic_dynamics.reset(index=jnp.array([0, 1]))
        assert isinstance(reset_dynamics_array, KinematicBicycle)

    def test_check_and_process_action_valid(
        self, basic_dynamics: KinematicBicycle, basic_agent: Agent
    ):
        # Test valid action processing
        dynamics, agent = basic_dynamics.check_and_process_action(basic_agent)

        assert isinstance(dynamics, KinematicBicycle)
        assert isinstance(agent, Agent)
        assert agent.state.force.shape == (2, 2)
        assert agent.state.torque.shape == (2, 1)

    def test_check_and_process_action_invalid(self, basic_dynamics: KinematicBicycle):
        # Create agent with insufficient action size
        agent = Agent.create(
            name="test_agent",
            action_size=1,  # Less than needed_action_size
        )
        agent = agent._spawn(id=1, batch_dim=2, dim_c=0, dim_p=2)

        # Test that it raises ValueError for insufficient action size
        with pytest.raises(ValueError):
            basic_dynamics.check_and_process_action(agent)

    def test_steering_angle_clipping(self, basic_dynamics: KinematicBicycle):
        agent = Agent.create(
            name="test_agent",
            action_size=2,
        )
        agent = agent._spawn(id=1, batch_dim=2, dim_c=0, dim_p=2)

        # Test with steering angles beyond limits
        large_steering = jnp.array([[1.0, 2 * jnp.pi], [1.0, -2 * jnp.pi]])
        agent = agent.replace(action=agent.action.replace(u=large_steering))

        dynamics, processed_agent = basic_dynamics.check_and_process_action(agent)

        # Forces and torques should be finite (not NaN) due to clipping
        assert jnp.all(jnp.isfinite(processed_agent.state.force))
        assert jnp.all(jnp.isfinite(processed_agent.state.torque))

    def test_is_jittable(self, basic_dynamics: KinematicBicycle, basic_agent: Agent):
        # Test jit compatibility of reset
        @eqx.filter_jit
        def reset_dynamics(dynamics: KinematicBicycle):
            return dynamics.reset()

        reset_result = reset_dynamics(basic_dynamics)
        assert isinstance(reset_result, KinematicBicycle)

        # Test jit compatibility of check_and_process_action
        @eqx.filter_jit
        def process_action(dynamics: KinematicBicycle, agent: Agent):
            return dynamics.check_and_process_action(agent)

        dynamics_result, agent_result = process_action(basic_dynamics, basic_agent)
        assert isinstance(dynamics_result, KinematicBicycle)
        assert isinstance(agent_result, Agent)

        # Test jit compatibility with array index
        @eqx.filter_jit
        def reset_with_index(dynamics: KinematicBicycle, index: Array):
            return dynamics.reset(index=index)

        reset_index_result = reset_with_index(basic_dynamics, jnp.array([0]))
        assert isinstance(reset_index_result, KinematicBicycle)

    def test_batch_processing(self, basic_dynamics: KinematicBicycle):
        # Test processing with different batch sizes
        for batch_dim in [1, 2, 4]:
            agent = Agent.create(
                name="test_agent",
                action_size=3,
            )
            agent = agent._spawn(id=1, batch_dim=batch_dim, dim_c=0, dim_p=2)

            dynamics, processed_agent = basic_dynamics.check_and_process_action(agent)
            assert processed_agent.state.force.shape == (batch_dim, 2)
            assert processed_agent.state.torque.shape == (batch_dim, 1)

    def test_zero_action(self, basic_dynamics: KinematicBicycle):
        # Test processing with zero actions
        agent = Agent.create(
            name="test_agent",
            action_size=3,
        )
        agent = agent._spawn(id=1, batch_dim=2, dim_c=0, dim_p=2)
        agent = agent.replace(action=agent.action.replace(u=jnp.zeros((2, 3))))

        dynamics, processed_agent = basic_dynamics.check_and_process_action(agent)
        assert jnp.allclose(processed_agent.state.force, 0)
        assert jnp.allclose(processed_agent.state.torque, 0)

    def test_straight_line_motion(self, basic_dynamics: KinematicBicycle):
        # Test straight line motion with forward velocity only
        agent = Agent.create(
            name="test_agent",
            action_size=2,
        )
        agent = agent._spawn(id=1, batch_dim=1, dim_c=0, dim_p=2)
        # Set forward velocity = 1, steering angle = 0
        agent = agent.replace(action=agent.action.replace(u=jnp.array([[1.0, 0.0]])))

        dynamics, processed_agent = basic_dynamics.check_and_process_action(agent)

        # For straight line motion at 0 steering angle:
        # - Force should be primarily in x direction
        # - Torque should be close to 0
        assert processed_agent.state.force[0, 0] > 0  # Positive x force
        assert jnp.abs(processed_agent.state.force[0, 1]) < 1e-5  # Near zero y force
        assert jnp.abs(processed_agent.state.torque[0, 0]) < 1e-5  # Near zero torque

    def test_turning_motion(self, basic_dynamics: KinematicBicycle):
        # Test turning motion with forward velocity and steering
        agent = Agent.create(
            name="test_agent",
            action_size=2,
        )
        agent = agent._spawn(id=1, batch_dim=1, dim_c=0, dim_p=2)

        # Set forward velocity = 1, steering angle = pi/4
        agent = agent.replace(
            action=agent.action.replace(u=jnp.array([[1.0, jnp.pi / 4]]))
        )

        dynamics, processed_agent = basic_dynamics.check_and_process_action(agent)

        # For turning motion:
        # - Both x and y forces should be non-zero
        # - Torque should be non-zero
        assert jnp.abs(processed_agent.state.force[0, 0]) > 0  # Non-zero x force
        assert jnp.abs(processed_agent.state.force[0, 1]) > 0  # Non-zero y force
        assert jnp.abs(processed_agent.state.torque[0, 0]) > 0  # Non-zero torque
