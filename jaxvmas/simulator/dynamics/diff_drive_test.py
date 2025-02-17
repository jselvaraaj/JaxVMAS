import equinox as eqx
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from jaxvmas.simulator.core.agent import Agent
from jaxvmas.simulator.core.world import World
from jaxvmas.simulator.dynamics.diff_drive import DiffDrive


class TestDiffDriveDynamics:
    @pytest.fixture
    def world(self):
        return World.create(batch_dim=2, dt=0.1)

    @pytest.fixture
    def basic_dynamics(self, world: World):
        return DiffDrive.create(world=world)

    @pytest.fixture
    def basic_agent(self):
        return Agent.create(
            batch_dim=2,
            name="test_agent",
            dim_c=0,
            dim_p=2,
            action_size=3,  # Larger than needed_action_size for testing
        )

    def test_create(self, world: World):
        # Test creation with default integration
        dynamics = DiffDrive.create(world=world)
        assert isinstance(dynamics, DiffDrive)
        assert dynamics.needed_action_size == 2
        assert dynamics.integration == "rk4"
        assert dynamics.dt == world.dt

        # Test creation with euler integration
        dynamics = DiffDrive.create(world=world, integration="euler")
        assert dynamics.integration == "euler"

        # Test invalid integration method
        with pytest.raises(AssertionError):
            DiffDrive.create(world=world, integration="invalid")

    def test_reset(self, basic_dynamics: DiffDrive):
        # Test reset without index
        reset_dynamics = basic_dynamics.reset()
        assert isinstance(reset_dynamics, DiffDrive)

        # Test reset with index
        reset_dynamics_index = basic_dynamics.reset(index=0)
        assert isinstance(reset_dynamics_index, DiffDrive)

        # Test reset with array index
        reset_dynamics_array = basic_dynamics.reset(index=jnp.array([0, 1]))
        assert isinstance(reset_dynamics_array, DiffDrive)

    def test_check_and_process_action_valid(
        self, basic_dynamics: DiffDrive, basic_agent: Agent
    ):
        # Test valid action processing
        dynamics, agent = basic_dynamics.check_and_process_action(basic_agent)

        assert isinstance(dynamics, DiffDrive)
        assert isinstance(agent, Agent)
        assert agent.state.force.shape == (2, 2)
        assert agent.state.torque.shape == (2, 1)

    def test_check_and_process_action_invalid(self, basic_dynamics: DiffDrive):
        # Create agent with insufficient action size
        agent = Agent.create(
            batch_dim=2,
            name="test_agent",
            dim_c=0,
            dim_p=2,
            action_size=1,  # Less than needed_action_size
        )

        # Test that it raises ValueError for insufficient action size
        with pytest.raises(ValueError):
            basic_dynamics.check_and_process_action(agent)

    def test_integration_methods(self, basic_dynamics: DiffDrive):
        # Test f function
        state = jnp.array([[1.0, 2.0, 0.0], [3.0, 4.0, jnp.pi / 2]])
        u_command = jnp.array([1.0, 2.0])
        ang_vel_command = jnp.array([0.1, 0.2])

        result = basic_dynamics.f(state, u_command, ang_vel_command)
        assert result.shape == (2, 3)

        # Test euler integration
        euler_result = basic_dynamics.euler(state, u_command, ang_vel_command)
        assert euler_result.shape == (2, 3)

        # Test runge-kutta integration
        rk4_result = basic_dynamics.runge_kutta(state, u_command, ang_vel_command)
        assert rk4_result.shape == (2, 3)

    def test_is_jittable(self, basic_dynamics: DiffDrive, basic_agent: Agent):
        # Test jit compatibility of reset
        @eqx.filter_jit
        def reset_dynamics(dynamics: DiffDrive):
            return dynamics.reset()

        reset_result = reset_dynamics(basic_dynamics)
        assert isinstance(reset_result, DiffDrive)

        # Test jit compatibility of check_and_process_action
        @eqx.filter_jit
        def process_action(dynamics: DiffDrive, agent: Agent):
            return dynamics.check_and_process_action(agent)

        dynamics_result, agent_result = process_action(basic_dynamics, basic_agent)
        assert isinstance(dynamics_result, DiffDrive)
        assert isinstance(agent_result, Agent)

        # Test jit compatibility with array index
        @eqx.filter_jit
        def reset_with_index(dynamics: DiffDrive, index: Array):
            return dynamics.reset(index=index)

        reset_index_result = reset_with_index(basic_dynamics, jnp.array([0]))
        assert isinstance(reset_index_result, DiffDrive)

    def test_batch_processing(self, basic_dynamics: DiffDrive):
        # Test processing with different batch sizes
        for batch_dim in [1, 2, 4]:
            agent = Agent.create(
                batch_dim=batch_dim,
                name="test_agent",
                dim_c=0,
                dim_p=2,
                action_size=3,
            )

            dynamics, processed_agent = basic_dynamics.check_and_process_action(agent)
            assert processed_agent.state.force.shape == (batch_dim, 2)
            assert processed_agent.state.torque.shape == (batch_dim, 1)

    def test_zero_action(self, basic_dynamics: DiffDrive):
        # Test processing with zero actions
        agent = Agent.create(
            batch_dim=2,
            name="test_agent",
            dim_c=0,
            dim_p=2,
            action_size=3,
        )
        agent = agent.replace(action=agent.action.replace(u=jnp.zeros((2, 3))))

        dynamics, processed_agent = basic_dynamics.check_and_process_action(agent)
        assert jnp.allclose(processed_agent.state.force, 0)
        assert jnp.allclose(processed_agent.state.torque, 0)

    def test_straight_line_motion(self, basic_dynamics: DiffDrive):
        # Test straight line motion with forward velocity only
        agent = Agent.create(
            batch_dim=1,
            name="test_agent",
            dim_c=0,
            dim_p=2,
            action_size=2,
        )
        # Set forward velocity = 1, angular velocity = 0
        agent = agent.replace(action=agent.action.replace(u=jnp.array([[1.0, 0.0]])))

        dynamics, processed_agent = basic_dynamics.check_and_process_action(agent)

        # For straight line motion at 0 rotation:
        # - Force should be primarily in x direction
        # - Torque should be close to 0
        assert processed_agent.state.force[0, 0] > 0  # Positive x force
        assert jnp.abs(processed_agent.state.force[0, 1]) < 1e-5  # Near zero y force
        assert jnp.abs(processed_agent.state.torque[0, 0]) < 1e-5  # Near zero torque

    def test_rotation_motion(self, basic_dynamics: DiffDrive):
        # Test pure rotation with angular velocity only
        agent = Agent.create(
            batch_dim=1,
            name="test_agent",
            dim_c=0,
            dim_p=2,
            action_size=2,
        )
        # Set forward velocity = 0, angular velocity = 1
        agent = agent.replace(action=agent.action.replace(u=jnp.array([[0.0, 1.0]])))

        dynamics, processed_agent = basic_dynamics.check_and_process_action(agent)

        # For pure rotation:
        # - Forces should be close to 0
        # - Torque should be positive
        assert jnp.all(jnp.abs(processed_agent.state.force) < 1e-5)  # Near zero forces
        assert processed_agent.state.torque[0, 0] > 0  # Positive torque
