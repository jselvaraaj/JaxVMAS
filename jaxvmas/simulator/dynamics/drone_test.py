import equinox as eqx
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from jaxvmas.simulator.core.agent import Agent
from jaxvmas.simulator.core.world import World
from jaxvmas.simulator.dynamics.drone import Drone


class TestDroneDynamics:
    @pytest.fixture
    def world(self):
        return World.create(dt=0.1, batch_dim=2)

    @pytest.fixture
    def basic_dynamics(self, world: World):
        return Drone.create(world=world)

    @pytest.fixture
    def basic_agent(self):
        return Agent.create(
            batch_dim=2,
            name="test_agent",
            dim_c=0,
            dim_p=2,
            action_size=5,  # Larger than needed_action_size for testing
        )

    def test_create(self, world: World):
        # Test creation with default parameters
        dynamics = Drone.create(world=world)
        assert isinstance(dynamics, Drone)
        assert dynamics.needed_action_size == 4
        assert dynamics.integration == "rk4"
        assert dynamics.dt == world.dt
        assert dynamics.I_xx == 8.1e-3
        assert dynamics.I_yy == 8.1e-3
        assert dynamics.I_zz == 14.2e-3
        assert dynamics.g == 9.81
        assert dynamics.batch_dim == world.batch_dim
        assert dynamics.drone_state.shape == (world.batch_dim, 12)

        # Test creation with custom parameters
        dynamics = Drone.create(
            world=world, I_xx=0.1, I_yy=0.2, I_zz=0.3, integration="euler"
        )
        assert dynamics.integration == "euler"
        assert dynamics.I_xx == 0.1
        assert dynamics.I_yy == 0.2
        assert dynamics.I_zz == 0.3

        # Test invalid integration method
        with pytest.raises(AssertionError):
            Drone.create(world=world, integration="invalid")

    def test_reset(self, basic_dynamics: Drone):
        # Test reset without index
        reset_dynamics = basic_dynamics.reset()
        assert isinstance(reset_dynamics, Drone)
        assert jnp.all(reset_dynamics.drone_state == 0)

        # Test reset with index
        reset_dynamics_index = basic_dynamics.reset(index=0)
        assert isinstance(reset_dynamics_index, Drone)

        # Test reset with array index
        reset_dynamics_array = basic_dynamics.reset(index=jnp.array([0, 1]))
        assert isinstance(reset_dynamics_array, Drone)

    def test_check_and_process_action_valid(
        self, basic_dynamics: Drone, basic_agent: Agent
    ):
        # Test valid action processing
        dynamics, agent = basic_dynamics.check_and_process_action(basic_agent)

        assert isinstance(dynamics, Drone)
        assert isinstance(agent, Agent)
        assert agent.state.force.shape == (2, 2)
        assert agent.state.torque.shape == (2, 1)

    def test_check_and_process_action_invalid(self, basic_dynamics: Drone):
        # Create agent with insufficient action size
        agent = Agent.create(
            batch_dim=2,
            name="test_agent",
            dim_c=0,
            dim_p=2,
            action_size=3,  # Less than needed_action_size
        )

        # Test that it raises ValueError for insufficient action size
        with pytest.raises(ValueError):
            basic_dynamics.check_and_process_action(agent)

    def test_needs_reset(self, basic_dynamics: Drone):
        # Test with angles within limits
        basic_dynamics = basic_dynamics.replace(drone_state=jnp.zeros((2, 12)))
        assert not jnp.any(basic_dynamics.needs_reset())

        # Test with angles exceeding limits (>30 degrees)
        large_angles = jnp.zeros((2, 12))
        large_angles = large_angles.at[:, 0].set(
            45 * (jnp.pi / 180)
        )  # Roll > 30 degrees
        basic_dynamics = basic_dynamics.replace(drone_state=large_angles)
        assert jnp.all(basic_dynamics.needs_reset())

    def test_integration_methods(self, basic_dynamics: Drone, basic_agent: Agent):
        # Test f function
        state = jnp.zeros((2, 12))
        thrust = jnp.ones(2)
        torque = jnp.ones((2, 3))

        result = basic_dynamics.f(basic_agent, state, thrust, torque)
        assert result.shape == (2, 12)

        # Test euler integration
        euler_result = basic_dynamics.euler(basic_agent, state, thrust, torque)
        assert euler_result.shape == (2, 12)

        # Test runge-kutta integration
        rk4_result = basic_dynamics.runge_kutta(basic_agent, state, thrust, torque)
        assert rk4_result.shape == (2, 12)

    def test_is_jittable(self, basic_dynamics: Drone, basic_agent: Agent):
        # Test jit compatibility of reset
        @eqx.filter_jit
        def reset_dynamics(dynamics: Drone):
            return dynamics.reset()

        reset_result = reset_dynamics(basic_dynamics)
        assert isinstance(reset_result, Drone)

        # Test jit compatibility of check_and_process_action
        @eqx.filter_jit
        def process_action(dynamics: Drone, agent: Agent):
            return dynamics.check_and_process_action(agent)

        dynamics_result, agent_result = process_action(basic_dynamics, basic_agent)
        assert isinstance(dynamics_result, Drone)
        assert isinstance(agent_result, Agent)

        # Test jit compatibility with array index
        @eqx.filter_jit
        def reset_with_index(dynamics: Drone, index: Array):
            return dynamics.reset(index=index)

        reset_index_result = reset_with_index(basic_dynamics, jnp.array([0]))
        assert isinstance(reset_index_result, Drone)

    def test_batch_processing(self, basic_dynamics: Drone):
        # Test processing with different batch sizes
        for batch_dim in [1, 2, 4]:
            world = World.create(dt=0.1, batch_dim=batch_dim)
            dynamics = Drone.create(world=world)
            agent = Agent.create(
                batch_dim=batch_dim,
                name="test_agent",
                dim_c=0,
                dim_p=2,
                action_size=5,
            )

            dynamics, processed_agent = dynamics.check_and_process_action(agent)
            assert processed_agent.state.force.shape == (batch_dim, 2)
            assert processed_agent.state.torque.shape == (batch_dim, 1)
            assert dynamics.drone_state.shape == (batch_dim, 12)

    def test_zero_action(self, basic_dynamics: Drone):
        # Test processing with zero actions
        agent = Agent.create(
            batch_dim=2,
            name="test_agent",
            dim_c=0,
            dim_p=2,
            action_size=5,
        )
        agent = agent.replace(action=agent.action.replace(u=jnp.zeros((2, 5))))

        dynamics, processed_agent = basic_dynamics.check_and_process_action(agent)
        # Note: Due to gravity compensation, force in z direction won't be zero
        assert jnp.allclose(processed_agent.state.force[:, 0], 0)  # x force
        assert jnp.allclose(processed_agent.state.force[:, 1], 0)  # y force
        assert jnp.allclose(processed_agent.state.torque, 0)

    def test_hover_action(self, basic_dynamics: Drone):
        # Test hovering (thrust = mg, no torques)
        agent = Agent.create(
            batch_dim=basic_dynamics.batch_dim,
            name="test_agent",
            dim_c=0,
            dim_p=2,
            action_size=4,
        )
        # Set thrust = 0 (will be compensated by mg), zero torques
        hover_action = jnp.zeros((basic_dynamics.batch_dim, 4))
        agent = agent.replace(action=agent.action.replace(u=hover_action))

        dynamics, processed_agent = basic_dynamics.check_and_process_action(agent)

        # For perfect hover:
        # - Forces should be near zero (gravity compensated)
        # - Torque should be zero
        assert jnp.allclose(processed_agent.state.force, 0, atol=1e-5)
        assert jnp.allclose(processed_agent.state.torque, 0, atol=1e-5)

    def test_state_update(self, basic_dynamics: Drone):
        # Test that drone state is properly updated from agent state
        agent = Agent.create(
            batch_dim=basic_dynamics.batch_dim,
            name="test_agent",
            dim_c=0,
            dim_p=2,
            action_size=4,
        )

        # Set some non-zero position and rotation
        pos = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        rot = jnp.array([[0.5], [0.6]])
        agent = agent.replace(
            state=agent.state.replace(pos=pos, rot=rot),
            action=agent.action.replace(u=jnp.zeros((2, 4))),
        )

        dynamics, _ = basic_dynamics.check_and_process_action(agent)

        # Check that drone state matches agent state
        assert jnp.allclose(dynamics.drone_state[:, 9:11], pos)  # x, y position
        assert jnp.allclose(dynamics.drone_state[:, 2], rot.squeeze())  # yaw angle
