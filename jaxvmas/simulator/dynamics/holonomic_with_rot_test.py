import equinox as eqx
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from jaxvmas.simulator.core.agent import Agent
from jaxvmas.simulator.dynamics.holonomic_with_rot import HolonomicWithRotation


class TestHolonomicWithRotationDynamics:
    @pytest.fixture
    def basic_dynamics(self):
        return HolonomicWithRotation()

    @pytest.fixture
    def basic_agent(self):
        return Agent.create(
            batch_dim=2,
            name="test_agent",
            dim_c=0,
            dim_p=2,
            action_size=4,  # Larger than needed_action_size for testing
        )

    def test_create(self, basic_dynamics: HolonomicWithRotation):
        assert isinstance(basic_dynamics, HolonomicWithRotation)
        assert basic_dynamics.needed_action_size == 3

    def test_reset(self, basic_dynamics: HolonomicWithRotation):
        # Test reset without index
        reset_dynamics = basic_dynamics.reset()
        assert isinstance(reset_dynamics, HolonomicWithRotation)

        # Test reset with index
        reset_dynamics_index = basic_dynamics.reset(index=0)
        assert isinstance(reset_dynamics_index, HolonomicWithRotation)

        # Test reset with array index
        reset_dynamics_array = basic_dynamics.reset(index=jnp.array([0, 1]))
        assert isinstance(reset_dynamics_array, HolonomicWithRotation)

    def test_check_and_process_action_valid(
        self, basic_dynamics: HolonomicWithRotation, basic_agent: Agent
    ):
        # Test valid action processing
        dynamics, agent = basic_dynamics.check_and_process_action(basic_agent)

        assert isinstance(dynamics, HolonomicWithRotation)
        assert isinstance(agent, Agent)
        assert jnp.array_equal(agent.state.force, basic_agent.action.u[:, :2])
        assert jnp.array_equal(
            agent.state.torque, basic_agent.action.u[:, 2][..., None]
        )

    def test_check_and_process_action_invalid(
        self, basic_dynamics: HolonomicWithRotation
    ):
        # Create agent with insufficient action size
        agent = Agent.create(
            batch_dim=2,
            name="test_agent",
            dim_c=0,
            dim_p=2,
            action_size=2,  # Less than needed_action_size
        )

        # Test that it raises ValueError for insufficient action size
        with pytest.raises(ValueError):
            basic_dynamics.check_and_process_action(agent)

    def test_process_action_shapes(
        self, basic_dynamics: HolonomicWithRotation, basic_agent: Agent
    ):
        # Test output shapes from process_action
        dynamics, agent = basic_dynamics.check_and_process_action(basic_agent)

        assert agent.state.force.shape == (2, 2)  # batch_dim x dim_p
        assert agent.state.torque.shape == (2, 1)  # batch_dim x 1

    def test_is_jittable(
        self, basic_dynamics: HolonomicWithRotation, basic_agent: Agent
    ):
        # Test jit compatibility of reset
        @eqx.filter_jit
        def reset_dynamics(dynamics: HolonomicWithRotation):
            return dynamics.reset()

        reset_result = reset_dynamics(basic_dynamics)
        assert isinstance(reset_result, HolonomicWithRotation)

        # Test jit compatibility of check_and_process_action
        @eqx.filter_jit
        def process_action(dynamics: HolonomicWithRotation, agent: Agent):
            return dynamics.check_and_process_action(agent)

        dynamics_result, agent_result = process_action(basic_dynamics, basic_agent)
        assert isinstance(dynamics_result, HolonomicWithRotation)
        assert isinstance(agent_result, Agent)

        # Test jit compatibility with array index
        @eqx.filter_jit
        def reset_with_index(dynamics: HolonomicWithRotation, index: Array):
            return dynamics.reset(index=index)

        reset_index_result = reset_with_index(basic_dynamics, jnp.array([0]))
        assert isinstance(reset_index_result, HolonomicWithRotation)

    def test_batch_processing(self, basic_dynamics: HolonomicWithRotation):
        # Test processing with different batch sizes
        for batch_dim in [1, 2, 4]:
            agent = Agent.create(
                batch_dim=batch_dim,
                name="test_agent",
                dim_c=0,
                dim_p=2,
                action_size=4,
            )

            dynamics, processed_agent = basic_dynamics.check_and_process_action(agent)
            assert processed_agent.state.force.shape == (batch_dim, 2)
            assert processed_agent.state.torque.shape == (batch_dim, 1)

    def test_zero_action(self, basic_dynamics: HolonomicWithRotation):
        # Test processing with zero actions
        agent = Agent.create(
            batch_dim=2,
            name="test_agent",
            dim_c=0,
            dim_p=2,
            action_size=4,
        )
        agent = agent.replace(action=agent.action.replace(u=jnp.zeros((2, 4))))

        dynamics, processed_agent = basic_dynamics.check_and_process_action(agent)
        assert jnp.all(processed_agent.state.force == 0)
        assert jnp.all(processed_agent.state.torque == 0)
