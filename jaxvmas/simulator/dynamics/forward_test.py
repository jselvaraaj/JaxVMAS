import equinox as eqx
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from jaxvmas.simulator.core import Agent
from jaxvmas.simulator.dynamics.forward import Forward
from jaxvmas.simulator.utils import JaxUtils, X


class TestForwardDynamics:
    @pytest.fixture
    def basic_dynamics(self):
        return Forward()

    @pytest.fixture
    def basic_agent(self):
        return Agent.create(
            batch_dim=2,
            name="test_agent",
            dim_c=0,
            dim_p=2,
            action_size=2,  # Larger than needed_action_size for testing
        )

    def test_create(self, basic_dynamics: Forward):
        assert isinstance(basic_dynamics, Forward)
        assert basic_dynamics.needed_action_size == 1

    def test_reset(self, basic_dynamics: Forward):
        # Test reset without index
        reset_dynamics = basic_dynamics.reset()
        assert isinstance(reset_dynamics, Forward)

        # Test reset with index
        reset_dynamics_index = basic_dynamics.reset(index=0)
        assert isinstance(reset_dynamics_index, Forward)

        # Test reset with array index
        reset_dynamics_array = basic_dynamics.reset(index=jnp.array([0, 1]))
        assert isinstance(reset_dynamics_array, Forward)

    def test_check_and_process_action_valid(
        self, basic_dynamics: Forward, basic_agent: Agent
    ):
        # Test valid action processing
        dynamics, agent = basic_dynamics.check_and_process_action(basic_agent)

        assert isinstance(dynamics, Forward)
        assert isinstance(agent, Agent)

        # Check force is correctly rotated based on agent rotation
        expected_force = jnp.zeros((agent.batch_dim, 2))
        expected_force = expected_force.at[:, X].set(basic_agent.action.u[:, 0])
        expected_force = JaxUtils.rotate_vector(expected_force, basic_agent.state.rot)
        assert jnp.allclose(agent.state.force, expected_force)

    def test_check_and_process_action_invalid(self, basic_dynamics: Forward):
        # Create agent with insufficient action size
        agent = Agent.create(
            batch_dim=2,
            name="test_agent",
            dim_c=0,
            dim_p=2,
            action_size=0,  # Less than needed_action_size
        )

        # Test that it raises ValueError for insufficient action size
        with pytest.raises(ValueError):
            basic_dynamics.check_and_process_action(agent)

    def test_process_action_shapes(self, basic_dynamics: Forward, basic_agent: Agent):
        # Test output shapes from process_action
        dynamics, agent = basic_dynamics.check_and_process_action(basic_agent)

        assert agent.state.force.shape == (2, 2)  # batch_dim x dim_p

    def test_is_jittable(self, basic_dynamics: Forward, basic_agent: Agent):
        # Test jit compatibility of reset
        @eqx.filter_jit
        def reset_dynamics(dynamics: Forward):
            return dynamics.reset()

        reset_result = reset_dynamics(basic_dynamics)
        assert isinstance(reset_result, Forward)

        # Test jit compatibility of check_and_process_action
        @eqx.filter_jit
        def process_action(dynamics: Forward, agent: Agent):
            return dynamics.check_and_process_action(agent)

        dynamics_result, agent_result = process_action(basic_dynamics, basic_agent)
        assert isinstance(dynamics_result, Forward)
        assert isinstance(agent_result, Agent)

        # Test jit compatibility with array index
        @eqx.filter_jit
        def reset_with_index(dynamics: Forward, index: Array):
            return dynamics.reset(index=index)

        reset_index_result = reset_with_index(basic_dynamics, jnp.array([0]))
        assert isinstance(reset_index_result, Forward)

    def test_batch_processing(self, basic_dynamics: Forward):
        # Test processing with different batch sizes
        for batch_dim in [1, 2, 4]:
            agent = Agent.create(
                batch_dim=batch_dim,
                name="test_agent",
                dim_c=0,
                dim_p=2,
                action_size=2,
            )

            dynamics, processed_agent = basic_dynamics.check_and_process_action(agent)
            assert processed_agent.state.force.shape == (batch_dim, 2)

    def test_zero_action(self, basic_dynamics: Forward):
        # Test processing with zero actions
        agent = Agent.create(
            batch_dim=2,
            name="test_agent",
            dim_c=0,
            dim_p=2,
            action_size=2,
        )
        agent = agent.replace(action=agent.action.replace(u=jnp.zeros((2, 2))))

        dynamics, processed_agent = basic_dynamics.check_and_process_action(agent)
        assert jnp.all(processed_agent.state.force == 0)

    def test_rotation_effect(self, basic_dynamics: Forward):
        # Test that force is correctly rotated based on agent rotation
        agent = Agent.create(
            batch_dim=1,
            name="test_agent",
            dim_c=0,
            dim_p=2,
            action_size=2,
        )

        # Test with different rotations
        rotations = [0.0, jnp.pi / 2, jnp.pi, 3 * jnp.pi / 2]
        action = jnp.array([[1.0, 0.0]])  # Forward action

        for rot in rotations:
            agent = agent.replace(
                action=agent.action.replace(u=action),
                state=agent.state.replace(rot=jnp.array([rot])),
            )

            dynamics, processed_agent = basic_dynamics.check_and_process_action(agent)

            # Expected force should be rotated by the agent's rotation
            expected_force = jnp.array([[jnp.cos(rot), jnp.sin(rot)]])
            assert jnp.allclose(processed_agent.state.force, expected_force)
