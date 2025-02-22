import jax.numpy as jnp
import pytest

from jaxvmas.simulator.core.action import Action


class TestAction:
    @pytest.fixture
    def basic_action(self):
        # Create basic action with float inputs
        action = Action.create(
            action_size=3,
            u_range=1.0,
            u_multiplier=1.0,
            u_noise=0.1,
        )
        action = action._spawn(2, 2)
        return action

    def test_create_with_float_inputs(self, basic_action: Action):
        # Test creation with float inputs
        assert basic_action.u.shape == (2, 3)
        assert basic_action.c.shape == (2, 2)
        assert basic_action.u_range == 1.0
        assert basic_action.u_multiplier == 1.0
        assert basic_action.u_noise == 0.1

    def test_create_with_sequence_inputs(self):
        # Test creation with sequence inputs
        action = Action.create(
            action_size=3,
            u_range=[1.0, 2.0, 3.0],
            u_multiplier=[0.5, 1.0, 1.5],
            u_noise=[0.1, 0.2, 0.3],
        )
        action = action._spawn(2, 2)
        assert jnp.array_equal(action.u_range_jax_array, jnp.asarray([1.0, 2.0, 3.0]))
        assert jnp.array_equal(
            action.u_multiplier_jax_array, jnp.asarray([0.5, 1.0, 1.5])
        )
        assert jnp.array_equal(action.u_noise_jax_array, jnp.asarray([0.1, 0.2, 0.3]))

    def test_jax_array_properties(self, basic_action):
        # Test jax array property conversion
        assert jnp.array_equal(
            basic_action.u_range_jax_array, jnp.array([1.0, 1.0, 1.0])
        )
        assert jnp.array_equal(
            basic_action.u_multiplier_jax_array, jnp.array([1.0, 1.0, 1.0])
        )
        assert jnp.array_equal(
            basic_action.u_noise_jax_array, jnp.array([0.1, 0.1, 0.1])
        )

    def test_reset_without_env_index(self, basic_action: Action):
        # Set non-zero values
        action = basic_action.replace(u=jnp.ones((2, 3)), c=jnp.ones((2, 2)))
        # Test reset
        reset_action = action._reset(jnp.nan)
        assert jnp.all(reset_action.u == 0)
        assert jnp.all(reset_action.c == 0)

    def test_reset_with_env_index(self, basic_action: Action):
        # Set non-zero values
        action = basic_action.replace(u=jnp.ones((2, 3)), c=jnp.ones((2, 2)))
        # Test reset for specific environment
        reset_action = action._reset(0)
        assert jnp.all(reset_action.u[0] == 0)
        assert jnp.all(reset_action.u[1] == 1)
        assert jnp.all(reset_action.c[0] == 0)
        assert jnp.all(reset_action.c[1] == 1)

    def test_invalid_sequence_length(self):
        # Test validation of sequence length
        with pytest.raises(AssertionError):
            Action.create(
                action_size=3,
                u_range=[1.0, 2.0],  # Wrong length
                u_multiplier=1.0,
                u_noise=0.1,
            )
