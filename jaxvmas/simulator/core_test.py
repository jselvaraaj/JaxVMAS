import equinox as eqx
import jax.numpy as jnp
import pytest

from jaxvmas.simulator.core import EntityState


@pytest.fixture
def entity_state():
    # Initialize EntityState with batch_dim=2, dim_c=3, dim_p=4
    return EntityState.create(batch_dim=2, dim_c=3, dim_p=4)


def test_entity_state_reset(entity_state: EntityState):
    # Test reset without env_index
    reset_state = entity_state._reset()

    assert jnp.array_equal(reset_state.pos, jnp.zeros((2, 4)))
    assert jnp.array_equal(reset_state.vel, jnp.zeros((2, 4)))
    assert jnp.array_equal(reset_state.rot, jnp.zeros((2, 1)))
    assert jnp.array_equal(reset_state.ang_vel, jnp.zeros((2, 1)))

    # Test reset with env_index
    reset_state_index = entity_state._reset(env_index=0)
    assert jnp.array_equal(reset_state_index.pos, jnp.zeros((2, 4)))
    assert jnp.array_equal(reset_state_index.vel, jnp.zeros((2, 4)))
    assert jnp.array_equal(reset_state_index.rot, jnp.zeros((2, 1)))
    assert jnp.array_equal(reset_state_index.ang_vel, jnp.zeros((2, 1)))


def test_entity_state_spawn(entity_state: EntityState):
    # Test spawn
    spawned_state = entity_state._spawn(dim_c=3, dim_p=4)
    assert jnp.array_equal(spawned_state.pos, jnp.zeros((2, 4)))
    assert jnp.array_equal(spawned_state.vel, jnp.zeros((2, 4)))
    assert jnp.array_equal(spawned_state.rot, jnp.zeros((2, 1)))
    assert jnp.array_equal(spawned_state.ang_vel, jnp.zeros((2, 1)))


def test_entity_state_is_jittable(entity_state: EntityState):
    @eqx.filter_jit
    def f(e_s):
        return e_s._spawn(dim_c=3, dim_p=4)

    f(entity_state)

    @eqx.filter_jit
    def f2(e_s):
        return e_s._reset()

    f2(entity_state)
