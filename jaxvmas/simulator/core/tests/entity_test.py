import equinox as eqx
import jax.numpy as jnp
import pytest

from jaxvmas.simulator.core.entity import Entity


class TestEntity:
    @pytest.fixture
    def basic_entity(self):
        return Entity.create(
            batch_dim=2,
            name="test_entity",
            movable=True,
            rotatable=True,
        )

    def test_create(self, basic_entity):
        # Test basic properties
        assert basic_entity.name == "test_entity"
        assert basic_entity.movable is True
        assert basic_entity.rotatable is True
        assert basic_entity.collide is True
        assert basic_entity.mass == 1.0
        assert basic_entity.density == 25.0
        assert jnp.array_equal(basic_entity.gravity, jnp.zeros((2, 1)))

        # Test creation with custom parameters
        custom_entity = Entity.create(
            batch_dim=2,
            name="custom",
            movable=False,
            gravity=jnp.asarray([0.0, -9.81])[..., None],
            mass=2.0,
            v_range=5.0,
            max_speed=10.0,
        )
        assert custom_entity.mass == 2.0
        assert custom_entity.v_range == 5.0
        assert custom_entity.max_speed == 10.0
        assert jnp.array_equal(
            custom_entity.gravity, jnp.array([0.0, -9.81])[..., None]
        )

    def test_property_setters(self, basic_entity):
        # Test setting position
        new_pos = jnp.array([1.0, 2.0])
        updated_entity = basic_entity.set_pos(new_pos, batch_index=0)
        assert jnp.array_equal(updated_entity.state.pos[0], new_pos)

        # Test setting velocity
        new_vel = jnp.array([0.5, 0.5])
        updated_entity = basic_entity.set_vel(new_vel, batch_index=0)
        assert jnp.array_equal(updated_entity.state.vel[0], new_vel)

        # Test setting rotation
        new_rot = jnp.array([1.5])
        updated_entity = basic_entity.set_rot(new_rot, batch_index=0)
        assert jnp.array_equal(updated_entity.state.rot[0], new_rot)

    def test_collision_filter(self, basic_entity):
        # Test default collision filter
        other_entity = Entity.create(batch_dim=2, name="other")
        assert jnp.all(basic_entity.collides(other_entity))

        # Test custom collision filter
        custom_filter_entity = Entity.create(
            batch_dim=2, name="filtered", collision_filter=lambda e: e.name == "target"
        )
        assert not jnp.all(custom_filter_entity.collides(other_entity))
        target_entity = Entity.create(batch_dim=2, name="target")
        assert jnp.all(custom_filter_entity.collides(target_entity))

    def test_render_flags(self, basic_entity):
        # Test initial render state
        assert jnp.all(basic_entity._render)

        basic_entity = basic_entity.replace(_render=jnp.zeros((2,), dtype=jnp.bool))

        # Test reset render
        basic_entity = basic_entity.reset_render()
        assert jnp.all(basic_entity._render)

    def test_spawn_and_reset(self, basic_entity):
        # Test spawn
        # First set non-zero values
        basic_entity = basic_entity.set_pos(jnp.array([1.0, 2.0]))
        basic_entity = basic_entity.set_vel(jnp.array([3.0, 4.0]))
        basic_entity = basic_entity.set_rot(jnp.array([0.5]))

        spawned_entity = basic_entity._spawn(dim_p=2)
        # Check shapes are preserved
        assert spawned_entity.state.pos.shape == (2, 2)
        assert spawned_entity.state.vel.shape == (2, 2)
        assert spawned_entity.state.rot.shape == (2, 1)
        # Check values are preserved
        assert jnp.allclose(
            spawned_entity.state.pos, jnp.zeros_like(spawned_entity.state.pos)
        )
        assert jnp.allclose(
            spawned_entity.state.vel, jnp.zeros_like(spawned_entity.state.vel)
        )
        assert jnp.allclose(
            spawned_entity.state.rot, jnp.zeros_like(spawned_entity.state.rot)
        )

        # Test reset
        reset_entity = basic_entity._reset(env_index=0)
        # Verify values were actually changed from non-zero to zero
        assert jnp.allclose(reset_entity.state.pos[0], jnp.zeros(2))
        assert jnp.allclose(reset_entity.state.pos[1], jnp.array([1.0, 2.0]))

        assert jnp.allclose(reset_entity.state.vel[0], jnp.zeros(2))
        assert jnp.allclose(reset_entity.state.vel[1], jnp.array([3.0, 4.0]))

        assert jnp.allclose(reset_entity.state.rot[0], jnp.zeros(1))
        assert jnp.allclose(reset_entity.state.rot[1], jnp.array([0.5]))

    def test_is_jittable(self, basic_entity):
        @eqx.filter_jit
        def f(entity, pos):
            return entity.set_pos(pos, batch_index=0)

        test_pos = jnp.array([1.0, 2.0])
        f(basic_entity, test_pos)
