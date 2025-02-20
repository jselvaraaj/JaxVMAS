import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from jaxvmas.simulator.core.agent import Agent
from jaxvmas.simulator.core.entity import Entity
from jaxvmas.simulator.core.landmark import Landmark
from jaxvmas.simulator.core.shapes import Box, Line, Shape, Sphere
from jaxvmas.simulator.core.world import (
    DRAG,
    World,
)
from jaxvmas.simulator.utils import LINE_MIN_DIST


# Test that an unsupported shape triggers a RuntimeError.
class DummyShape(Shape):
    def moment_of_inertia(self, mass):
        pass

    def get_delta_from_anchor(self, anchor):
        pass

    def get_geometry(self):
        pass

    def circumscribed_radius(self):
        pass


class TestWorld:
    @pytest.fixture
    def basic_world(self):
        # Create a basic world with minimal configuration
        return World.create(batch_dim=2)

    @pytest.fixture
    def agent(self):
        return Agent.create(
            batch_dim=2,
            name="test_agent",
            dim_c=3,
            dim_p=2,
            movable=True,
            rotatable=True,
        )

    @pytest.fixture
    def landmark(self):
        return Landmark.create(batch_dim=2, name="test_landmark")

    @pytest.fixture
    def world_with_agent(self, basic_world: World, agent: Agent):
        return basic_world.add_agent(agent)

    def test_create(self, basic_world: World):
        # Test basic properties
        assert basic_world.batch_dim == 2
        assert basic_world.dt == 0.1
        assert basic_world.substeps == 1
        assert basic_world.drag == DRAG
        assert len(basic_world.agents) == 0
        assert len(basic_world.landmarks) == 0
        assert isinstance(basic_world._joints, dict)

    def test_add_agent(self, basic_world: World, agent: Agent):
        world = basic_world.add_agent(agent)
        assert len(world.agents) == 1
        assert world.agents[0].name == "test_agent"
        assert world.agents[0].batch_dim == 2

    def test_add_landmark(self, basic_world: World, landmark: Landmark):
        world = basic_world.add_landmark(landmark)
        assert len(world.landmarks) == 1
        assert world.landmarks[0].name == "test_landmark"
        assert world.landmarks[0].batch_dim == 2

    def test_reset(self, world_with_agent: World):
        # Modify agent state
        agent = world_with_agent.agents[0]
        agent = agent.replace(state=agent.state.replace(pos=jnp.ones((2, 2))))
        world = world_with_agent.replace(agents=[agent])

        # Test reset
        world = world.reset(env_index=0)
        assert jnp.all(world.agents[0].state.pos[0] == 0)
        assert jnp.all(world.agents[0].state.pos[1] == 1)

    def test_step(self, world_with_agent: World):
        # Test basic stepping without forces
        world = world_with_agent.step()
        assert isinstance(world, World)

        # Test stepping with forces
        agent = world.agents[0]
        agent = agent.replace(state=agent.state.replace(force=jnp.ones((2, 2))))
        world = world.replace(agents=[agent])
        world = world.step()
        assert isinstance(world, World)
        # Velocity should have changed due to force
        assert not jnp.all(world.agents[0].state.vel == 0)

    def test_collisions(self, basic_world: World):
        # Create two overlapping spherical agents
        # Create two overlapping spherical agents
        agent1 = Agent.create(
            batch_dim=2,
            name="agent1",
            dim_c=0,
            dim_p=2,
            movable=True,
        )
        agent2 = Agent.create(
            batch_dim=2,
            name="agent2",
            dim_c=0,
            dim_p=2,
            movable=True,
        )

        world = basic_world.add_agent(agent1).add_agent(agent2)

        # Test collision detection
        assert world.collides(agent1, agent2)

        # Test no collision with non-collidable entity
        landmark = Landmark.create(batch_dim=2, name="landmark", collide=False)
        world = world.add_landmark(landmark)
        assert not world.collides(agent1, landmark)

    def test_communication(self, basic_world: World):
        # Create world with communication
        world = World.create(batch_dim=2, dim_c=3)
        agent = Agent.create(batch_dim=2, name="agent", dim_c=3, dim_p=2, silent=False)
        world = world.add_agent(agent)

        # Set communication action
        agent = world.agents[0]
        agent = agent.replace(action=agent.action.replace(c=jnp.ones((2, 3))))
        world = world.replace(agents=[agent])

        # Step world and check communication state
        world = world.step()
        assert jnp.all(world.agents[0].state.c == 1)

    def test_joints(self, basic_world: World):
        from jaxvmas.simulator.joints import Joint, JointConstraint

        # Create world with higher substeps for joint stability
        world = World.create(batch_dim=2, substeps=5)

        # Create two agents to be joined
        agent1 = Agent.create(
            batch_dim=2,
            name="agent1",
            dim_c=0,
            dim_p=2,
            movable=True,
        )
        agent2 = Agent.create(
            batch_dim=2,
            name="agent2",
            dim_c=0,
            dim_p=2,
            movable=True,
        )

        world = world.add_agent(agent1).add_agent(agent2)

        # Create and add joint
        joint = Joint.create(
            batch_dim=2,
            entity_a=agent1,
            entity_b=agent2,
            anchor_a=(0.0, 0.0),
            anchor_b=(0.0, 0.0),
            dist=1.0,
        )
        world = world.add_joint(joint)

        assert len(world._joints) == 2
        assert isinstance(list(world._joints.values())[0], JointConstraint)
        assert isinstance(list(world._joints.values())[1], JointConstraint)

    def test_entity_index_map(self, world_with_agent: World):
        # Test entity index map is properly updated
        world = world_with_agent.reset()
        assert len(world.entity_index_map) == 1
        assert world.agents[0].name in world.entity_index_map
        assert world.entity_index_map[world.agents[0].name] == 0

    def test_boundary_conditions(self, basic_world: World):
        # Test world with boundaries
        world = World.create(batch_dim=2, x_semidim=1.0, y_semidim=1.0)
        agent = Agent.create(
            batch_dim=2,
            name="agent",
            dim_c=0,
            dim_p=2,
            movable=True,
        )
        world = world.add_agent(agent)

        # Move agent beyond boundary
        agent = world.agents[0]
        agent = agent.replace(
            state=agent.state.replace(pos=jnp.array([[2.0, 0.0], [0.0, 2.0]]))
        )
        world = world.replace(agents=[agent])

        # Step world and check if position is constrained
        with jax.disable_jit():
            world = world.step()
        assert jnp.all(jnp.abs(world.agents[0].state.pos) <= 1.0)

    def test_collision_response(self):
        # Create two colliding agents with no initial forces
        agent1 = Agent.create(
            batch_dim=1,
            name="collider1",
            dim_p=2,
            dim_c=0,
            mass=1.0,
        )
        agent1 = agent1.replace(
            state=agent1.state.replace(
                pos=jnp.array([[0.0, 0.0]]),
            ),
            shape=Sphere(radius=0.1),
        )

        agent2 = Agent.create(
            batch_dim=1,
            name="collider2",
            dim_p=2,
            dim_c=0,
            mass=1.0,
        )
        agent2 = agent2.replace(
            state=agent2.state.replace(
                pos=jnp.array([[0.15, 0.0]]),
            ),
            shape=Sphere(radius=0.1),
        )

        world = World.create(batch_dim=1, substeps=10)
        world = world.replace(agents=[agent1, agent2])
        initial_dist = jnp.linalg.norm(
            world.agents[0].state.pos - world.agents[1].state.pos
        )

        # Step and verify collision response
        world = world.step()
        # Calculate final distance
        final_dist = jnp.linalg.norm(
            world.agents[0].state.pos - world.agents[1].state.pos
        )
        assert (
            final_dist > initial_dist
        ), f"Agents moved closer together (from {initial_dist} to {final_dist})"

    def test_joint_constraint_satisfaction(self):
        # Create joined agents
        agent1 = Agent.create(batch_dim=1, name="joint1", dim_p=2, dim_c=0)
        agent2 = Agent.create(batch_dim=1, name="joint2", dim_p=2, dim_c=0)
        world = World.create(
            batch_dim=1,
            dt=0.1,
            substeps=10,  # Higher substeps for joint stability
            joint_force=500.0,  # Increase joint force for stronger constraint
            linear_friction=0.0,  # Remove friction to allow easier movement
            drag=0.1,  # Reduce drag for smoother motion
        )
        world = world.add_agent(agent1).add_agent(agent2)
        world = world.reset()
        world = world.replace(
            agents=[
                agent1.replace(
                    state=agent1.state.replace(pos=jnp.array([[-0.5, 0.0]]))
                ),
                agent2.replace(state=agent2.state.replace(pos=jnp.array([[0.5, 0.0]]))),
            ]
        )

        # Add joint constraint
        from jaxvmas.simulator.joints import JointConstraint

        # Create a direct joint constraint between agents
        constraint = JointConstraint.create(
            entity_a=agent1,
            entity_b=agent2,
            anchor_a=(0.0, 0.0),
            anchor_b=(0.0, 0.0),
            dist=0.5,  # Target distance
        )
        world = world.replace(
            _joints={frozenset({agent1.name, agent2.name}): constraint}
        )

        # Step and verify joint constraint
        stepped_world = world
        for _ in range(100):
            stepped_world = stepped_world.step()

        # Distance between agents should be close to joint distance
        final_dist = jnp.linalg.norm(
            stepped_world.agents[0].state.pos - stepped_world.agents[1].state.pos
        )
        print(f"Final distance: {final_dist}, Target distance: 0.5")

        # Check if final distance is reasonably close to desired distance
        assert jnp.abs(final_dist - 0.5) < 0.1, (
            f"Joint constraint not satisfied. Expected distance close to 0.5, "
            f"but got {final_dist:.3f}"
        )

    def test_extreme_boundary_conditions(self):
        # Test negative boundaries
        world = World.create(batch_dim=1, x_semidim=-1.0, y_semidim=-0.5)
        agent = Agent.create(batch_dim=1, name="boundary_test", dim_p=2, dim_c=0)
        agent = agent.replace(state=agent.state.replace(pos=jnp.array([[2.0, 1.0]])))
        world = world.add_agent(agent)

        # Step and verify boundary handling
        stepped_world = world.step()

        # Agent should be constrained by boundaries
        final_pos = stepped_world.agents[0].state.pos[0]
        assert jnp.all(jnp.abs(final_pos) <= jnp.array([1.0, 0.5]))

    def test_is_jittable(self, basic_world: World, agent: Agent, landmark: Landmark):
        # Test jit compatibility of world creation and entity addition
        @eqx.filter_jit
        def create_world_with_entities(world: World, agent: Agent, landmark: Landmark):
            world = world.add_agent(agent)
            world = world.add_landmark(landmark)
            world = world.reset()
            return world

        world = create_world_with_entities(basic_world, agent, landmark)
        assert len(world.agents) == 1
        assert len(world.landmarks) == 1

        # Test jit compatibility of stepping with forces
        @eqx.filter_jit
        def step_with_force(world: World):
            agent = world.agents[0]
            agent = agent.replace(state=agent.state.replace(force=jnp.ones((2, 2))))
            world = world.replace(agents=[agent])
            return world.step()

        stepped_world = step_with_force(world)
        assert not jnp.all(stepped_world.agents[0].state.vel == 0)

        # Test jit compatibility of reset with specific index
        @eqx.filter_jit
        def reset_env(world: World, env_idx: int):
            return world.reset(env_index=env_idx)

        reset_world = reset_env(world, 0)
        assert jnp.all(reset_world.agents[0].state.pos[0] == 0)

        # Test jit compatibility of collision detection
        @eqx.filter_jit
        def check_collision(world: World, entity1: Entity, entity2: Entity):
            return world.collides(entity1, entity2)

        collides = check_collision(world, world.agents[0], world.landmarks[0])
        assert isinstance(collides, Array)

        # Test jit compatibility with joints
        from jaxvmas.simulator.joints import Joint

        @eqx.filter_jit
        def add_and_step_with_joint(world: World):
            agent2 = Agent.create(
                batch_dim=2,
                name="agent2",
                dim_c=0,
                dim_p=2,
                movable=True,
            )
            world = world.add_agent(agent2)
            world = world.replace(substeps=2)
            joint = Joint.create(
                batch_dim=2,
                entity_a=world.agents[0],
                entity_b=world.agents[1],
                anchor_a=(0.0, 0.0),
                anchor_b=(0.0, 0.0),
                dist=1.0,
            )
            world = world.add_joint(joint)

            world = world.reset()

            return world.step()

        joint_world = add_and_step_with_joint(world)
        assert len(joint_world._joints) == 2

        # Test jit compatibility with boundary conditions
        @eqx.filter_jit
        def step_with_boundaries(world: World, pos: Array):
            agent = world.agents[0]
            agent = agent.replace(state=agent.state.replace(pos=pos))
            world = world.replace(agents=[agent])
            world = world.replace(x_semidim=1.0, y_semidim=1.0)
            return world.step()

        boundary_world = step_with_boundaries(
            world, jnp.array([[2.0, 2.0], [0.0, 0.0]])
        )
        assert jnp.all(jnp.abs(boundary_world.agents[0].state.pos[0]) <= 1.0)

        # Test jit compatibility with multiple substeps
        @eqx.filter_jit
        def step_with_substeps(world: World, substeps: int):
            return world.replace(substeps=substeps).step()

        multi_step_world = step_with_substeps(world, 5)
        assert isinstance(multi_step_world, World)


class TestRayCasting:

    def test_cast_ray_sphere(self):
        # Create world with a sphere agent
        with jax.disable_jit():
            world = World.create(batch_dim=1)

            # Create source agent (ray origin)
            source_agent = Agent.create(
                batch_dim=1,
                name="source",
                dim_p=2,
                dim_c=0,
            )
            # Create target sphere
            sphere_agent = Agent.create(
                batch_dim=1,
                name="sphere",
                dim_p=2,
                dim_c=0,
                shape=Sphere(radius=0.5),
            )
            world = world.add_agent(source_agent).add_agent(sphere_agent)

            source_agent = source_agent.replace(
                state=source_agent.state.replace(pos=jnp.array([[0.0, 0.0]]))
            )
            sphere_agent = sphere_agent.replace(
                state=sphere_agent.state.replace(pos=jnp.array([[1.0, 0.0]]))
            )
            world = world.replace(agents=[source_agent, sphere_agent])

            # Test direct hit
            angles = jnp.array([0.0])  # Ray pointing right
            dists = world.cast_ray(
                source_agent,
                angles,
                max_range=5.0,
                entity_filter=lambda e: e.name == "sphere",  # Only detect sphere
            )
            assert jnp.allclose(dists[0], 0.5)  # Should hit at radius distance

            # Test miss
            angles = jnp.array([jnp.pi])  # Ray pointing left
            dists = world.cast_ray(
                source_agent,
                angles,
                max_range=5.0,
                entity_filter=lambda e: e.name == "sphere",
            )
            assert jnp.allclose(dists[0], 5.0)  # Should return max_range

    def test_cast_ray_box(self):
        # Create world with a box agent
        with jax.disable_jit():
            world = World.create(batch_dim=1)

            # Create source agent (ray origin)
            source_agent = Agent.create(
                batch_dim=1,
                name="source",
                dim_p=2,
                dim_c=0,
            )
            # Create target box
            box_agent = Agent.create(
                batch_dim=1,
                name="box",
                dim_p=2,
                dim_c=0,
                shape=Box(length=1.0, width=1.0),
            )
            world = world.add_agent(source_agent).add_agent(box_agent)

            source_agent = source_agent.replace(
                state=source_agent.state.replace(pos=jnp.array([[0.0, 0.0]]))
            )

            # Position box closer and at 45 degrees for better diagonal test
            box_agent = box_agent.replace(
                state=box_agent.state.replace(
                    pos=jnp.array([[1.0, 1.0]]),  # Move box to (1,1) for 45-degree test
                    rot=jnp.array([[0.0]]),
                )
            )
            world = world.replace(agents=[source_agent, box_agent])
            # Test direct hit
            angles = jnp.array([jnp.pi / 4])  # 45 degrees to hit box directly
            dists = world.cast_ray(
                source_agent,
                angles,
                max_range=5.0,
                entity_filter=lambda e: e.name == "box",  # Only detect box
            )

            # Calculate expected distance
            # At 45 degrees, distance to corner = sqrt(2) * distance to center - half_diagonal
            center_dist = jnp.sqrt(2.0)  # sqrt((1-0)^2 + (1-0)^2)
            half_diagonal = jnp.sqrt(2.0) * 0.5  # sqrt((0.5)^2 + (0.5)^2)
            expected_dist = center_dist - half_diagonal
            assert jnp.allclose(
                dists[0], expected_dist, atol=1e-3
            ), f"Expected distance {expected_dist}, got {dists[0]}"

            # Test parallel to box edge
            angles = jnp.array([0.0])  # Parallel to box edge
            dists = world.cast_ray(
                source_agent,
                angles,
                max_range=5.0,
                entity_filter=lambda e: e.name == "box",
            )
            assert jnp.allclose(
                dists[0], 5.0
            ), f"Expected miss (distance 5.0), got {dists[0]}"

    def test_cast_rays_multiple_objects(self):
        # Create world with multiple objects
        with jax.disable_jit():
            world = World.create(batch_dim=1)

            # Create source agent (ray origin)
            source_agent = Agent.create(
                batch_dim=1,
                name="source",
                dim_p=2,
                dim_c=0,
            )
            # Add sphere
            sphere = Agent.create(
                batch_dim=1,
                name="sphere",
                dim_p=2,
                dim_c=0,
                shape=Sphere(radius=0.5),
            )
            # Add box
            box = Agent.create(
                batch_dim=1,
                name="box",
                dim_p=2,
                dim_c=0,
                shape=Box(length=1.0, width=1.0),
            )

            world = world.add_agent(source_agent).add_agent(sphere).add_agent(box)

            source_agent = source_agent.replace(
                state=source_agent.state.replace(pos=jnp.array([[0.0, 0.0]]))
            )
            sphere = sphere.replace(
                state=sphere.state.replace(pos=jnp.array([[0.0, 1.0]]))
            )
            box = box.replace(
                state=box.state.replace(
                    pos=jnp.array(
                        [[0.0, -1.0]]
                    ),  # Changed from [1.0, -1.0] to [0.0, -1.0]
                    rot=jnp.array([[0.0]]),
                )
            )
            world = world.replace(agents=[source_agent, sphere, box])

            # Cast rays in multiple directions
            angles = jnp.expand_dims(
                jnp.array([0.0, jnp.pi / 2, jnp.pi, 3 * jnp.pi / 2]), axis=0
            )  # Shape becomes (1, 4)
            dists = world.cast_rays(
                source_agent, angles, max_range=5.0, entity_filter=lambda e: True
            )
            # Verify distances
            assert dists.shape == (1, 4)  # Should have batch_dim=1 and 4 angles
            assert jnp.all(dists <= 5.0)  # All distances should be within max_range
            assert jnp.any(dists < 5.0)  # At least one ray should hit something

            # Verify specific hits
            # Ray at 0 degrees should miss both objects
            assert jnp.allclose(
                dists[0, 0], 5.0
            ), f"Expected miss at 0 degrees, got {dists[0, 0]}"

            # Ray at 90 degrees (π/2) should hit sphere
            expected_dist_sphere = (
                jnp.sqrt(1.0) - 0.5
            )  # Distance to center minus radius
            assert jnp.allclose(
                dists[0, 1], expected_dist_sphere, atol=1e-3
            ), f"Expected hit on sphere at {expected_dist_sphere}, got {dists[0, 1]}"

            # Ray at 180 degrees (π) should miss
            assert jnp.allclose(
                dists[0, 2], 5.0
            ), f"Expected miss at 180 degrees, got {dists[0, 2]}"

            # Ray at 270 degrees (3π/2) should hit box
            expected_dist_box = 1.0 - 0.5  # Distance to center minus half width
            assert jnp.allclose(
                dists[0, 3], expected_dist_box, atol=1e-3
            ), f"Expected hit on box at {expected_dist_box}, got {dists[0, 3]}"

    def test_ray_filtering(self):
        with jax.disable_jit():
            world = World.create(batch_dim=1)

            # Create source agent (ray origin)
            source_agent = Agent.create(
                batch_dim=1,
                name="source",
                dim_p=2,
                dim_c=0,
            )
            # Add two target agents
            agent1 = Agent.create(
                batch_dim=1,
                name="agent1",
                dim_p=2,
                dim_c=0,
                shape=Sphere(radius=0.5),
            )
            agent2 = Agent.create(
                batch_dim=1,
                name="agent2",
                dim_p=2,
                dim_c=0,
                shape=Sphere(radius=0.5),
            )

            world = world.add_agent(source_agent).add_agent(agent1).add_agent(agent2)

            source_agent = source_agent.replace(
                state=source_agent.state.replace(pos=jnp.array([[0.0, 0.0]]))
            )
            agent1 = agent1.replace(
                state=agent1.state.replace(pos=jnp.array([[1.0, 0.0]]))
            )
            agent2 = agent2.replace(
                state=agent2.state.replace(pos=jnp.array([[2.0, 0.0]]))
            )
            world = world.replace(agents=[source_agent, agent1, agent2])

            # Test filtering
            angles = jnp.array([0.0])

            # Filter for agent2
            dists1 = world.cast_ray(
                source_agent,
                angles,
                max_range=5.0,
                entity_filter=lambda e: e.name == "agent2",
            )

            # Filter for agent1
            dists2 = world.cast_ray(
                source_agent,
                angles,
                max_range=5.0,
                entity_filter=lambda e: e.name == "agent1",
            )

            # Distance calculations:
            # For agent2: center at x=2.0, radius=0.5, so ray hits at 2.0 - 0.5 = 1.5
            assert jnp.allclose(
                dists1[0], 1.5
            ), f"Expected distance to agent2: 1.5 (center at 2.0 - radius 0.5), got {dists1[0]}"

            # For agent1: center at x=1.0, radius=0.5, so ray hits at 1.0 - 0.5 = 0.5
            assert jnp.allclose(
                dists2[0], 0.5
            ), f"Expected distance to agent1: 0.5 (center at 1.0 - radius 0.5), got {dists2[0]}"

    def test_edge_cases(self):
        with jax.disable_jit():
            world = World.create(batch_dim=1)

            # Create source agent (ray origin)
            source_agent = Agent.create(
                batch_dim=1,
                name="source",
                dim_p=2,
                dim_c=0,
            )
            # Add target agents
            sphere_agent = Agent.create(
                batch_dim=1,
                name="sphere",
                dim_p=2,
                dim_c=0,
                shape=Sphere(radius=0.5),
            )
            box_agent = Agent.create(
                batch_dim=1,
                name="box",
                dim_p=2,
                dim_c=0,
                shape=Box(length=1.0, width=1.0),
            )

            world = (
                world.add_agent(source_agent)
                .add_agent(sphere_agent)
                .add_agent(box_agent)
            )

            source_agent = source_agent.replace(
                state=source_agent.state.replace(pos=jnp.array([[0.0, 0.0]]))
            )
            sphere_agent = sphere_agent.replace(
                state=sphere_agent.state.replace(pos=jnp.array([[0.0, 0.0]]))
            )
            box_agent = box_agent.replace(
                state=box_agent.state.replace(
                    pos=jnp.array([[1.0, 0.0]]),
                    rot=jnp.array([[0.0]]),
                )
            )
            world = world.replace(agents=[source_agent, sphere_agent, box_agent])

            # Test empty angles array
            empty_angles = jnp.empty((1, 0))
            dists = world.cast_rays(
                source_agent, empty_angles, max_range=5.0, entity_filter=lambda e: True
            )
            assert dists.shape == (1, 0)

            # Test very small max_range
            angles = jnp.array([0.0])
            dists = world.cast_ray(
                source_agent, angles, max_range=0.1, entity_filter=lambda e: True
            )
            assert jnp.allclose(dists[0], 0.1)

            # Test parallel rays to box edges
            parallel_angles = jnp.array([jnp.pi / 2])  # Parallel to box edge
            dists = world.cast_ray(
                source_agent,
                parallel_angles,
                max_range=5.0,
                entity_filter=lambda e: e.name == "box",
            )
            assert jnp.allclose(dists[0], 5.0)  # Should miss

    def test_cast_ray_line_perpendicular(self):
        # Test ray collision with a line oriented perpendicular to the ray.
        with jax.disable_jit():
            world = World.create(batch_dim=1)
            source_agent = Agent.create(
                batch_dim=1,
                name="source",
                dim_p=2,
                dim_c=0,
            )
            # Create a line agent with a 2.0 length segment.
            line_agent = Agent.create(
                batch_dim=1,
                name="line",
                dim_p=2,
                dim_c=0,
                shape=Line(length=2.0),
            )
            world = world.add_agent(source_agent).add_agent(line_agent)

            source_agent = source_agent.replace(
                state=source_agent.state.replace(pos=jnp.array([[0.0, 0.0]]))
            )
            # Position the line at (2, 0) and rotate it 90° so it is vertical.
            line_agent = line_agent.replace(
                state=line_agent.state.replace(
                    pos=jnp.array([[2.0, 0.0]]),
                    rot=jnp.array([[jnp.pi / 2]]),
                )
            )
            world = world.replace(agents=[source_agent, line_agent])
            angles = jnp.array([0.0])
            dists = world.cast_ray(
                source_agent,
                angles,
                max_range=5.0,
                entity_filter=lambda e: e.name == "line",
            )
            # The ray should hit the vertical line at a distance of 2.0.
            assert jnp.allclose(dists[0], 2.0, atol=1e-3)

    def test_cast_ray_multiple_overlapping(self):
        # Test that when multiple objects lie along the ray, the closest hit is returned.
        with jax.disable_jit():
            world = World.create(batch_dim=1)
            source_agent = Agent.create(
                batch_dim=1,
                name="source",
                dim_p=2,
                dim_c=0,
            )
            sphere_agent = Agent.create(
                batch_dim=1,
                name="sphere",
                dim_p=2,
                dim_c=0,
                shape=Sphere(radius=0.5),
            )
            box_agent = Agent.create(
                batch_dim=1,
                name="box",
                dim_p=2,
                dim_c=0,
                shape=Box(length=1.0, width=1.0),
            )
            world = (
                world.add_agent(source_agent)
                .add_agent(sphere_agent)
                .add_agent(box_agent)
            )

            source_agent = source_agent.replace(
                state=source_agent.state.replace(pos=jnp.array([[0.0, 0.0]]))
            )
            sphere_agent = sphere_agent.replace(
                state=sphere_agent.state.replace(pos=jnp.array([[1.0, 0.0]]))
            )
            box_agent = box_agent.replace(
                state=box_agent.state.replace(
                    pos=jnp.array([[1.5, 0.0]]),
                    rot=jnp.array([[0.0]]),
                )
            )
            world = world.replace(agents=[source_agent, sphere_agent, box_agent])
            angles = jnp.array([0.0])
            dists = world.cast_ray(
                source_agent,
                angles,
                max_range=5.0,
                entity_filter=lambda e: True,
            )
            # The sphere is closer: expected hit at (1.0 - 0.5) = 0.5.
            assert jnp.allclose(dists[0], 0.5, atol=1e-3)

    def test_cast_ray_multiple_batches(self):
        # Test cast_ray in a multi-batch world (batch_dim > 1).
        with jax.disable_jit():
            world = World.create(batch_dim=2)
            source_agent = Agent.create(
                batch_dim=2,
                name="source",
                dim_p=2,
                dim_c=0,
            )
            sphere_agent = Agent.create(
                batch_dim=2,
                name="sphere",
                dim_p=2,
                dim_c=0,
                shape=Sphere(radius=0.5),
            )
            world = world.add_agent(source_agent).add_agent(sphere_agent)
            # For batch 0 and batch 1, set different positions.
            source_agent = source_agent.replace(
                state=source_agent.state.replace(
                    pos=jnp.array([[0.0, 0.0], [1.0, 1.0]])
                )
            )
            sphere_agent = sphere_agent.replace(
                state=sphere_agent.state.replace(
                    pos=jnp.array([[1.0, 0.0], [2.0, 1.0]])
                )
            )
            world = world.replace(agents=[source_agent, sphere_agent])
            # Provide one angle per batch.
            angles = jnp.array([0.0, 0.0])
            dists = world.cast_ray(
                source_agent,
                angles,
                max_range=5.0,
                entity_filter=lambda e: e.name == "sphere",
            )
            # For both batches, the expected hit is at distance 1.0 - 0.5 = 0.5.
            expected = jnp.array([0.5, 0.5])
            assert jnp.allclose(dists, expected, atol=1e-3)

    def test_cast_ray_invalid_shape(self):

        with jax.disable_jit():
            world = World.create(batch_dim=1)
            source_agent = Agent.create(
                batch_dim=1,
                name="source",
                dim_p=2,
                dim_c=0,
            )
            dummy_agent = Agent.create(
                batch_dim=1,
                name="dummy",
                dim_p=2,
                dim_c=0,
                shape=DummyShape(),
            )
            world = world.add_agent(source_agent).add_agent(dummy_agent)
            source_agent = source_agent.replace(
                state=source_agent.state.replace(pos=jnp.array([[0.0, 0.0]]))
            )
            dummy_agent = dummy_agent.replace(
                state=dummy_agent.state.replace(pos=jnp.array([[1.0, 0.0]]))
            )
            world = world.replace(agents=[source_agent, dummy_agent])
            angles = jnp.array([0.0])
            import pytest

            with pytest.raises(RuntimeError):
                _ = world.cast_ray(
                    source_agent,
                    angles,
                    max_range=5.0,
                    entity_filter=lambda e: e.name == "dummy",
                )

    def test_cast_rays_no_targets(self):
        # Test that cast_rays returns max_range when there are no target agents.
        with jax.disable_jit():
            world = World.create(batch_dim=1)
            source_agent = Agent.create(
                batch_dim=1,
                name="source",
                dim_p=2,
                dim_c=0,
            )
            world = world.add_agent(source_agent)
            source_agent = source_agent.replace(
                state=source_agent.state.replace(pos=jnp.array([[0.0, 0.0]]))
            )
            world = world.replace(agents=[source_agent])
            angles = jnp.array([0.0, jnp.pi / 2])
            dists = world.cast_rays(
                source_agent,
                angles,
                max_range=5.0,
                entity_filter=lambda e: True,
            )
            # Since there are no other agents, all rays should return max_range.
            assert jnp.allclose(dists, jnp.full((1, 2), 5.0))

    def test_cast_ray_negative_angles(self):
        # Test that negative angles (e.g., -pi/2) are handled correctly.
        with jax.disable_jit():
            world = World.create(batch_dim=1)
            source_agent = Agent.create(
                batch_dim=1,
                name="source",
                dim_p=2,
                dim_c=0,
            )
            sphere_agent = Agent.create(
                batch_dim=1,
                name="sphere",
                dim_p=2,
                dim_c=0,
                shape=Sphere(radius=0.5),
            )
            world = world.add_agent(source_agent).add_agent(sphere_agent)
            source_agent = source_agent.replace(
                state=source_agent.state.replace(pos=jnp.array([[0.0, 0.0]]))
            )
            # Place the sphere below the source.
            sphere_agent = sphere_agent.replace(
                state=sphere_agent.state.replace(pos=jnp.array([[0.0, -1.0]]))
            )
            world = world.replace(agents=[source_agent, sphere_agent])
            angles = jnp.array([-jnp.pi / 2])
            dists = world.cast_ray(
                source_agent,
                angles,
                max_range=5.0,
                entity_filter=lambda e: e.name == "sphere",
            )
            # The ray at -pi/2 should hit the sphere at a distance of 1.0 - 0.5 = 0.5.
            assert jnp.allclose(dists[0], 0.5, atol=1e-3)

    def test_cast_ray_large_angles(self):
        # Test that angles larger than 2*pi (e.g., 2*pi) are correctly wrapped.
        with jax.disable_jit():
            world = World.create(batch_dim=1)
            source_agent = Agent.create(
                batch_dim=1,
                name="source",
                dim_p=2,
                dim_c=0,
            )
            sphere_agent = Agent.create(
                batch_dim=1,
                name="sphere",
                dim_p=2,
                dim_c=0,
                shape=Sphere(radius=0.5),
            )
            world = world.add_agent(source_agent).add_agent(sphere_agent)
            source_agent = source_agent.replace(
                state=source_agent.state.replace(pos=jnp.array([[0.0, 0.0]]))
            )
            sphere_agent = sphere_agent.replace(
                state=sphere_agent.state.replace(pos=jnp.array([[1.0, 0.0]]))
            )
            world = world.replace(agents=[source_agent, sphere_agent])
            # 2*pi is equivalent to 0 radians.
            angles = jnp.array([2 * jnp.pi])
            dists = world.cast_ray(
                source_agent,
                angles,
                max_range=5.0,
                entity_filter=lambda e: e.name == "sphere",
            )
            # Expected hit: distance = 1.0 - 0.5 = 0.5.
            assert jnp.allclose(dists[0], 0.5, atol=1e-3)


class TestGetDistanceFromPoint:
    def test_sphere_distance(self):
        """Test that a Sphere returns the norm difference minus its radius."""
        # Create a world with a single environment (batch_dim=1)
        world = World.create(batch_dim=1)
        world = world.reset()
        # Create a sphere agent at position [1, 1] with radius 0.5.
        sphere_agent = Agent.create(
            batch_dim=1, name="sphere", dim_p=2, dim_c=0, shape=Sphere(radius=0.5)
        )
        world = world.add_agent(sphere_agent)
        sphere_agent = sphere_agent.replace(
            state=sphere_agent.state.replace(pos=jnp.array([[1.0, 1.0]]))
        )
        # Choose a test point.
        test_point = jnp.array([[2.0, 1.0]])
        # Expected: norm([1,1] - [2,1]) = 1.0, then 1.0 - 0.5 = 0.5.
        expected = 0.5
        result = world.get_distance_from_point(sphere_agent, test_point)
        assert jnp.allclose(result, expected, atol=1e-3)

    def test_invalid_shape(self):
        """Test that an unsupported shape raises a RuntimeError."""
        world = World.create(batch_dim=1)
        dummy_agent = Agent.create(
            batch_dim=1, name="dummy", dim_p=2, dim_c=0, shape=DummyShape()
        )
        dummy_agent = dummy_agent.replace(
            state=dummy_agent.state.replace(pos=jnp.array([[0.0, 0.0]]))
        )
        test_point = jnp.array([[1.0, 1.0]])
        with pytest.raises(RuntimeError):
            _ = world.get_distance_from_point(dummy_agent, test_point)

    def test_env_index_selection(self):
        """Test that when an env_index is provided, a scalar from the batch is returned."""
        # Create a world with batch_dim=2.
        world = World.create(batch_dim=2)
        sphere_agent = Agent.create(
            batch_dim=2, name="sphere", dim_p=2, dim_c=0, shape=Sphere(radius=0.5)
        )
        # Set different positions for the two environments.
        sphere_agent = sphere_agent.replace(
            state=sphere_agent.state.replace(pos=jnp.array([[1.0, 1.0], [2.0, 2.0]]))
        )
        # Choose test points for each batch.
        test_point = jnp.array([[2.0, 1.0], [3.0, 2.0]])
        # For batch 0: expected distance = norm([1,1]-[2,1]) = 1.0 - 0.5 = 0.5.
        # For batch 1: expected distance = norm([2,2]-[3,2]) = 1.0 - 0.5 = 0.5.
        result0 = world.get_distance_from_point(sphere_agent, test_point, env_index=0)
        result1 = world.get_distance_from_point(sphere_agent, test_point, env_index=1)
        assert jnp.allclose(result0, 0.5, atol=1e-3)
        assert jnp.allclose(result1, 0.5, atol=1e-3)


class TestDistanceAndOverlap:
    # --------- get_distance tests ---------

    def test_get_distance_sphere_sphere(self):
        """
        For two spheres:
          get_distance = ||A.pos - B.pos|| - (A.radius + B.radius)
        """
        world = World.create(batch_dim=1)
        # Create two spheres.
        sphere_a = Agent.create(
            batch_dim=1, name="A", dim_p=2, dim_c=0, shape=Sphere(radius=0.5)
        )
        sphere_b = Agent.create(
            batch_dim=1, name="B", dim_p=2, dim_c=0, shape=Sphere(radius=0.5)
        )
        sphere_a = sphere_a.replace(
            state=sphere_a.state.replace(pos=jnp.array([[0.0, 0.0]]))
        )
        sphere_b = sphere_b.replace(
            state=sphere_b.state.replace(pos=jnp.array([[2.0, 0.0]]))
        )
        world = world.replace(agents=[sphere_a, sphere_b])
        # Expected: norm([0,0]-[2,0]) = 2, then 2 - (0.5+0.5) = 1.
        result = world.get_distance(sphere_a, sphere_b)
        assert jnp.allclose(result, 1.0, atol=1e-3)

    def test_get_distance_box_sphere_non_overlapping(self):
        """
        For a box vs. sphere that do not overlap:
          The box's get_distance_from_point returns:
             ||test_point - closest_point|| - LINE_MIN_DIST.
          For a box centered at [0,0] (width=4, length=4) and a sphere at [10,0],
          the closest point on the box is [2,0], so distance = 8 - LINE_MIN_DIST.
          Then, get_distance subtracts the sphere radius (0.5),
          so expected = 8 - LINE_MIN_DIST - 0.5.
        """
        world = World.create(batch_dim=1)
        box = Agent.create(
            batch_dim=1, name="Box", dim_p=2, dim_c=0, shape=Box(length=4.0, width=4.0)
        )
        sphere = Agent.create(
            batch_dim=1, name="Sphere", dim_p=2, dim_c=0, shape=Sphere(radius=0.5)
        )
        box = box.replace(
            state=box.state.replace(pos=jnp.array([[0.0, 0.0]]), rot=jnp.array([[0.0]]))
        )
        # Place sphere well outside the box.
        sphere = sphere.replace(
            state=sphere.state.replace(pos=jnp.array([[10.0, 0.0]]))
        )
        world = world.replace(agents=[box, sphere])
        # Expected: distance from [10,0] to box boundary [2,0] = 8, then subtract LINE_MIN_DIST and sphere.radius.
        expected = 8.0 - LINE_MIN_DIST - 0.5
        result = world.get_distance(box, sphere)
        # In this branch, after computing get_distance_from_point, if not overlapping, the value remains unchanged.
        assert jnp.allclose(result, expected, atol=1e-3)

    def test_get_distance_box_sphere_overlapping(self):
        """
        For a box vs. sphere that overlap:
          When the sphere is inside the box, the computed distance would be negative.
          In this branch the code forces the final return value to -1.
        """
        world = World.create(batch_dim=1)
        box = Agent.create(
            batch_dim=1, name="Box", dim_p=2, dim_c=0, shape=Box(length=4.0, width=4.0)
        )
        sphere = Agent.create(
            batch_dim=1, name="Sphere", dim_p=2, dim_c=0, shape=Sphere(radius=0.5)
        )
        box = box.replace(
            state=box.state.replace(pos=jnp.array([[0.0, 0.0]]), rot=jnp.array([[0.0]]))
        )
        # Place the sphere inside the box.
        sphere = sphere.replace(state=sphere.state.replace(pos=jnp.array([[1.0, 0.0]])))
        world = world.replace(agents=[box, sphere])
        result = world.get_distance(box, sphere)
        # The code forces overlapping cases to return -1.
        assert jnp.allclose(result, -1.0, atol=1e-3)

    def test_get_distance_line_sphere(self):
        """
        For a line vs. sphere:
        The line is assumed to be centered at its pos.
        For a line at [0,0] with length 5 (extending from [-2.5,0] to [2.5,0])
        and a sphere at [6,0] (radius 0.5), the closest point on the line is [2.5,0].
        Then, distance = norm([6,0] - [2.5,0]) = 3.5.
        Subtract LINE_MIN_DIST and the sphere's radius:
            expected = 3.5 - LINE_MIN_DIST - 0.5 = 3.0 - LINE_MIN_DIST.
        """
        world = World.create(batch_dim=1)
        line = Agent.create(
            batch_dim=1, name="Line", dim_p=2, dim_c=0, shape=Line(length=5.0)
        )
        sphere = Agent.create(
            batch_dim=1, name="Sphere", dim_p=2, dim_c=0, shape=Sphere(radius=0.5)
        )
        line = line.replace(
            state=line.state.replace(
                pos=jnp.array([[0.0, 0.0]]), rot=jnp.array([[0.0]])
            )
        )
        sphere = sphere.replace(state=sphere.state.replace(pos=jnp.array([[6.0, 0.0]])))
        world = world.replace(agents=[line, sphere])
        expected = 3.0 - LINE_MIN_DIST
        result = world.get_distance(line, sphere)
        assert jnp.allclose(result, expected, atol=1e-3)

    def test_get_distance_line_line(self):
        """
        For two lines:
          Let line A be at [0,0] (length 5, along x-axis) and
          line B be at [0,1] (length 5, along x-axis).
          Their closest points should be at the left endpoints: [0,0] and [0,1],
          so distance = 1 - LINE_MIN_DIST.
        """
        world = World.create(batch_dim=1)
        line_a = Agent.create(
            batch_dim=1, name="LineA", dim_p=2, dim_c=0, shape=Line(length=5.0)
        )
        line_b = Agent.create(
            batch_dim=1, name="LineB", dim_p=2, dim_c=0, shape=Line(length=5.0)
        )
        line_a = line_a.replace(
            state=line_a.state.replace(
                pos=jnp.array([[0.0, 0.0]]), rot=jnp.array([[0.0]])
            )
        )
        line_b = line_b.replace(
            state=line_b.state.replace(
                pos=jnp.array([[0.0, 1.0]]), rot=jnp.array([[0.0]])
            )
        )
        world = world.replace(agents=[line_a, line_b])
        expected = 1.0 - LINE_MIN_DIST
        result = world.get_distance(line_a, line_b)
        assert jnp.allclose(result, expected, atol=1e-3)

    def test_get_distance_box_line(self):
        """
        For a box vs. a line:
        Use a box at [0,0] (4x4) and a line at [3,0] (length 5, centered at [3,0]).
        For a box centered at [0,0], the right boundary is at x=2.
        A line with length 5 centered at [3,0] has endpoints at [0.5,0] and [5.5,0],
        so the closest point on the line to the box is [2,0].
        Thus, the gap is 0, and the expected value is 0 - LINE_MIN_DIST = -LINE_MIN_DIST.
        """
        world = World.create(batch_dim=1)
        box = Agent.create(
            batch_dim=1, name="Box", dim_p=2, dim_c=0, shape=Box(length=4.0, width=4.0)
        )
        line = Agent.create(
            batch_dim=1, name="Line", dim_p=2, dim_c=0, shape=Line(length=5.0)
        )
        box = box.replace(
            state=box.state.replace(pos=jnp.array([[0.0, 0.0]]), rot=jnp.array([[0.0]]))
        )
        # Place line so that its center is at [3, 0]
        line = line.replace(
            state=line.state.replace(
                pos=jnp.array([[3.0, 0.0]]), rot=jnp.array([[0.0]])
            )
        )
        world = world.replace(agents=[box, line])
        expected = -LINE_MIN_DIST  # Updated expected value
        result = world.get_distance(box, line)
        assert jnp.allclose(result, expected, atol=1e-3)

    # TODO: Don't understand this test but the output seems to match VMAS library's output.
    def test_get_distance_box_box(self):
        """
        For two boxes:
        Let box A be at [0,0] (4×4) and box B be at [0,6] (4×4).
        Although our intuitive expectation might be a gap of 2 (2 – LINE_MIN_DIST),
        the implementation of get_distance for boxes computes the closest distance as 0 (i.e. the boxes are just touching)
        and then returns 0 – LINE_MIN_DIST.
        Thus, the expected result is –LINE_MIN_DIST.
        """
        world = World.create(batch_dim=1)
        box_a = Agent.create(
            batch_dim=1, name="BoxA", dim_p=2, dim_c=0, shape=Box(length=4.0, width=4.0)
        )
        box_b = Agent.create(
            batch_dim=1, name="BoxB", dim_p=2, dim_c=0, shape=Box(length=4.0, width=4.0)
        )
        box_a = box_a.replace(
            state=box_a.state.replace(
                pos=jnp.array([[0.0, 0.0]]), rot=jnp.array([[0.0]])
            )
        )
        box_b = box_b.replace(
            state=box_b.state.replace(
                pos=jnp.array([[0.0, 6.0]]), rot=jnp.array([[0.0]])
            )
        )
        world = world.replace(agents=[box_a, box_b])
        expected = 2 - LINE_MIN_DIST
        result = world.get_distance(box_a, box_b)
        assert jnp.allclose(result, expected, atol=1e-3)

    def test_get_distance_invalid(self):
        """
        Passing an unsupported shape combination should raise a RuntimeError.
        """

        world = World.create(batch_dim=1)
        a = Agent.create(
            batch_dim=1, name="A", dim_p=2, dim_c=0, shape=Sphere(radius=0.5)
        )
        b = Agent.create(batch_dim=1, name="B", dim_p=2, dim_c=0, shape=DummyShape())
        a = a.replace(state=a.state.replace(pos=jnp.array([[0.0, 0.0]])))
        b = b.replace(state=b.state.replace(pos=jnp.array([[1.0, 1.0]])))
        world = world.replace(agents=[a, b])
        with pytest.raises(RuntimeError):
            _ = world.get_distance(a, b)

    # --------- is_overlapping tests ---------

    def test_is_overlapping_sphere_sphere_overlapping(self):
        """
        For two spheres that overlap:
          If sphere A is at [0,0] (radius 0.5) and sphere B is at [0.4,0] (radius 0.5),
          then get_distance returns a negative value and is_overlapping should be True.
        """
        world = World.create(batch_dim=1)
        sphere_a = Agent.create(
            batch_dim=1, name="A", dim_p=2, dim_c=0, shape=Sphere(radius=0.5)
        )
        sphere_b = Agent.create(
            batch_dim=1, name="B", dim_p=2, dim_c=0, shape=Sphere(radius=0.5)
        )
        sphere_a = sphere_a.replace(
            state=sphere_a.state.replace(pos=jnp.array([[0.0, 0.0]]))
        )
        sphere_b = sphere_b.replace(
            state=sphere_b.state.replace(pos=jnp.array([[0.4, 0.0]]))
        )
        world = world.replace(agents=[sphere_a, sphere_b])
        assert world.is_overlapping(sphere_a, sphere_b)

    def test_is_overlapping_sphere_sphere_non_overlapping(self):
        """
        For two spheres that do not overlap:
          E.g. sphere A at [0,0] and sphere B at [2,0] should not overlap.
        """
        world = World.create(batch_dim=1)
        sphere_a = Agent.create(
            batch_dim=1, name="A", dim_p=2, dim_c=0, shape=Sphere(radius=0.5)
        )
        sphere_b = Agent.create(
            batch_dim=1, name="B", dim_p=2, dim_c=0, shape=Sphere(radius=0.5)
        )
        sphere_a = sphere_a.replace(
            state=sphere_a.state.replace(pos=jnp.array([[0.0, 0.0]]))
        )
        sphere_b = sphere_b.replace(
            state=sphere_b.state.replace(pos=jnp.array([[2.0, 0.0]]))
        )
        world = world.replace(agents=[sphere_a, sphere_b])
        assert not world.is_overlapping(sphere_a, sphere_b)

    def test_is_overlapping_box_box_overlapping(self):
        """
        Two boxes that overlap (e.g. centers at [0,0] and [0,1]) should be overlapping.
        """
        world = World.create(batch_dim=1)
        box_a = Agent.create(
            batch_dim=1, name="A", dim_p=2, dim_c=0, shape=Box(length=4.0, width=4.0)
        )
        box_b = Agent.create(
            batch_dim=1, name="B", dim_p=2, dim_c=0, shape=Box(length=4.0, width=4.0)
        )
        box_a = box_a.replace(
            state=box_a.state.replace(
                pos=jnp.array([[0.0, 0.0]]), rot=jnp.array([[0.0]])
            )
        )
        box_b = box_b.replace(
            state=box_b.state.replace(
                pos=jnp.array([[0.0, 1.0]]), rot=jnp.array([[0.0]])
            )
        )
        world = world.replace(agents=[box_a, box_b])
        assert world.is_overlapping(box_a, box_b)

    def test_is_overlapping_box_box_non_overlapping(self):
        """
        Two boxes that do not overlap (e.g. centers at [0,0] and [0,6]) should not be overlapping.
        """
        world = World.create(batch_dim=1)
        box_a = Agent.create(
            batch_dim=1, name="A", dim_p=2, dim_c=0, shape=Box(length=4.0, width=4.0)
        )
        box_b = Agent.create(
            batch_dim=1, name="B", dim_p=2, dim_c=0, shape=Box(length=4.0, width=4.0)
        )
        box_a = box_a.replace(
            state=box_a.state.replace(
                pos=jnp.array([[0.0, 0.0]]), rot=jnp.array([[0.0]])
            )
        )
        box_b = box_b.replace(
            state=box_b.state.replace(
                pos=jnp.array([[0.0, 6.0]]), rot=jnp.array([[0.0]])
            )
        )
        world = world.replace(agents=[box_a, box_b])
        assert not world.is_overlapping(box_a, box_b)

    def test_is_overlapping_box_sphere_overlapping(self):
        """
        A box and a sphere that overlap:
          For a box at [0,0] (4x4) and a sphere centered at [0,0] (radius 0.5),
          the sphere lies within the box.
        """
        world = World.create(batch_dim=1)
        box = Agent.create(
            batch_dim=1, name="Box", dim_p=2, dim_c=0, shape=Box(length=4.0, width=4.0)
        )
        sphere = Agent.create(
            batch_dim=1, name="Sphere", dim_p=2, dim_c=0, shape=Sphere(radius=0.5)
        )
        box = box.replace(
            state=box.state.replace(pos=jnp.array([[0.0, 0.0]]), rot=jnp.array([[0.0]]))
        )
        sphere = sphere.replace(state=sphere.state.replace(pos=jnp.array([[0.0, 0.0]])))
        world = world.replace(agents=[box, sphere])
        assert world.is_overlapping(box, sphere)

    def test_is_overlapping_box_sphere_non_overlapping(self):
        """
        A box and a sphere that do not overlap:
          For a box at [0,0] (4x4) and a sphere at [10,0] (radius 0.5),
          they should not overlap.
        """
        world = World.create(batch_dim=1)
        box = Agent.create(
            batch_dim=1, name="Box", dim_p=2, dim_c=0, shape=Box(length=4.0, width=4.0)
        )
        sphere = Agent.create(
            batch_dim=1, name="Sphere", dim_p=2, dim_c=0, shape=Sphere(radius=0.5)
        )
        box = box.replace(
            state=box.state.replace(pos=jnp.array([[0.0, 0.0]]), rot=jnp.array([[0.0]]))
        )
        sphere = sphere.replace(
            state=sphere.state.replace(pos=jnp.array([[10.0, 0.0]]))
        )
        world = world.replace(agents=[box, sphere])
        assert not world.is_overlapping(box, sphere)

    def test_is_overlapping_line_sphere_overlapping(self):
        """
        For a line and a sphere that overlap:
          Place a line at [0,0] (length 5, along x-axis) and a sphere at [2,0] (radius 0.5).
          The projection of [2,0] onto the line is [2,0], so the distance will be negative.
        """
        world = World.create(batch_dim=1)
        line = Agent.create(
            batch_dim=1, name="Line", dim_p=2, dim_c=0, shape=Line(length=5.0)
        )
        sphere = Agent.create(
            batch_dim=1, name="Sphere", dim_p=2, dim_c=0, shape=Sphere(radius=0.5)
        )
        line = line.replace(
            state=line.state.replace(
                pos=jnp.array([[0.0, 0.0]]), rot=jnp.array([[0.0]])
            )
        )
        sphere = sphere.replace(state=sphere.state.replace(pos=jnp.array([[2.0, 0.0]])))
        world = world.replace(agents=[line, sphere])
        assert world.is_overlapping(line, sphere)

    def test_is_overlapping_line_sphere_non_overlapping(self):
        """
        For a line and a sphere that do not overlap:
          Place a line at [0,0] (length 5) and a sphere at [6,0] (radius 0.5).
        """
        world = World.create(batch_dim=1)
        line = Agent.create(
            batch_dim=1, name="Line", dim_p=2, dim_c=0, shape=Line(length=5.0)
        )
        sphere = Agent.create(
            batch_dim=1, name="Sphere", dim_p=2, dim_c=0, shape=Sphere(radius=0.5)
        )
        line = line.replace(
            state=line.state.replace(
                pos=jnp.array([[0.0, 0.0]]), rot=jnp.array([[0.0]])
            )
        )
        sphere = sphere.replace(state=sphere.state.replace(pos=jnp.array([[6.0, 0.0]])))
        world = world.replace(agents=[line, sphere])
        assert not world.is_overlapping(line, sphere)

    def test_is_overlapping_line_line_overlapping(self):
        """
        For two lines that intersect:
          For example, one horizontal (rot=0) and one vertical (rot=pi/2) starting at the same point.
          They should overlap.
        """
        world = World.create(batch_dim=1)
        line_h = Agent.create(
            batch_dim=1, name="Horiz", dim_p=2, dim_c=0, shape=Line(length=5.0)
        )
        line_v = Agent.create(
            batch_dim=1, name="Vert", dim_p=2, dim_c=0, shape=Line(length=5.0)
        )
        line_h = line_h.replace(
            state=line_h.state.replace(
                pos=jnp.array([[0.0, 0.0]]), rot=jnp.array([[0.0]])
            )
        )
        line_v = line_v.replace(
            state=line_v.state.replace(
                pos=jnp.array([[0.0, 0.0]]), rot=jnp.array([[jnp.pi / 2]])
            )
        )
        world = world.replace(agents=[line_h, line_v])
        assert world.is_overlapping(line_h, line_v)

    def test_is_overlapping_line_line_non_overlapping(self):
        """
        For two lines that are far apart:
          For example, one at [0,0] and one at [10,10] should not overlap.
        """
        world = World.create(batch_dim=1)
        line_a = Agent.create(
            batch_dim=1, name="A", dim_p=2, dim_c=0, shape=Line(length=5.0)
        )
        line_b = Agent.create(
            batch_dim=1, name="B", dim_p=2, dim_c=0, shape=Line(length=5.0)
        )
        line_a = line_a.replace(
            state=line_a.state.replace(
                pos=jnp.array([[0.0, 0.0]]), rot=jnp.array([[0.0]])
            )
        )
        line_b = line_b.replace(
            state=line_b.state.replace(
                pos=jnp.array([[10.0, 10.0]]), rot=jnp.array([[0.0]])
            )
        )
        world = world.replace(agents=[line_a, line_b])
        assert not world.is_overlapping(line_a, line_b)

    def test_is_overlapping_box_line_overlapping(self):
        """
        For a box and a line that overlap:
          Place a box at [0,0] (4x4) and a line that starts at [0,0] (length 5).
          They intersect.
        """
        world = World.create(batch_dim=1)
        box = Agent.create(
            batch_dim=1, name="Box", dim_p=2, dim_c=0, shape=Box(length=4.0, width=4.0)
        )
        line = Agent.create(
            batch_dim=1, name="Line", dim_p=2, dim_c=0, shape=Line(length=5.0)
        )
        box = box.replace(
            state=box.state.replace(pos=jnp.array([[0.0, 0.0]]), rot=jnp.array([[0.0]]))
        )
        line = line.replace(
            state=line.state.replace(
                pos=jnp.array([[0.0, 0.0]]), rot=jnp.array([[0.0]])
            )
        )
        world = world.replace(agents=[box, line])
        assert world.is_overlapping(box, line)

    def test_is_overlapping_box_line_non_overlapping(self):
        """
        For a box and a line that do not overlap:
          Place a box at [0,0] (4x4) and a line starting at [10,0] (length 5).
        """
        world = World.create(batch_dim=1)
        box = Agent.create(
            batch_dim=1, name="Box", dim_p=2, dim_c=0, shape=Box(length=4.0, width=4.0)
        )
        line = Agent.create(
            batch_dim=1, name="Line", dim_p=2, dim_c=0, shape=Line(length=5.0)
        )
        box = box.replace(
            state=box.state.replace(pos=jnp.array([[0.0, 0.0]]), rot=jnp.array([[0.0]]))
        )
        line = line.replace(
            state=line.state.replace(
                pos=jnp.array([[10.0, 0.0]]), rot=jnp.array([[0.0]])
            )
        )
        world = world.replace(agents=[box, line])
        assert not world.is_overlapping(box, line)


class TestForceApplication:
    def test_apply_action_force(self):
        # Create a basic world and agent
        world = World.create(
            batch_dim=1,
            dim_p=2,
            dim_c=0,
        )
        agent1 = Agent.create(
            batch_dim=1,
            name="test",
            dim_p=2,
            dim_c=0,
            movable=True,
            max_f=2.0,
        )
        agent2 = Agent.create(
            batch_dim=1,
            name="test2",
            dim_p=2,
            dim_c=0,
            movable=True,
            f_range=1.5,
        )
        world = world.add_agent(agent1)
        world = world.add_agent(agent2)
        world = world.reset()

        # Test force clamping with max_f
        agent1 = agent1.replace(
            state=agent1.state.replace(force=jnp.array([[3.0, 0.0]]))
        )
        agent1, world = world._apply_action_force(agent1)
        assert jnp.allclose(agent1.state.force, jnp.array([[2.0, 0.0]]))
        assert jnp.allclose(world.force_dict[agent1.name], jnp.array([[2.0, 0.0]]))

        # Test force range limiting
        agent2 = agent2.replace(
            state=agent2.state.replace(force=jnp.array([[1.0, 2.0]]))
        )
        agent2, world = world._apply_action_force(agent2)
        assert jnp.allclose(agent2.state.force, jnp.array([[1.0, 1.5]]))
        assert jnp.allclose(world.force_dict[agent2.name], jnp.array([[1.0, 1.5]]))

    def test_apply_action_torque(self):
        world = World.create(batch_dim=1)
        agent = Agent.create(
            batch_dim=1,
            name="test",
            dim_p=2,
            dim_c=0,
            rotatable=True,
            t_range=1.5,
        )
        agent2 = Agent.create(
            batch_dim=1,
            name="test2",
            dim_p=2,
            dim_c=0,
            rotatable=True,
            max_t=2.0,
        )
        world = world.add_agent(agent)
        world = world.add_agent(agent2)
        world = world.reset()

        # Test torque clamping with max_t
        agent = agent.replace(state=agent.state.replace(torque=jnp.array([[3.0]])))
        agent, world = world._apply_action_torque(agent)
        assert jnp.allclose(agent.state.torque, jnp.array([[1.5]]))
        assert jnp.allclose(world.torque_dict[agent.name], jnp.array([[1.5]]))

        # Test torque range limiting
        agent2 = agent2.replace(state=agent2.state.replace(torque=jnp.array([[-3.0]])))
        agent2, world = world._apply_action_torque(agent2)
        assert jnp.allclose(agent2.state.torque, jnp.array([[-2.0]]))
        assert jnp.allclose(world.torque_dict[agent2.name], jnp.array([[-2.0]]))

    def test_apply_gravity(self):
        world = World.create(batch_dim=1, gravity=jnp.array([0.0, -9.81]))
        agent = Agent.create(
            batch_dim=1,
            name="test",
            dim_p=2,
            dim_c=0,
            movable=True,
            mass=2.0,
        )
        agent2 = Agent.create(
            batch_dim=1,
            name="test2",
            dim_p=2,
            dim_c=0,
            movable=True,
            mass=2.0,
            gravity=jnp.asarray([[0.0, -5.0]]),
        )
        world = world.add_agent(agent)
        world = world.add_agent(agent2)
        world = world.reset()

        # Test global gravity
        _, world = world._apply_gravity(agent)
        expected_force = jnp.array([[0.0, -19.62]])  # 2.0 * -9.81
        assert jnp.allclose(world.force_dict[agent.name], expected_force)

        # Test entity-specific gravity
        agent2 = agent2.replace(gravity=jnp.asarray([[0.0, -5.0]]))
        _, world = world._apply_gravity(agent2)
        expected_force = jnp.array([[0.0, -29.62]])  # Previous + (2.0 * -5.0)
        assert jnp.allclose(world.force_dict[agent2.name], expected_force)

    def test_apply_friction_force(self):
        world = World.create(batch_dim=1, dt=0.1, substeps=1)
        agent = Agent.create(
            batch_dim=1,
            name="test",
            dim_p=2,
            dim_c=0,
            movable=True,
            rotatable=True,
            mass=1.0,
            linear_friction=0.5,
            angular_friction=0.3,
            shape=Box(
                length=1.0,
                width=1.0,
            ),
        )
        world = world.add_agent(agent)
        world = world.reset()
        # Test linear friction
        agent = agent.replace(state=agent.state.replace(vel=jnp.array([[1.0, 0.0]])))
        _, world = world._apply_friction_force(agent)
        assert jnp.all(world.force_dict[agent.name] <= 0)  # Friction opposes motion

        # Test angular friction
        agent = agent.replace(state=agent.state.replace(ang_vel=jnp.array([[1.0]])))
        _, world = world._apply_friction_force(agent)
        assert jnp.all(world.torque_dict[agent.name] <= 0)  # Friction opposes rotation
