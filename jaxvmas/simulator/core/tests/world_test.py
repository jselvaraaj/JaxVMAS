import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from jaxvmas.simulator.core.action import Action
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
            name="test_agent",
            movable=True,
            rotatable=True,
        )

    @pytest.fixture
    def landmark(self):
        return Landmark.create(name="test_landmark")

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
            name="agent1",
            movable=True,
        )
        agent2 = Agent.create(
            name="agent2",
            movable=True,
        )

        world = basic_world.add_agent(agent1).add_agent(agent2)

        agent1, agent2 = world.agents

        # Test collision detection
        assert world.collides(agent1, agent2)

        # Test no collision with non-collidable entity
        landmark = Landmark.create(name="landmark", collide=False)
        world = world.add_landmark(landmark)
        (landmark,) = world.landmarks
        assert not world.collides(agent1, landmark)

    def test_communication(self, basic_world: World):
        # Create world with communication
        world = World.create(batch_dim=2, dim_c=3)
        agent = Agent.create(name="agent", silent=False)
        world = world.add_agent(agent)
        (agent,) = world.agents

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
            name="agent1",
            movable=True,
        )
        agent2 = Agent.create(
            name="agent2",
            movable=True,
        )

        world = world.add_agent(agent1).add_agent(agent2)
        agent1, agent2 = world.agents

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
            name="agent",
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
        world = World.create(batch_dim=1, substeps=10)

        # Create two colliding agents with no initial forces
        agent1 = Agent.create(
            name="collider1",
            mass=1.0,
        )
        agent2 = Agent.create(
            name="collider2",
            mass=1.0,
        )
        world = world.add_agent(agent1).add_agent(agent2)
        agent1, agent2 = world.agents
        agent1 = agent1.replace(
            state=agent1.state.replace(
                pos=jnp.array([[0.0, 0.0]]),
            ),
            shape=Sphere(radius=0.1),
        )

        agent2 = agent2.replace(
            state=agent2.state.replace(
                pos=jnp.array([[0.15, 0.0]]),
            ),
            shape=Sphere(radius=0.1),
        )

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
        agent1 = Agent.create(name="joint1")
        agent2 = Agent.create(name="joint2")
        world = World.create(
            batch_dim=1,
            dt=0.1,
            substeps=10,  # Higher substeps for joint stability
            joint_force=500.0,  # Increase joint force for stronger constraint
            linear_friction=0.0,  # Remove friction to allow easier movement
            drag=0.1,  # Reduce drag for smoother motion
        )
        world = world.add_agent(agent1).add_agent(agent2)
        agent1, agent2 = world.agents
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
        agent = Agent.create(name="boundary_test")
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
                name="source",
            )
            # Create target sphere
            sphere_agent = Agent.create(
                name="sphere",
                shape=Sphere(radius=0.5),
            )
            world = world.add_agent(source_agent).add_agent(sphere_agent)
            source_agent, sphere_agent = world.agents

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
                name="source",
            )
            # Create target box
            box_agent = Agent.create(
                name="box",
                shape=Box(length=1.0, width=1.0),
            )
            world = world.add_agent(source_agent).add_agent(box_agent)
            source_agent, box_agent = world.agents

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
                name="source",
            )
            # Add sphere
            sphere = Agent.create(
                name="sphere",
                shape=Sphere(radius=0.5),
            )
            # Add box
            box = Agent.create(
                name="box",
                shape=Box(length=1.0, width=1.0),
            )

            world = world.add_agent(source_agent).add_agent(sphere).add_agent(box)
            source_agent, sphere, box = world.agents

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
                name="source",
            )
            # Add two target agents
            agent1 = Agent.create(
                name="agent1",
                shape=Sphere(radius=0.5),
            )
            agent2 = Agent.create(
                name="agent2",
                shape=Sphere(radius=0.5),
            )

            world = world.add_agent(source_agent).add_agent(agent1).add_agent(agent2)

            source_agent, agent1, agent2 = world.agents

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
                name="source",
            )
            # Add target agents
            sphere_agent = Agent.create(
                name="sphere",
                shape=Sphere(radius=0.5),
            )
            box_agent = Agent.create(
                name="box",
                shape=Box(length=1.0, width=1.0),
            )

            world = (
                world.add_agent(source_agent)
                .add_agent(sphere_agent)
                .add_agent(box_agent)
            )
            source_agent, sphere_agent, box_agent = world.agents

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
                name="source",
            )
            # Create a line agent with a 2.0 length segment.
            line_agent = Agent.create(
                name="line",
                shape=Line(length=2.0),
            )
            world = world.add_agent(source_agent).add_agent(line_agent)
            source_agent, line_agent = world.agents

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
                name="source",
            )
            sphere_agent = Agent.create(
                name="sphere",
                shape=Sphere(radius=0.5),
            )
            box_agent = Agent.create(
                name="box",
                shape=Box(length=1.0, width=1.0),
            )
            world = (
                world.add_agent(source_agent)
                .add_agent(sphere_agent)
                .add_agent(box_agent)
            )
            source_agent, sphere_agent, box_agent = world.agents

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
                name="source",
            )
            sphere_agent = Agent.create(
                name="sphere",
                shape=Sphere(radius=0.5),
            )
            world = world.add_agent(source_agent).add_agent(sphere_agent)
            source_agent, sphere_agent = world.agents
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
                name="source",
            )
            dummy_agent = Agent.create(
                name="dummy",
                shape=DummyShape(),
            )
            world = world.add_agent(source_agent).add_agent(dummy_agent)
            source_agent, dummy_agent = world.agents
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
                name="source",
            )
            world = world.add_agent(source_agent)
            source_agent = source_agent.replace(
                state=source_agent.state.replace(pos=jnp.array([[0.0, 0.0]]))
            )
            world = world.replace(agents=[source_agent])
            (source_agent,) = world.agents
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
                name="source",
            )
            sphere_agent = Agent.create(
                name="sphere",
                shape=Sphere(radius=0.5),
            )
            world = world.add_agent(source_agent).add_agent(sphere_agent)
            source_agent, sphere_agent = world.agents
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
                name="source",
            )
            sphere_agent = Agent.create(
                name="sphere",
                shape=Sphere(radius=0.5),
            )
            world = world.add_agent(source_agent).add_agent(sphere_agent)
            source_agent, sphere_agent = world.agents
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
        sphere_agent = Agent.create(name="sphere", shape=Sphere(radius=0.5))
        world = world.add_agent(sphere_agent)
        (sphere_agent,) = world.agents
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
        dummy_agent = Agent.create(name="dummy", shape=DummyShape())
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
        sphere_agent = Agent.create(name="sphere", shape=Sphere(radius=0.5))
        world = world.add_agent(sphere_agent)
        (sphere_agent,) = world.agents
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
        sphere_a = Agent.create(name="A", shape=Sphere(radius=0.5))
        sphere_b = Agent.create(name="B", shape=Sphere(radius=0.5))
        world = world.add_agent(sphere_a).add_agent(sphere_b)
        sphere_a, sphere_b = world.agents
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
        box = Agent.create(name="Box", shape=Box(length=4.0, width=4.0))
        sphere = Agent.create(name="Sphere", shape=Sphere(radius=0.5))
        world = world.add_agent(box).add_agent(sphere)
        box, sphere = world.agents
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
        box = Agent.create(name="Box", shape=Box(length=4.0, width=4.0))
        sphere = Agent.create(name="Sphere", shape=Sphere(radius=0.5))
        world = world.add_agent(box).add_agent(sphere)
        box, sphere = world.agents
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
        line = Agent.create(name="Line", shape=Line(length=5.0))
        sphere = Agent.create(name="Sphere", shape=Sphere(radius=0.5))
        world = world.add_agent(line).add_agent(sphere)
        line, sphere = world.agents
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
        line_a = Agent.create(name="LineA", shape=Line(length=5.0))
        line_b = Agent.create(name="LineB", shape=Line(length=5.0))
        world = world.add_agent(line_a).add_agent(line_b)
        line_a, line_b = world.agents
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
        box = Agent.create(name="Box", shape=Box(length=4.0, width=4.0))
        line = Agent.create(name="Line", shape=Line(length=5.0))
        world = world.add_agent(box).add_agent(line)
        box, line = world.agents
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
        box_a = Agent.create(name="BoxA", shape=Box(length=4.0, width=4.0))
        box_b = Agent.create(name="BoxB", shape=Box(length=4.0, width=4.0))
        world = world.add_agent(box_a).add_agent(box_b)
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
        a = Agent.create(name="A", shape=Sphere(radius=0.5))
        b = Agent.create(name="B", shape=DummyShape())
        world = world.add_agent(a).add_agent(b)
        a, b = world.agents
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
        sphere_a = Agent.create(name="A", shape=Sphere(radius=0.5))
        sphere_b = Agent.create(name="B", shape=Sphere(radius=0.5))
        world = world.add_agent(sphere_a).add_agent(sphere_b)
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
        sphere_a = Agent.create(name="A", shape=Sphere(radius=0.5))
        sphere_b = Agent.create(name="B", shape=Sphere(radius=0.5))
        world = world.add_agent(sphere_a).add_agent(sphere_b)
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
        box_a = Agent.create(name="A", shape=Box(length=4.0, width=4.0))
        box_b = Agent.create(name="B", shape=Box(length=4.0, width=4.0))
        world = world.add_agent(box_a).add_agent(box_b)
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
        box_a = Agent.create(name="A", shape=Box(length=4.0, width=4.0))
        box_b = Agent.create(name="B", shape=Box(length=4.0, width=4.0))
        world = world.add_agent(box_a).add_agent(box_b)
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
        box = Agent.create(name="Box", shape=Box(length=4.0, width=4.0))
        sphere = Agent.create(name="Sphere", shape=Sphere(radius=0.5))
        world = world.add_agent(box).add_agent(sphere)
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
        box = Agent.create(name="Box", shape=Box(length=4.0, width=4.0))
        sphere = Agent.create(name="Sphere", shape=Sphere(radius=0.5))
        world = world.add_agent(box).add_agent(sphere)
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
        line = Agent.create(name="Line", shape=Line(length=5.0))
        sphere = Agent.create(name="Sphere", shape=Sphere(radius=0.5))
        world = world.add_agent(line).add_agent(sphere)
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
        line = Agent.create(name="Line", shape=Line(length=5.0))
        sphere = Agent.create(name="Sphere", shape=Sphere(radius=0.5))
        world = world.add_agent(line).add_agent(sphere)
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
        line_h = Agent.create(name="Horiz", shape=Line(length=5.0))
        line_v = Agent.create(name="Vert", shape=Line(length=5.0))
        world = world.add_agent(line_h).add_agent(line_v)
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
        line_a = Agent.create(name="A", shape=Line(length=5.0))
        line_b = Agent.create(name="B", shape=Line(length=5.0))
        world = world.add_agent(line_a).add_agent(line_b)
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
        box = Agent.create(name="Box", shape=Box(length=4.0, width=4.0))
        line = Agent.create(name="Line", shape=Line(length=5.0))
        world = world.add_agent(box).add_agent(line)
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
        box = Agent.create(name="Box", shape=Box(length=4.0, width=4.0))
        line = Agent.create(name="Line", shape=Line(length=5.0))
        world = world.add_agent(box).add_agent(line)
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
        agent1 = Agent.create(name="test", movable=True, max_f=2.0)
        agent2 = Agent.create(name="test2", movable=True, f_range=1.5)
        world = world.add_agent(agent1).add_agent(agent2)
        agent1, agent2 = world.agents
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
        agent = Agent.create(name="test", rotatable=True, t_range=1.5)
        agent2 = Agent.create(name="test2", rotatable=True, max_t=2.0)
        world = world.add_agent(agent).add_agent(agent2)
        agent, agent2 = world.agents
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
        agent = Agent.create(name="test", movable=True, mass=2.0)
        agent2 = Agent.create(name="test2", movable=True, mass=2.0)
        world = world.add_agent(agent).add_agent(agent2)
        agent, agent2 = world.agents
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
            name="test",
            movable=True,
            rotatable=True,
            mass=1.0,
            linear_friction=0.5,
            angular_friction=0.3,
            shape=Box(length=1.0, width=1.0),
        )
        world = world.add_agent(agent)
        (agent,) = world.agents
        world = world.reset()
        # Test linear friction
        agent = agent.replace(state=agent.state.replace(vel=jnp.array([[1.0, 0.0]])))
        _, world = world._apply_friction_force(agent)
        assert jnp.all(world.force_dict[agent.name] <= 0)  # Friction opposes motion

        # Test angular friction
        agent = agent.replace(state=agent.state.replace(ang_vel=jnp.array([[1.0]])))
        _, world = world._apply_friction_force(agent)
        assert jnp.all(world.torque_dict[agent.name] <= 0)  # Friction opposes rotation


class TestVectorizedShapeForce:
    def test_update_env_forces(self):
        """Test updating environment forces and torques for entities."""
        world = World.create(batch_dim=1)

        # Create two agents with different movement capabilities
        agent1 = Agent.create(
            batch_dim=1,
            name="agent1",
            dim_p=2,
            dim_c=0,
            movable=True,
            rotatable=True,
        )
        agent2 = Agent.create(
            batch_dim=1,
            name="agent2",
            dim_p=2,
            dim_c=0,
            movable=True,
            rotatable=False,
        )

        world = world.add_agent(agent1).add_agent(agent2)
        world = world.reset()

        # Test force and torque updates
        f_a = jnp.array([[1.0, 1.0]])
        t_a = jnp.array([[0.5]])
        f_b = jnp.array([[2.0, 2.0]])
        t_b = jnp.array([[1.0]])

        updated_world = world.update_env_forces(agent1, f_a, t_a, agent2, f_b, t_b)

        # Check force updates
        assert jnp.allclose(updated_world.force_dict[agent1.name], f_a)
        assert jnp.allclose(updated_world.force_dict[agent2.name], f_b)

        # Check torque updates - agent1 should get torque, agent2 shouldn't
        assert jnp.allclose(updated_world.torque_dict[agent1.name], t_a)
        assert jnp.allclose(updated_world.torque_dict[agent2.name], 0.0)

    def test_vectorized_joint_constraints(self):
        """Test joint constraint forces and torques."""
        world = World.create(batch_dim=1)

        # Create two agents to be joined
        agent1 = Agent.create(
            batch_dim=1,
            name="agent1",
            dim_p=2,
            dim_c=0,
            movable=True,
            rotatable=True,
        )
        agent2 = Agent.create(
            batch_dim=1,
            name="agent2",
            dim_p=2,
            dim_c=0,
            movable=True,
            rotatable=True,
        )

        world = world.add_agent(agent1).add_agent(agent2)
        world = world.reset()

        # Create a joint between agents
        from jaxvmas.simulator.joints import JointConstraint

        joint = JointConstraint.create(
            entity_a=agent1,
            entity_b=agent2,
            anchor_a=(0.0, 0.0),
            anchor_b=(0.0, 0.0),
            dist=1.0,
            rotate=False,
            fixed_rotation=0.0,
        )

        # Test with rotating joint
        world_rotate = world._vectorized_joint_constraints([joint])
        assert len(world_rotate.force_dict) > 0
        assert len(world_rotate.torque_dict) > 0

        # Test with fixed rotation joint
        joint_fixed = joint.replace(rotate=False)
        world_fixed = world._vectorized_joint_constraints([joint_fixed])
        assert len(world_fixed.force_dict) > 0
        assert len(world_fixed.torque_dict) > 0

        # Test with empty joint list
        world_empty = world._vectorized_joint_constraints([])
        assert len(world_empty.force_dict) == len(world.force_dict)

    def test_sphere_sphere_vectorized_collision(self):
        """Test collision forces between spheres."""
        world = World.create(batch_dim=1)

        # Create two sphere agents
        sphere1 = Agent.create(name="sphere1", shape=Sphere(radius=0.5), movable=True)
        sphere2 = Agent.create(name="sphere2", shape=Sphere(radius=0.5), movable=True)

        world = world.add_agent(sphere1).add_agent(sphere2)
        world = world.reset()
        sphere1, sphere2 = world.agents

        # Test collision detection
        collision_mask = jnp.array([True])

        # Test with colliding spheres
        sphere1 = sphere1.replace(
            state=sphere1.state.replace(pos=jnp.array([[0.0, 0.0]]))
        )
        sphere2 = sphere2.replace(
            state=sphere2.state.replace(pos=jnp.array([[0.8, 0.0]]))
        )
        sphere_pairs = [(sphere1, sphere2)]
        world = world.replace(agents=[sphere1, sphere2])

        world_collision = world._sphere_sphere_vectorized_collision(
            sphere_pairs, collision_mask
        )
        assert jnp.any(world_collision.force_dict["sphere1"] != 0)
        assert jnp.any(world_collision.force_dict["sphere2"] != 0)

        # Test with non-colliding spheres
        sphere2 = sphere2.replace(
            state=sphere2.state.replace(pos=jnp.array([[3.0, 0.0]]))
        )
        world = world.replace(agents=[sphere1, sphere2])
        collision_mask = jnp.array([False])

        world_no_collision = world._sphere_sphere_vectorized_collision(
            sphere_pairs, collision_mask
        )
        assert jnp.all(world_no_collision.force_dict["sphere1"] == 0)
        assert jnp.all(world_no_collision.force_dict["sphere2"] == 0)

    def test_sphere_line_vectorized_collision(self):
        """Test collision forces between sphere and line."""
        world = World.create(batch_dim=1)

        # Create sphere and line agents
        sphere = Agent.create(name="sphere", shape=Sphere(radius=0.5), movable=True)
        line = Agent.create(
            name="line", shape=Line(length=2.0), movable=True, rotatable=True
        )

        world = world.add_agent(sphere).add_agent(line)
        world = world.reset()
        sphere, line = world.agents

        # Test collision detection
        collision_mask = jnp.array([True])

        # Test with colliding objects
        sphere = sphere.replace(state=sphere.state.replace(pos=jnp.array([[1.0, 0.5]])))
        line = line.replace(
            state=line.state.replace(
                pos=jnp.array([[0.0, 0.0]]),
                rot=jnp.array([[0.0]]),
            )
        )
        pairs = [(line, sphere)]

        world = world.replace(agents=[sphere, line])

        world_collision = world._sphere_line_vectorized_collision(pairs, collision_mask)
        assert jnp.any(world_collision.force_dict["sphere"] != 0)
        assert jnp.any(world_collision.force_dict["line"] != 0)
        assert jnp.any(world_collision.torque_dict["line"] != 0)

        # Test with non-colliding objects
        sphere = sphere.replace(state=sphere.state.replace(pos=jnp.array([[0.0, 3.0]])))
        world = world.replace(agents=[sphere, line])
        collision_mask = jnp.array([False])

        world_no_collision = world._sphere_line_vectorized_collision(
            pairs, collision_mask
        )
        assert jnp.all(world_no_collision.force_dict["sphere"] == 0)
        assert jnp.all(world_no_collision.force_dict["line"] == 0)
        assert jnp.all(world_no_collision.torque_dict["line"] == 0)

    def test_line_line_vectorized_collision(self):
        """Test collision forces between lines."""
        world = World.create(batch_dim=1)

        # Create two line agents
        line1 = Agent.create(
            name="line1", shape=Line(length=5.0), movable=True, rotatable=True
        )
        line2 = Agent.create(
            name="line2", shape=Line(length=5.0), movable=True, rotatable=True
        )

        world = world.add_agent(line1).add_agent(line2)
        world = world.reset()
        line1, line2 = world.agents
        # Test collision detection
        collision_mask = jnp.array([True])

        # Position lines to create definite intersection
        line1 = line1.replace(
            state=line1.state.replace(
                pos=jnp.array([[0.0, 0.0]]),  # First line at origin
                rot=jnp.array([[0.0]]),  # Horizontal
            )
        )
        line2 = line2.replace(
            state=line2.state.replace(
                pos=jnp.array([[1.0, 1.0]]),  # Larger offset for clear intersection
                rot=jnp.array([[jnp.pi / 2]]),  # Vertical
            )
        )
        world = world.replace(agents=[line1, line2])
        pairs = [(line1, line2)]

        world_collision = world._line_line_vectorized_collision(pairs, collision_mask)

        # Check both force and torque generation
        forces_present = jnp.any(world_collision.force_dict["line1"] != 0) or jnp.any(
            world_collision.force_dict["line2"] != 0
        )
        torques_present = jnp.any(world_collision.torque_dict["line1"] != 0) or jnp.any(
            world_collision.torque_dict["line2"] != 0
        )

        assert forces_present, "No collision forces generated"
        assert torques_present, "No collision torques generated"

        # Test with non-intersecting lines
        line2 = line2.replace(
            state=line2.state.replace(
                pos=jnp.array([[3.0, 3.0]]),
                rot=jnp.array([[jnp.pi / 2]]),
            )
        )
        world = world.replace(agents=[line1, line2])
        collision_mask = jnp.array([False])

        world_no_collision = world._line_line_vectorized_collision(
            pairs, collision_mask
        )
        assert jnp.all(world_no_collision.force_dict["line1"] == 0)
        assert jnp.all(world_no_collision.force_dict["line2"] == 0)
        assert jnp.all(world_no_collision.torque_dict["line1"] == 0)
        assert jnp.all(world_no_collision.torque_dict["line2"] == 0)

    def test_multiple_batch_dimensions(self):
        """Test vectorized collision handling with multiple batch dimensions."""
        world = World.create(batch_dim=2)

        # Create two sphere agents
        sphere1 = Agent.create(name="sphere1", shape=Sphere(radius=0.5), movable=True)
        sphere2 = Agent.create(name="sphere2", shape=Sphere(radius=0.5), movable=True)

        world = world.add_agent(sphere1).add_agent(sphere2)
        world = world.reset()
        sphere1, sphere2 = world.agents

        # Position spheres differently in each batch
        sphere1 = sphere1.replace(
            state=sphere1.state.replace(pos=jnp.array([[0.0, 0.0], [0.0, 0.0]]))
        )
        sphere2 = sphere2.replace(
            state=sphere2.state.replace(pos=jnp.array([[0.8, 0.0], [3.0, 0.0]]))
        )
        world = world.replace(agents=[sphere1, sphere2])

        # Test collision detection
        collision_mask = jnp.array(
            [True, False]
        )  # First batch collides, second doesn't
        sphere_pairs = [(sphere1, sphere2)]

        world_collision = world._sphere_sphere_vectorized_collision(
            sphere_pairs, collision_mask
        )

        # Check that forces are applied only in the first batch
        assert jnp.any(world_collision.force_dict["sphere1"][0] != 0)
        assert jnp.all(world_collision.force_dict["sphere1"][1] == 0)
        assert jnp.any(world_collision.force_dict["sphere2"][0] != 0)
        assert jnp.all(world_collision.force_dict["sphere2"][1] == 0)

    def test_box_sphere_vectorized_collision(self):
        """Test collision forces between box and sphere."""
        world = World.create(batch_dim=1)

        # Create box and sphere agents
        box = Agent.create(
            name="box", shape=Box(length=2.0, width=2.0), movable=True, rotatable=True
        )
        sphere = Agent.create(
            name="sphere",
            shape=Sphere(radius=0.5),
            movable=True,
        )

        world = world.add_agent(box).add_agent(sphere)
        world = world.reset()
        box, sphere = world.agents
        # Test collision detection
        collision_mask = jnp.array([True])

        # Test with colliding objects
        box = box.replace(
            state=box.state.replace(
                pos=jnp.array([[0.0, 0.0]]),
                rot=jnp.array([[0.0]]),
            )
        )
        sphere = sphere.replace(state=sphere.state.replace(pos=jnp.array([[1.0, 0.0]])))
        world = world.replace(agents=[box, sphere])
        pairs = [(box, sphere)]

        world_collision = world._box_sphere_vectorized_collision(pairs, collision_mask)
        assert jnp.any(world_collision.force_dict["box"] != 0)
        assert jnp.any(world_collision.force_dict["sphere"] != 0)
        assert jnp.any(world_collision.torque_dict["box"] != 0)

        # Test with non-colliding objects
        sphere = sphere.replace(state=sphere.state.replace(pos=jnp.array([[3.0, 0.0]])))
        world = world.replace(agents=[box, sphere])
        collision_mask = jnp.array([False])

        world_no_collision = world._box_sphere_vectorized_collision(
            pairs, collision_mask
        )
        assert jnp.all(world_no_collision.force_dict["box"] == 0)
        assert jnp.all(world_no_collision.force_dict["sphere"] == 0)
        assert jnp.all(world_no_collision.torque_dict["box"] == 0)

        # Test with hollow box
        hollow_box = Agent.create(
            name="hollow_box",
            shape=Box(length=2.0, width=2.0, hollow=True),
            movable=True,
            rotatable=True,
        )

        world = World.create(batch_dim=1)
        world = world.add_agent(hollow_box).add_agent(sphere)
        world = world.reset()
        hollow_box, sphere = world.agents

        hollow_box = hollow_box.replace(
            state=hollow_box.state.replace(
                pos=jnp.array([[0.0, 0.0]]),
                rot=jnp.array([[0.0]]),
            )
        )
        sphere = sphere.replace(state=sphere.state.replace(pos=jnp.array([[0.0, 0.0]])))
        world = world.replace(agents=[hollow_box, sphere])
        pairs = [(hollow_box, sphere)]
        collision_mask = jnp.array([True])

        world_hollow = world._box_sphere_vectorized_collision(pairs, collision_mask)
        assert jnp.any(world_hollow.force_dict["hollow_box"] != 0)
        assert jnp.any(world_hollow.force_dict["sphere"] != 0)

    def test_box_line_vectorized_collision(self):
        """Test collision forces between box and line."""
        world = World.create(batch_dim=1)

        # Create box and line agents
        box = Agent.create(
            name="box", shape=Box(length=2.0, width=2.0), movable=True, rotatable=True
        )
        line = Agent.create(
            name="line", shape=Line(length=2.0), movable=True, rotatable=True
        )

        world = world.add_agent(box).add_agent(line)
        world = world.reset()
        box, line = world.agents
        # Test collision detection
        collision_mask = jnp.array([True])

        # Test with intersecting objects
        box = box.replace(
            state=box.state.replace(
                pos=jnp.array([[0.0, 0.0]]),
                rot=jnp.array([[0.0]]),
            )
        )
        line = line.replace(
            state=line.state.replace(
                pos=jnp.array([[1.0, 0.0]]),
                rot=jnp.array([[jnp.pi / 2]]),
            )
        )
        world = world.replace(agents=[box, line])
        pairs = [(box, line)]

        world_collision = world._box_line_vectorized_collision(pairs, collision_mask)
        assert jnp.any(world_collision.force_dict["box"] != 0)
        assert jnp.any(world_collision.force_dict["line"] != 0)
        assert jnp.any(world_collision.torque_dict["box"] != 0)
        assert jnp.any(world_collision.torque_dict["line"] != 0)

        # Test with non-intersecting objects
        line = line.replace(
            state=line.state.replace(
                pos=jnp.array([[3.0, 0.0]]),
                rot=jnp.array([[0.0]]),
            )
        )
        world = world.replace(agents=[box, line])
        collision_mask = jnp.array([False])

        world_no_collision = world._box_line_vectorized_collision(pairs, collision_mask)
        assert jnp.all(world_no_collision.force_dict["box"] == 0)
        assert jnp.all(world_no_collision.force_dict["line"] == 0)
        assert jnp.all(world_no_collision.torque_dict["box"] == 0)
        assert jnp.all(world_no_collision.torque_dict["line"] == 0)

        # Test with hollow box
        hollow_box = Agent.create(
            name="hollow_box",
            shape=Box(length=2.0, width=2.0, hollow=True),
            movable=True,
            rotatable=True,
        )
        world = World.create(batch_dim=1)
        world = world.add_agent(hollow_box).add_agent(line)
        world = world.reset()
        hollow_box, line = world.agents

        hollow_box = hollow_box.replace(
            state=hollow_box.state.replace(
                pos=jnp.array([[0.0, 0.0]]),
                rot=jnp.array([[0.0]]),
            )
        )
        line = line.replace(
            state=line.state.replace(
                pos=jnp.array([[0.0, 0.0]]),
                rot=jnp.array([[0.0]]),
            )
        )
        world = world.replace(agents=[hollow_box, line])
        pairs = [(hollow_box, line)]
        collision_mask = jnp.array([True])

        world_hollow = world._box_line_vectorized_collision(pairs, collision_mask)
        assert jnp.any(world_hollow.force_dict["hollow_box"] != 0)
        assert jnp.any(world_hollow.force_dict["line"] != 0)

    def test_box_box_vectorized_collision(self):
        """Test collision forces between two boxes."""
        world = World.create(batch_dim=1)

        # Create two box agents
        box1 = Agent.create(
            name="box1", shape=Box(length=2.0, width=2.0), movable=True, rotatable=True
        )
        box2 = Agent.create(
            name="box2", shape=Box(length=2.0, width=2.0), movable=True, rotatable=True
        )

        world = world.add_agent(box1).add_agent(box2)
        world = world.reset()
        box1, box2 = world.agents

        # Test collision detection
        collision_mask = jnp.array([True])

        # Test with overlapping boxes
        box1 = box1.replace(
            state=box1.state.replace(
                pos=jnp.array([[0.0, 0.0]]),
                rot=jnp.array([[0.0]]),
            )
        )
        box2 = box2.replace(
            state=box2.state.replace(
                pos=jnp.array([[1.5, 0.0]]),
                rot=jnp.array([[jnp.pi / 4]]),
            )
        )
        world = world.replace(agents=[box1, box2])
        pairs = [(box1, box2)]

        world_collision = world._box_box_vectorized_collision(pairs, collision_mask)
        assert jnp.any(world_collision.force_dict["box1"] != 0)
        assert jnp.any(world_collision.force_dict["box2"] != 0)
        assert jnp.any(world_collision.torque_dict["box1"] != 0)
        assert jnp.any(world_collision.torque_dict["box2"] != 0)

        # Test with non-overlapping boxes
        box2 = box2.replace(
            state=box2.state.replace(
                pos=jnp.array([[4.0, 0.0]]),
                rot=jnp.array([[0.0]]),
            )
        )
        world = world.replace(agents=[box1, box2])
        collision_mask = jnp.array([False])

        world_no_collision = world._box_box_vectorized_collision(pairs, collision_mask)
        assert jnp.all(world_no_collision.force_dict["box1"] == 0)
        assert jnp.all(world_no_collision.force_dict["box2"] == 0)
        assert jnp.all(world_no_collision.torque_dict["box1"] == 0)
        assert jnp.all(world_no_collision.torque_dict["box2"] == 0)

        # Test with one hollow box and one solid box
        hollow_box = Agent.create(
            name="hollow_box",
            shape=Box(length=2.0, width=2.0, hollow=True),
            movable=True,
            rotatable=True,
        )
        world = World.create(batch_dim=1)
        world = world.add_agent(hollow_box).add_agent(box2)
        world = world.reset()
        hollow_box, box2 = world.agents

        hollow_box = hollow_box.replace(
            state=hollow_box.state.replace(
                pos=jnp.array([[0.0, 0.0]]),
                rot=jnp.array([[0.0]]),
            )
        )
        box2 = box2.replace(
            state=box2.state.replace(
                pos=jnp.array([[1.0, 0.0]]),
                rot=jnp.array([[0.0]]),
            )
        )
        world = world.replace(agents=[hollow_box, box2])
        pairs = [(hollow_box, box2)]
        collision_mask = jnp.array([True])

        world_hollow = world._box_box_vectorized_collision(pairs, collision_mask)
        assert jnp.any(world_hollow.force_dict["hollow_box"] != 0)
        assert jnp.any(world_hollow.force_dict["box2"] != 0)

    def test_box_collisions_batch_dimensions(self):
        """Test box collisions with multiple batch dimensions."""
        world = World.create(batch_dim=2)

        # Create box and sphere for testing
        box = Agent.create(
            name="box", shape=Box(length=2.0, width=2.0), movable=True, rotatable=True
        )
        sphere = Agent.create(name="sphere", shape=Sphere(radius=0.5), movable=True)

        world = world.add_agent(box).add_agent(sphere)
        world = world.reset()
        box, sphere = world.agents
        # Position differently in each batch - first batch collides, second doesn't
        box = box.replace(
            state=box.state.replace(
                pos=jnp.array([[0.0, 0.0], [0.0, 0.0]]),
                rot=jnp.array([[0.0], [0.0]]),
            )
        )
        sphere = sphere.replace(
            state=sphere.state.replace(pos=jnp.array([[1.0, 0.0], [3.0, 0.0]]))
        )
        world = world.replace(agents=[box, sphere])

        collision_mask = jnp.array([True, False])
        pairs = [(box, sphere)]

        world_collision = world._box_sphere_vectorized_collision(pairs, collision_mask)

        # Check first batch has collision forces, second doesn't
        assert jnp.any(world_collision.force_dict["box"][0] != 0)
        assert jnp.all(world_collision.force_dict["box"][1] == 0)
        assert jnp.any(world_collision.force_dict["sphere"][0] != 0)
        assert jnp.all(world_collision.force_dict["sphere"][1] == 0)


class TestVectorizedEnvironmentForce:
    def test_vectorized_environment_force(self):
        """Test vectorized environment force application for all shape combinations."""

        # Create entities with different shapes
        sphere1 = Agent.create(name="sphere1", shape=Sphere(radius=0.5), movable=True)
        sphere2 = Agent.create(name="sphere2", shape=Sphere(radius=0.5), movable=True)
        line1 = Agent.create(name="line1", shape=Line(length=2.0), movable=True)
        line2 = Agent.create(name="line2", shape=Line(length=2.0), movable=True)
        box1 = Agent.create(name="box1", shape=Box(length=1.0, width=1.0), movable=True)
        box2 = Agent.create(name="box2", shape=Box(length=1.0, width=1.0), movable=True)

        # Test sphere-sphere collision
        world = World.create(batch_dim=1)
        world = world.add_agent(sphere1).add_agent(sphere2)
        sphere1, sphere2 = world.agents
        sphere1 = sphere1.replace(
            state=sphere1.state.replace(pos=jnp.array([[0.0, 0.0]]))
        )
        sphere2 = sphere2.replace(
            state=sphere2.state.replace(pos=jnp.array([[0.8, 0.0]]))
        )
        world = world.replace(agents=[sphere1, sphere2])
        world = world._apply_vectorized_enviornment_force()
        assert len(world.force_dict) > 0  # Forces should be applied due to collision

        # Test line-sphere collision
        world = World.create(batch_dim=1)
        world = world.add_agent(line1).add_agent(sphere1)
        line1, sphere1 = world.agents
        line1 = line1.replace(
            state=line1.state.replace(
                pos=jnp.array([[0.0, 0.0]]),
                rot=jnp.array([[0.0]]),
            )
        )
        sphere1 = sphere1.replace(
            state=sphere1.state.replace(pos=jnp.array([[0.5, 0.0]]))
        )
        world = world.replace(agents=[line1, sphere1])
        world = world._apply_vectorized_enviornment_force()
        assert len(world.force_dict) > 0

        # Test line-line collision
        world = World.create(batch_dim=1)
        world = world.add_agent(line1).add_agent(line2)
        line1, line2 = world.agents
        line1 = line1.replace(
            state=line1.state.replace(
                pos=jnp.array([[0.0, 0.0]]),
                rot=jnp.array([[0.0]]),
            )
        )
        line2 = line2.replace(
            state=line2.state.replace(
                pos=jnp.array([[0.0, 0.0]]),
                rot=jnp.array([[jnp.pi / 2]]),
            )
        )
        world = world.replace(agents=[line1, line2])
        world = world._apply_vectorized_enviornment_force()
        assert len(world.force_dict) > 0

        # Test box-sphere collision
        world = World.create(batch_dim=1)
        world = world.add_agent(box1).add_agent(sphere1)
        box1, sphere1 = world.agents
        box1 = box1.replace(
            state=box1.state.replace(
                pos=jnp.array([[0.0, 0.0]]),
                rot=jnp.array([[0.0]]),
            )
        )
        sphere1 = sphere1.replace(
            state=sphere1.state.replace(pos=jnp.array([[0.5, 0.0]]))
        )
        world = world.replace(agents=[box1, sphere1])
        world = world._apply_vectorized_enviornment_force()
        assert len(world.force_dict) > 0

        # Test box-line collision
        world = World.create(batch_dim=1)
        world = world.add_agent(box1).add_agent(line1)
        box1, line1 = world.agents
        box1 = box1.replace(
            state=box1.state.replace(
                pos=jnp.array([[0.0, 0.0]]),
                rot=jnp.array([[0.0]]),
            )
        )
        line1 = line1.replace(
            state=line1.state.replace(
                pos=jnp.array([[0.0, 0.0]]),
                rot=jnp.array([[0.0]]),
            )
        )
        world = world.replace(agents=[box1, line1])
        world = world._apply_vectorized_enviornment_force()
        assert len(world.force_dict) > 0

        # Test box-box collision
        world = World.create(batch_dim=1)
        world = world.add_agent(box1).add_agent(box2)
        box1, box2 = world.agents
        box1 = box1.replace(
            state=box1.state.replace(
                pos=jnp.array([[0.0, 0.0]]),
                rot=jnp.array([[0.0]]),
            )
        )
        box2 = box2.replace(
            state=box2.state.replace(
                pos=jnp.array([[0.8, 0.0]]),
                rot=jnp.array([[0.0]]),
            )
        )
        world = world.replace(agents=[box1, box2])
        world = world._apply_vectorized_enviornment_force()
        assert len(world.force_dict) > 0

    def test_vectorized_environment_force_with_joints(self):
        """Test vectorized environment force application with joint constraints."""
        world = World.create(batch_dim=1)

        # Create two agents to be joined
        agent1 = Agent.create(
            batch_dim=1,
            name="agent1",
            dim_p=2,
            dim_c=0,
            shape=Sphere(radius=0.5),
            movable=True,
        )
        agent2 = Agent.create(
            batch_dim=1,
            name="agent2",
            dim_p=2,
            dim_c=0,
            shape=Sphere(radius=0.5),
            movable=True,
        )

        # Add joint between agents
        from jaxvmas.simulator.joints import Joint

        world = world.add_agent(agent1).add_agent(agent2)
        joint = Joint.create(
            batch_dim=1,
            entity_a=agent1,
            entity_b=agent2,
            anchor_a=(0.0, 0.0),
            anchor_b=(0.0, 0.0),
            dist=1.0,
        )
        world = world.add_joint(joint)

        # Position agents at distance > joint distance
        agent1 = agent1.replace(state=agent1.state.replace(pos=jnp.array([[0.0, 0.0]])))
        agent2 = agent2.replace(state=agent2.state.replace(pos=jnp.array([[2.0, 0.0]])))
        world = world.replace(agents=[agent1, agent2])

        # Apply forces
        world = world._apply_vectorized_enviornment_force()
        assert len(world.force_dict) > 0  # Joint forces should be applied

    def test_vectorized_environment_force_multiple_collisions(self):
        """Test handling of multiple simultaneous collisions."""
        world = World.create(batch_dim=1)

        # Create multiple spheres that will collide
        sphere1 = Agent.create(name="sphere1", shape=Sphere(radius=0.5), movable=True)
        sphere2 = Agent.create(name="sphere2", shape=Sphere(radius=0.5), movable=True)
        sphere3 = Agent.create(name="sphere3", shape=Sphere(radius=0.5), movable=True)

        # Position spheres to create multiple collisions
        world = world.add_agent(sphere1).add_agent(sphere2).add_agent(sphere3)
        sphere1, sphere2, sphere3 = world.agents
        sphere1 = sphere1.replace(
            state=sphere1.state.replace(pos=jnp.array([[0.0, 0.0]]))
        )
        sphere2 = sphere2.replace(
            state=sphere2.state.replace(pos=jnp.array([[0.8, 0.0]]))
        )
        sphere3 = sphere3.replace(
            state=sphere3.state.replace(pos=jnp.array([[0.0, 0.8]]))
        )
        world = world.replace(agents=[sphere1, sphere2, sphere3])

        # Apply forces
        world = world._apply_vectorized_enviornment_force()
        assert len(world.force_dict) == 3  # Forces should be applied to all agents

    def test_vectorized_environment_force_invalid_shape(self):
        """Test handling of invalid shape combinations."""
        world = World.create(batch_dim=1)

        # Create agents with valid and invalid shapes
        valid_agent = Agent.create(name="valid", shape=Sphere(radius=0.5), movable=True)
        invalid_agent = Agent.create(name="invalid", shape=DummyShape(), movable=True)

        world = world.add_agent(valid_agent).add_agent(invalid_agent)
        valid_agent, invalid_agent = world.agents
        valid_agent = valid_agent.replace(
            state=valid_agent.state.replace(pos=jnp.array([[0.0, 0.0]]))
        )
        invalid_agent = invalid_agent.replace(
            state=invalid_agent.state.replace(pos=jnp.array([[1.0, 0.0]]))
        )
        world = world.replace(agents=[valid_agent, invalid_agent])

        with pytest.raises(AssertionError):
            world._apply_vectorized_enviornment_force()

    def test_vectorized_environment_force_batch_dim(self):
        """Test vectorized environment force with multiple batch dimensions."""
        world = World.create(batch_dim=2)

        # Create two spheres
        sphere1 = Agent.create(name="sphere1", shape=Sphere(radius=0.5), movable=True)
        sphere2 = Agent.create(name="sphere2", shape=Sphere(radius=0.5), movable=True)

        # Position spheres differently in each batch
        world = world.add_agent(sphere1).add_agent(sphere2)
        sphere1, sphere2 = world.agents
        sphere1 = sphere1.replace(
            state=sphere1.state.replace(pos=jnp.array([[0.0, 0.0], [0.0, 0.0]]))
        )
        sphere2 = sphere2.replace(
            state=sphere2.state.replace(pos=jnp.array([[0.8, 0.0], [2.0, 0.0]]))
        )
        world = world.replace(agents=[sphere1, sphere2])

        # Apply forces
        world = world._apply_vectorized_enviornment_force()
        # First batch should have collision forces, second batch should not
        assert jnp.any(world.force_dict["sphere1"][0] != 0)
        assert jnp.all(world.force_dict["sphere1"][1] == 0)

    def test_vectorized_environment_force_empty_world(self):
        """Test vectorized environment force with no entities."""
        world = World.create(batch_dim=1)
        world = world._apply_vectorized_enviornment_force()
        assert len(world.force_dict) == 0

    def test_vectorized_environment_force_single_entity(self):
        """Test vectorized environment force with a single entity."""
        world = World.create(batch_dim=1)

        sphere = Agent.create(name="sphere", shape=Sphere(radius=0.5), movable=True)

        world = world.add_agent(sphere)
        sphere = sphere.replace(state=sphere.state.replace(pos=jnp.array([[0.0, 0.0]])))
        world = world.replace(agents=[sphere])
        sphere = world.agents[0]

        world = world._apply_vectorized_enviornment_force()
        assert len(world.force_dict) == 0  # No forces should be applied


class TestMoreTests:
    def test_collides(self):
        """Test collision detection between entities."""
        world = World.create(batch_dim=1)

        # Create test agents
        sphere1 = Agent.create(name="sphere1", shape=Sphere(radius=0.5), movable=True)
        sphere2 = Agent.create(name="sphere2", shape=Sphere(radius=0.5), movable=True)
        world = world.add_agent(sphere1).add_agent(sphere2)
        sphere1, sphere2 = world.agents

        # Test same entity (should not collide)
        assert not world.collides(sphere1, sphere1)

        # Test overlapping entities
        sphere1 = sphere1.replace(
            state=sphere1.state.replace(pos=jnp.array([[0.0, 0.0]]))
        )
        sphere2 = sphere2.replace(
            state=sphere2.state.replace(pos=jnp.array([[0.8, 0.0]]))
        )
        assert world.collides(sphere1, sphere2)

        # Test non-overlapping entities
        sphere2 = sphere2.replace(
            state=sphere2.state.replace(pos=jnp.array([[3.0, 0.0]]))
        )
        assert not world.collides(sphere1, sphere2)
        world = world.replace(agents=[])
        # Test with non-movable entities
        static_sphere = Agent.create(
            name="static", shape=Sphere(radius=0.5), movable=False
        )
        moving_sphere = Agent.create(
            name="moving", shape=Sphere(radius=0.5), movable=True
        )
        world = world.add_agent(static_sphere).add_agent(moving_sphere)
        static_sphere, moving_sphere = world.agents
        static_sphere = static_sphere.replace(
            movable=True,
        )
        static_sphere = static_sphere.replace(
            state=static_sphere.state.replace(pos=jnp.array([[0.0, 0.0]]))
        )
        moving_sphere = moving_sphere.replace(
            state=moving_sphere.state.replace(pos=jnp.array([[0.8, 0.0]]))
        )
        assert world.collides(static_sphere, moving_sphere)

    def test_get_constraint_forces(self):
        """Test constraint force calculations."""
        world = World.create(batch_dim=1)

        # Test repulsive forces (default)
        pos_a = jnp.array([[0.0, 0.0]])
        pos_b = jnp.array([[0.5, 0.0]])
        dist_min = 0.5
        force_multiplier = 1.0

        force_a, force_b = world._get_constraint_forces(
            pos_a, pos_b, dist_min, force_multiplier
        )

        # Forces should be equal and opposite
        assert jnp.allclose(force_a, -force_b)
        # Force should point along the line connecting the points
        assert jnp.allclose(force_a[0, 1], 0.0)  # No y-component
        assert force_a[0, 0] < 0  # x-component should be negative (repulsive)

        # Test attractive forces
        force_a, force_b = world._get_constraint_forces(
            pos_a, pos_b, dist_min, force_multiplier, attractive=True
        )
        assert force_a[0, 0] > 0  # x-component should be positive (attractive)

        # Test zero distance case
        pos_b = jnp.array([[0.0, 0.0]])
        force_a, force_b = world._get_constraint_forces(
            pos_a, pos_b, dist_min, force_multiplier
        )
        assert jnp.allclose(force_a, 0.0)
        assert jnp.allclose(force_b, 0.0)

        # Test large distance case (beyond dist_min)
        pos_b = jnp.array([[2.0, 0.0]])
        force_a, force_b = world._get_constraint_forces(
            pos_a, pos_b, dist_min, force_multiplier
        )
        assert jnp.allclose(force_a, 0.0)
        assert jnp.allclose(force_b, 0.0)

    def test_get_constraint_torques(self):
        """Test constraint torque calculations."""
        world = World.create(batch_dim=1)

        # Test aligned rotations
        rot_a = jnp.array([[0.0]])
        rot_b = jnp.array([[0.0]])
        torque_a, torque_b = world._get_constraint_torques(rot_a, rot_b)
        assert jnp.allclose(torque_a, 0.0)
        assert jnp.allclose(torque_b, 0.0)

        # Test misaligned rotations
        rot_b = jnp.array([[jnp.pi / 4]])
        torque_a, torque_b = world._get_constraint_torques(rot_a, rot_b)
        assert jnp.allclose(torque_a, -torque_b)  # Equal and opposite torques
        assert torque_a[0, 0] > 0  # Positive torque to align with rot_b

        # Test with custom force multiplier
        force_multiplier = 2.0
        torque_a, torque_b = world._get_constraint_torques(
            rot_a, rot_b, force_multiplier
        )
        assert jnp.abs(torque_a) > jnp.abs(
            world._get_constraint_torques(rot_a, rot_b)[0]
        )

        # Test small angle difference
        rot_b = jnp.array([[1e-10]])
        torque_a, torque_b = world._get_constraint_torques(rot_a, rot_b)
        assert jnp.allclose(torque_a, 0.0)
        assert jnp.allclose(torque_b, 0.0)

    def test_integrate_state(self):
        """Test physical state integration."""
        world = World.create(batch_dim=1, drag=0.0)

        # Test translation
        agent = Agent.create(
            name="agent", shape=Sphere(radius=0.5), movable=True, mass=1.0
        )
        world = world.add_agent(agent)
        world = world.reset()
        agent = world.agents[-1]

        # Set initial conditions
        agent = agent.replace(
            state=agent.state.replace(
                pos=jnp.array([[0.0, 0.0]]),
                vel=jnp.array([[1.0, 1.0]]),
            )
        )
        world = world.replace(force_dict={"agent": jnp.array([[1.0, 1.0]])})

        # Test integration
        updated_agent = world._integrate_state(agent, substep=0)
        assert jnp.all(
            updated_agent.state.pos > agent.state.pos
        )  # Position should increase
        assert jnp.all(
            updated_agent.state.vel > agent.state.vel
        )  # Velocity should increase

        # Test rotation
        rotating_agent = Agent.create(
            name="rotating",
            shape=Box(length=1.0, width=1.0),
            movable=True,
            rotatable=True,
        )
        world = world.add_agent(rotating_agent)
        world = world.reset()
        rotating_agent = world.agents[-1]

        rotating_agent = rotating_agent.replace(
            state=rotating_agent.state.replace(
                rot=jnp.array([[0.0]]),
                ang_vel=jnp.array([[1.0]]),
            )
        )
        world = world.replace(torque_dict={"rotating": jnp.array([[1.0]])})

        updated_rotating = world._integrate_state(rotating_agent, substep=0)
        assert jnp.all(updated_rotating.state.rot > rotating_agent.state.rot)
        assert jnp.all(updated_rotating.state.ang_vel > rotating_agent.state.ang_vel)

        # Test boundary conditions
        world = World.create(batch_dim=1, x_semidim=1.0, y_semidim=1.0)
        agent = Agent.create(name="agent", shape=Sphere(radius=0.5), movable=True)
        world = world.add_agent(agent)
        world = world.reset()
        agent = world.agents[-1]
        agent = agent.replace(
            state=agent.state.replace(
                pos=jnp.array([[0.0, 0.0]]),
                vel=jnp.array([[2.0, 2.0]]),
            )
        )
        updated_agent = world._integrate_state(agent, substep=0)
        assert jnp.all(
            jnp.abs(updated_agent.state.pos) <= 1.0
        )  # Position should be clamped

        # Test velocity limits
        agent_with_limits = agent.replace(
            max_speed=1.0,
            v_range=0.5,
        )
        updated_agent = world._integrate_state(agent_with_limits, substep=0)
        assert jnp.all(
            jnp.abs(updated_agent.state.vel) <= 0.5
        )  # Velocity should be clamped

    def test_update_comm_state(self):
        """Test communication state updates."""
        world = World.create(batch_dim=1)

        # Test silent agent
        silent_agent = Agent.create(name="silent", silent=True)
        world = world.add_agent(silent_agent)
        silent_agent = world.agents[-1]
        silent_agent = silent_agent.replace(
            action=Action.create(
                batch_dim=1,
                action_size=2,
                comm_dim=2,
                u_range=1.0,
                u_multiplier=1.0,
                u_noise=0.0,
            ).replace(c=jnp.array([[1.0, 2.0]]))
        )
        updated_silent = world._update_comm_state(silent_agent)
        assert jnp.allclose(
            updated_silent.state.c, silent_agent.state.c
        )  # Should not change

        world = World.create(batch_dim=1, dim_p=2, dim_c=2)
        # Test communicating agent
        comm_agent = Agent.create(name="comm", silent=False)
        world = world.add_agent(comm_agent)
        comm_agent = world.agents[-1]
        comm_agent = comm_agent.replace(
            action=Action.create(
                batch_dim=1,
                action_size=2,
                comm_dim=2,
                u_range=1.0,
                u_multiplier=1.0,
                u_noise=0.0,
            ).replace(c=jnp.array([[1.0, 2.0]]))
        )
        updated_comm = world._update_comm_state(comm_agent)
        assert jnp.allclose(
            updated_comm.state.c, comm_agent.action.c
        )  # Should update to action
