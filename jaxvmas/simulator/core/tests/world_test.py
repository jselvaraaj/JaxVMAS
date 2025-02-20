import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from jaxvmas.simulator.core.agent import Agent
from jaxvmas.simulator.core.entity import Entity
from jaxvmas.simulator.core.landmark import Landmark
from jaxvmas.simulator.core.shapes import Box, Sphere
from jaxvmas.simulator.core.world import (
    DRAG,
    World,
)


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
