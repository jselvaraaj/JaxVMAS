import math

import jax.numpy as jnp
import pytest

from jaxvmas.simulator.core.entity import Entity
from jaxvmas.simulator.core.ray_physics import (
    cast_ray_to_box,
    cast_ray_to_line,
    cast_ray_to_sphere,
    cast_rays_to_box,
    cast_rays_to_line,
    cast_rays_to_sphere,
)
from jaxvmas.simulator.core.shapes import Box, Line, Sphere

# Constants for testing
MAX_RANGE = 10.0
EPSILON = 1e-5


@pytest.fixture
def box_entity():
    entity = Entity.create(shape=Box(length=2.0, width=1.0), name="box")
    entity = entity._spawn(id=jnp.asarray(2), batch_dim=1, dim_p=2)

    return entity


@pytest.fixture
def sphere_entity():
    entity = Entity.create(shape=Sphere(radius=1.0), name="sphere")
    entity = entity._spawn(id=jnp.asarray(2), batch_dim=1, dim_p=2)
    return entity


@pytest.fixture
def line_entity():
    entity = Entity.create(shape=Line(length=2.0), name="line")
    entity = entity._spawn(id=jnp.asarray(2), batch_dim=1, dim_p=2)

    return entity


def test_cast_ray_to_box_direct_hit(box_entity):
    ray_origin = jnp.array([[2.0, 0.0]])  # Ray from right side
    ray_direction = jnp.array([jnp.pi])  # Pointing left

    dist = cast_ray_to_box(box_entity, ray_origin, ray_direction, MAX_RANGE)
    expected_dist = 1.0  # Distance to box edge
    assert jnp.abs(dist[0] - expected_dist) < EPSILON


def test_cast_ray_to_box_miss(box_entity):
    ray_origin = jnp.array([[2.0, 2.0]])  # Ray from top-right
    ray_direction = jnp.array([jnp.pi])  # Pointing left

    dist = cast_ray_to_box(box_entity, ray_origin, ray_direction, MAX_RANGE)
    assert jnp.all(dist == MAX_RANGE)


def test_cast_rays_to_box_batch():
    batch_dim = 2
    box_pos = jnp.array([[[0.0, 0.0]], [[1.0, 1.0]]])
    box_rot = jnp.array([[0.0], [0.0]])
    box_length = jnp.array([[2.0], [2.0]])
    box_width = jnp.array([[1.0], [1.0]])
    ray_origin = jnp.array([[2.0, 0.0], [3.0, 1.0]])
    ray_direction = jnp.array([[jnp.pi], [jnp.pi]])

    dist = cast_rays_to_box(
        batch_dim,
        box_pos,
        box_rot,
        box_length,
        box_width,
        ray_origin,
        ray_direction,
        MAX_RANGE,
    )
    assert dist.shape == (2, 1, 1)  # [batch_size, n_boxes, n_angles]


def test_cast_ray_to_sphere_direct_hit(sphere_entity):
    ray_origin = jnp.array([[2.0, 0.0]])
    ray_direction = jnp.array([jnp.pi])

    dist = cast_ray_to_sphere(sphere_entity, ray_origin, ray_direction, MAX_RANGE)
    expected_dist = 1.0  # Distance to sphere edge
    assert jnp.abs(dist[0] - expected_dist) < EPSILON


def test_cast_ray_to_sphere_miss(sphere_entity):
    ray_origin = jnp.array([[2.0, 2.0]])
    ray_direction = jnp.array([jnp.pi])

    dist = cast_ray_to_sphere(sphere_entity, ray_origin, ray_direction, MAX_RANGE)
    assert jnp.all(dist == MAX_RANGE)


def test_cast_rays_to_sphere_batch():
    batch_dim = 2
    sphere_pos = jnp.array([[[0.0, 0.0]], [[1.0, 1.0]]])
    sphere_radius = jnp.array([[1.0], [1.0]])
    ray_origin = jnp.array([[2.0, 0.0], [3.0, 1.0]])
    ray_direction = jnp.array([[jnp.pi], [jnp.pi]])

    dist = cast_rays_to_sphere(
        batch_dim, sphere_pos, sphere_radius, ray_origin, ray_direction, MAX_RANGE
    )
    assert dist.shape == (2, 1, 1)


def test_cast_ray_to_line_direct_hit(line_entity):
    ray_origin = jnp.array([[2.0, 0.0]])  # Ray from right side
    ray_direction = jnp.array([jnp.pi])  # Pointing left
    line_entity = line_entity.set_pos(ray_origin)
    dist = cast_ray_to_line(line_entity, ray_origin, ray_direction, MAX_RANGE)
    expected_dist = 0.0
    assert jnp.abs(dist[0] - expected_dist) < EPSILON


def test_cast_ray_to_line_parallel(line_entity: Entity):
    # Test horizontal parallel lines
    ray_origin = jnp.array([[2.0, 0.0]])  # Ray from right side
    ray_direction = jnp.array([0.0])  # Pointing horizontally
    dist = cast_ray_to_line(line_entity, ray_origin, ray_direction, MAX_RANGE)
    assert jnp.all(dist == MAX_RANGE)

    # Test parallel lines at 45 degrees
    line_entity = line_entity.set_rot(jnp.array([jnp.pi / 4]))
    ray_origin = jnp.array([[2.0, 2.0]])
    ray_direction = jnp.array([jnp.pi / 4])
    dist = cast_ray_to_line(line_entity, ray_origin, ray_direction, MAX_RANGE)
    assert jnp.all(dist == MAX_RANGE)

    # Test parallel but offset lines
    line_entity = line_entity.set_rot(jnp.array([0.0]))
    ray_origin = jnp.array([[2.0, 1.0]])  # Offset by 1 unit vertically
    ray_direction = jnp.array([0.0])
    dist = cast_ray_to_line(line_entity, ray_origin, ray_direction, MAX_RANGE)
    assert jnp.all(dist == MAX_RANGE)


def test_cast_rays_to_line_batch():
    batch_dim = 2
    line_pos = jnp.array([[[0.0, 0.0]], [[1.0, 1.0]]])
    line_rot = jnp.array([[0.0], [0.0]])
    line_length = jnp.array([[2.0], [2.0]])
    ray_origin = jnp.array([[2.0, 0.0], [3.0, 1.0]])
    ray_direction = jnp.array([[jnp.pi], [jnp.pi]])

    dist = cast_rays_to_line(
        batch_dim, line_pos, line_rot, line_length, ray_origin, ray_direction, MAX_RANGE
    )
    assert dist.shape == (2, 1, 1)


def test_edge_cases():
    # Test ray origin inside objects
    box = Entity.create(shape=Box(length=2.0, width=1.0), name="box")
    box = box._spawn(id=jnp.asarray(2), batch_dim=1, dim_p=2)

    ray_origin = jnp.array([[0.0, 0.0]])  # Origin inside box
    ray_direction = jnp.array([0.0])

    dist = cast_ray_to_box(box, ray_origin, ray_direction, MAX_RANGE)
    assert jnp.all(dist == MAX_RANGE)


def test_rotated_objects():
    # Test with rotated box
    box = Entity.create(shape=Box(length=2.0, width=1.0), name="box")
    box = box._spawn(id=jnp.asarray(2), batch_dim=1, dim_p=2)

    ray_origin = jnp.array([[2.0, 0.0]])
    ray_direction = jnp.array([jnp.pi])

    dist = cast_ray_to_box(box, ray_origin, ray_direction, MAX_RANGE)
    assert dist.shape == (1,)


def test_cast_ray_to_line_comprehensive(line_entity: Entity):
    """
    Comprehensive tests for cast_ray_to_line.

    **Assumptions (per the source code):**
      - line.state.pos is the *starting endpoint* of the segment.
      - line.state.rot is the orientation (in radians).
      - line.shape.length is the full length.
      - The segment spans from pos to pos + ([cos(rot), sin(rot)] * length).
    """

    # Ensure the line has a known length (2.0) for these tests.
    # (Assumes you can replace the shape via the replace method.)
    line_entity = line_entity.replace(shape=Line(length=2.0))

    # ---------------------------
    # Test 6: Rotated line test
    # ---------------------------
    # Set the line's starting endpoint to (0,0) and rotate it 45° (pi/4).
    # Then the line spans from (0,0) to:
    #   (cos(pi/4)*2, sin(pi/4)*2) = (√2, √2) ≈ (1.414, 1.414)
    line_entity = line_entity.set_pos(jnp.array([[0.0, 0.0]]))
    line_entity = line_entity.set_rot(jnp.array([[math.pi / 4]]))

    # Ray from (1, -1) pointing up-left (direction = 3π/4).
    # With this configuration, the algorithm finds the intersection at (0,0).
    # Thus, the ray travels from (1,-1) to (0,0): a distance of √((1)^2+(1)^2)=√2.
    expected = math.sqrt(2)
    ray_origin = jnp.array([[1.0, -1.0]])
    ray_direction = jnp.array([3 * math.pi / 4])
    dist = cast_ray_to_line(line_entity, ray_origin, ray_direction, MAX_RANGE)
    assert (
        jnp.abs(dist[0] - expected) < EPSILON
    ), f"Rotated line hit failed: expected {expected}, got {dist[0]}"

    # ---------------------------
    # Test 7: Exact endpoint hit
    # ---------------------------
    # Reset to a horizontal line (rot = 0), which spans from (0,0) to (2,0).
    # A ray from (2,1) with direction 225° (5π/4) should hit the line at (1,0).
    # The distance from (2,1) to (1,0) is √((2-1)² + (1-0)²)=√2.
    line_entity = line_entity.set_rot(jnp.array([[0.0]]))
    ray_origin = jnp.array([[2.0, 1.0]])
    ray_direction = jnp.array([5 * math.pi / 4])
    dist = cast_ray_to_line(line_entity, ray_origin, ray_direction, MAX_RANGE)
    expected = math.sqrt(2)
    assert (
        jnp.abs(dist[0] - expected) < EPSILON
    ), f"Exact endpoint hit failed: expected {expected}, got {dist[0]}"

    # ---------------------------
    # Test 8: Perpendicular ray
    # ---------------------------
    # For the same horizontal line (from (0,0) to (2,0)):
    # A ray from (0,1) pointing straight down (3π/2) should hit at (0,0).
    # The distance from (0,1) to (0,0) is 1.
    ray_origin = jnp.array([[0.0, 1.0]])
    ray_direction = jnp.array([3 * math.pi / 2])
    dist = cast_ray_to_line(line_entity, ray_origin, ray_direction, MAX_RANGE)
    expected = 1.0
    assert (
        jnp.abs(dist[0] - expected) < EPSILON
    ), f"Perpendicular ray failed: expected {expected}, got {dist[0]}"

    # ---------------------------
    # Test 9: Glancing ray
    # ---------------------------
    # For a horizontal line from (0,0) to (2,0):
    # A ray from (2,0) pointing left (π) is collinear with the line direction.
    # The algorithm treats such glancing intersections as invalid and returns MAX_RANGE.
    ray_origin = jnp.array([[2.0, 0.0]])
    ray_direction = jnp.array([math.pi])
    dist = cast_ray_to_line(line_entity, ray_origin, ray_direction, MAX_RANGE)
    expected = MAX_RANGE
    assert (
        jnp.abs(dist[0] - expected) < EPSILON
    ), f"Glancing ray failed: expected {expected}, got {dist[0]}"


def test_cast_rays_to_line_comprehensive():
    """
    Comprehensive test for cast_rays_to_line.

    Expected shapes:
      - ray_origin: [batch_dim, 2]
      - ray_direction: [batch_dim, n_angles]
      - line_pos: [batch_dim, n_lines, 2]
      - line_rot: [batch_dim, n_lines]
      - line_length: [batch_dim, n_lines]  <-- note the extra dimension!

    This test uses two batches (batch_dim=2) and one line per batch (n_lines=1).
    """
    batch_dim = 2
    n_lines = 1
    n_angles = 1

    # Create ray_origin with shape (2, 2)
    # Batch 0: ray_origin = (2, 1)
    # Batch 1: ray_origin = (0, 1)
    ray_origin = jnp.array([[2.0, 1.0], [0.0, 1.0]], dtype=jnp.float32)  # shape: (2, 2)

    # Create ray_direction with shape (2, 1)
    # Batch 0: ray_direction = 3.9269907 (≈225°)
    # Batch 1: ray_direction = 4.712389  (≈270°)
    ray_direction = jnp.array(
        [[3.9269907], [4.712389]], dtype=jnp.float32
    )  # shape: (2, 1)

    # Create line_pos with shape (2, 1, 2)
    # Both batches: line starting at (0, 0)
    line_pos = jnp.array(
        [[[0.0, 0.0]], [[0.0, 0.0]]], dtype=jnp.float32
    )  # shape: (2, 1, 2)

    # Create line_rot with shape (2, 1)
    # Both batches: horizontal line (0 radians)
    line_rot = jnp.array([[0.0], [0.0]], dtype=jnp.float32)  # shape: (2, 1)

    # IMPORTANT: Create line_length with shape (2, 1), not (2,)
    line_length = jnp.array([[2.0], [2.0]], dtype=jnp.float32)  # shape: (2, 1)

    # Now call cast_rays_to_line.
    # This function expects:
    #   batch_dim, line_pos, line_rot, line_length, ray_origin, ray_direction, max_range.
    distances = cast_rays_to_line(
        batch_dim, line_pos, line_rot, line_length, ray_origin, ray_direction, MAX_RANGE
    )
    # distances should have shape [batch_dim, n_lines, n_angles] -> (2, 1, 1)
    assert distances.shape == (
        2,
        1,
        1,
    ), f"Expected shape (2,1,1), got {distances.shape}"

    # For Batch 0:
    #   Horizontal line from (0,0) to (2,0). Ray_origin=(2,1) with direction=225° (5π/4)
    #   should hit the line at (1,0) → distance = sqrt((2-1)^2 + (1-0)^2) = sqrt(2).
    expected_0 = math.sqrt(2)
    # For Batch 1:
    #   Same horizontal line. Ray_origin=(0,1) with direction=270° (down)
    #   should hit at (0,0) → distance = 1.
    expected_1 = 1.0

    d0 = distances[0, 0, 0]
    d1 = distances[1, 0, 0]
    assert (
        jnp.abs(d0 - expected_0) < EPSILON
    ), f"Batch 0: expected {expected_0}, got {d0}"
    assert (
        jnp.abs(d1 - expected_1) < EPSILON
    ), f"Batch 1: expected {expected_1}, got {d1}"
