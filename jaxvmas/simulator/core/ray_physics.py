import chex
import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, jaxtyped

from jaxvmas.simulator.core.entity import Entity
from jaxvmas.simulator.core.shapes import Box, Line, Sphere
from jaxvmas.simulator.physics import _get_closest_point_line
from jaxvmas.simulator.utils import (
    JaxUtils,
    X,
    Y,
)


@eqx.filter_jit
@jaxtyped(typechecker=beartype)
def cast_ray_to_box(
    box: Entity,
    ray_origin: Array,
    ray_direction: Array,
    max_range: float,
):
    """
    Inspired from https://tavianator.com/2011/ray_box.html
    Computes distance of ray originating from pos at angle to a box and sets distance to
    max_range if there is no intersection.
    """
    assert ray_origin.ndim == 2 and ray_direction.ndim == 1
    assert ray_origin.shape[0] == ray_direction.shape[0]
    assert isinstance(box.shape, Box)

    pos_origin = ray_origin - box.state.pos
    pos_aabb = JaxUtils.rotate_vector(pos_origin, -box.state.rot)
    ray_dir_world = jnp.stack([jnp.cos(ray_direction), jnp.sin(ray_direction)], axis=-1)
    ray_dir_aabb = JaxUtils.rotate_vector(ray_dir_world, -box.state.rot)

    tx1 = (-box.shape.length / 2 - pos_aabb[:, X]) / ray_dir_aabb[:, X]
    tx2 = (box.shape.length / 2 - pos_aabb[:, X]) / ray_dir_aabb[:, X]
    tx = jnp.stack([tx1, tx2], axis=-1)
    tmin = jnp.min(tx, axis=-1)
    tmax = jnp.max(tx, axis=-1)

    ty1 = (-box.shape.width / 2 - pos_aabb[:, Y]) / ray_dir_aabb[:, Y]
    ty2 = (box.shape.width / 2 - pos_aabb[:, Y]) / ray_dir_aabb[:, Y]
    ty = jnp.stack([ty1, ty2], axis=-1)
    tymin = jnp.min(ty, axis=-1)
    tymax = jnp.max(ty, axis=-1)
    tmin = jnp.max(jnp.stack([tmin, tymin], axis=-1), axis=-1)
    tmax = jnp.min(jnp.stack([tmax, tymax], axis=-1), axis=-1)

    intersect_aabb = tmin[:, None] * ray_dir_aabb + pos_aabb
    intersect_world = (
        JaxUtils.rotate_vector(intersect_aabb, box.state.rot) + box.state.pos
    )

    collision = (tmax >= tmin) & (tmin > 0.0)
    dist = jnp.linalg.norm(ray_origin - intersect_world, axis=1)
    dist = jnp.where(collision, dist, max_range)
    return dist


@chex.assert_max_traces(1)
@jaxtyped(typechecker=beartype)
def cast_rays_to_box(
    batch_dim: int,
    box_pos: Array,
    box_rot: Array,
    box_length: Array,
    box_width: Array,
    ray_origin: Array,
    ray_direction: Array,
    max_range: float,
):
    """
    Inspired from https://tavianator.com/2011/ray_box.html
    Computes distance of ray originating from pos at angle to a box and sets distance to
    max_range if there is no intersection.
    """
    batch_size = ray_origin.shape[:-1]
    assert batch_size[0] == batch_dim
    assert ray_origin.shape[-1] == 2  # ray_origin is [*batch_size, 2]
    assert (
        ray_direction.shape[:-1] == batch_size
    )  # ray_direction is [*batch_size, n_angles]
    assert box_pos.shape[:-2] == batch_size
    assert box_pos.shape[-1] == 2
    assert box_rot.shape[:-1] == batch_size
    assert box_width.shape[:-1] == batch_size
    assert box_length.shape[:-1] == batch_size

    num_angles = ray_direction.shape[-1]
    n_boxes = box_pos.shape[-2]

    # Expand input to [*batch_size, n_boxes, num_angles, 2]
    ray_origin = jnp.broadcast_to(
        jnp.expand_dims(jnp.expand_dims(ray_origin, -2), -2),
        (*batch_size, n_boxes, num_angles, 2),
    )
    box_pos_expanded = jnp.broadcast_to(
        jnp.expand_dims(box_pos, -2), (*batch_size, n_boxes, num_angles, 2)
    )
    # Expand input to [*batch_size, n_boxes, num_angles]
    ray_direction = jnp.broadcast_to(
        jnp.expand_dims(ray_direction, -2), (*batch_size, n_boxes, num_angles)
    )
    box_rot_expanded = jnp.broadcast_to(
        jnp.expand_dims(box_rot, -1), (*batch_size, n_boxes, num_angles)
    )
    box_width_expanded = jnp.broadcast_to(
        jnp.expand_dims(box_width, -1), (*batch_size, n_boxes, num_angles)
    )
    box_length_expanded = jnp.broadcast_to(
        jnp.expand_dims(box_length, -1), (*batch_size, n_boxes, num_angles)
    )

    # Compute pos_origin and pos_aabb
    pos_origin = ray_origin - box_pos_expanded
    pos_aabb = JaxUtils.rotate_vector(pos_origin, -box_rot_expanded)

    # Calculate ray_dir_world
    ray_dir_world = jnp.stack([jnp.cos(ray_direction), jnp.sin(ray_direction)], axis=-1)

    # Calculate ray_dir_aabb
    ray_dir_aabb = JaxUtils.rotate_vector(ray_dir_world, -box_rot_expanded)

    # Calculate tx, ty, tmin, and tmax
    tx1 = (-box_length_expanded / 2 - pos_aabb[..., X]) / ray_dir_aabb[..., X]
    tx2 = (box_length_expanded / 2 - pos_aabb[..., X]) / ray_dir_aabb[..., X]
    tx = jnp.stack([tx1, tx2], axis=-1)
    tmin = jnp.min(tx, axis=-1)
    tmax = jnp.max(tx, axis=-1)

    ty1 = (-box_width_expanded / 2 - pos_aabb[..., Y]) / ray_dir_aabb[..., Y]
    ty2 = (box_width_expanded / 2 - pos_aabb[..., Y]) / ray_dir_aabb[..., Y]
    ty = jnp.stack([ty1, ty2], axis=-1)
    tymin = jnp.min(ty, axis=-1)
    tymax = jnp.max(ty, axis=-1)
    tmin = jnp.max(jnp.stack([tmin, tymin], axis=-1), axis=-1)
    tmax = jnp.min(jnp.stack([tmax, tymax], axis=-1), axis=-1)

    # Compute intersect_aabb and intersect_world
    intersect_aabb = tmin[..., None] * ray_dir_aabb + pos_aabb
    intersect_world = (
        JaxUtils.rotate_vector(intersect_aabb, box_rot_expanded) + box_pos_expanded
    )

    # Calculate collision and distances
    collision = (tmax >= tmin) & (tmin > 0.0)
    dist = jnp.linalg.norm(ray_origin - intersect_world, axis=-1)
    dist = jnp.where(collision, dist, max_range)
    return dist


@eqx.filter_jit
@jaxtyped(typechecker=beartype)
def cast_ray_to_sphere(
    sphere: Entity,
    ray_origin: Array,
    ray_direction: Array,
    max_range: float,
):
    ray_dir_world = jnp.stack([jnp.cos(ray_direction), jnp.sin(ray_direction)], axis=-1)
    test_point_pos = sphere.state.pos
    line_rot = ray_direction
    line_length = max_range
    line_pos = ray_origin + ray_dir_world * (line_length / 2)

    closest_point = _get_closest_point_line(
        line_pos,
        line_rot[..., None],
        line_length,
        test_point_pos,
        limit_to_line_length=False,
    )

    d = test_point_pos - closest_point
    d_norm = jnp.linalg.vector_norm(d, axis=1)

    assert isinstance(sphere.shape, Sphere)
    ray_intersects = d_norm < sphere.shape.radius
    a = sphere.shape.radius**2 - d_norm**2
    m = jnp.sqrt(jnp.where(a > 0, a, 1e-8))

    u = test_point_pos - ray_origin
    u1 = closest_point - ray_origin

    # Dot product of u and u1
    u_dot_ray = (u * ray_dir_world).sum(-1)
    sphere_is_in_front = u_dot_ray > 0.0
    dist = jnp.linalg.vector_norm(u1, axis=1) - m
    dist = jnp.where(ray_intersects & sphere_is_in_front, dist, max_range)

    return dist


@chex.assert_max_traces(1)
@jaxtyped(typechecker=beartype)
def cast_rays_to_sphere(
    batch_dim: int,
    sphere_pos: Array,
    sphere_radius: Array,
    ray_origin: Array,
    ray_direction: Array,
    max_range: float,
):
    batch_size = ray_origin.shape[:-1]
    assert batch_size[0] == batch_dim
    assert ray_origin.shape[-1] == 2  # ray_origin is [*batch_size, 2]
    assert (
        ray_direction.shape[:-1] == batch_size
    )  # ray_direction is [*batch_size, n_angles]
    assert sphere_pos.shape[:-2] == batch_size
    assert sphere_pos.shape[-1] == 2
    assert sphere_radius.shape[:-1] == batch_size

    num_angles = ray_direction.shape[-1]
    n_spheres = sphere_pos.shape[-2]

    # Expand input to [*batch_size, n_spheres, num_angles, 2]
    ray_origin = jnp.broadcast_to(
        ray_origin[..., None, None, :], (*batch_size, n_spheres, num_angles, 2)
    )
    sphere_pos_expanded = jnp.broadcast_to(
        sphere_pos[..., None, :], (*batch_size, n_spheres, num_angles, 2)
    )
    # Expand input to [*batch_size, n_spheres, num_angles]
    ray_direction = jnp.broadcast_to(
        ray_direction[..., None, :], (*batch_size, n_spheres, num_angles)
    )
    sphere_radius_expanded = jnp.broadcast_to(
        sphere_radius[..., None], (*batch_size, n_spheres, num_angles)
    )

    # Calculate ray_dir_world
    ray_dir_world = jnp.stack([jnp.cos(ray_direction), jnp.sin(ray_direction)], axis=-1)

    line_rot = ray_direction[..., None]

    # line_length remains scalar and will be broadcasted as needed
    line_length = max_range

    # Calculate line_pos
    line_pos = ray_origin + ray_dir_world * (line_length / 2)

    # Call the updated _get_closest_point_line function
    closest_point = _get_closest_point_line(
        line_pos,
        line_rot,
        line_length,
        sphere_pos_expanded,
        limit_to_line_length=False,
    )

    # Calculate distances and other metrics
    d = sphere_pos_expanded - closest_point
    d_norm = jnp.linalg.vector_norm(d, axis=-1)
    ray_intersects = d_norm < sphere_radius_expanded
    a = sphere_radius_expanded**2 - d_norm**2
    m = jnp.sqrt(jnp.where(a > 0, a, 1e-8))

    u = sphere_pos_expanded - ray_origin
    u1 = closest_point - ray_origin

    # Dot product of u and u1
    u_dot_ray = (u * ray_dir_world).sum(-1)
    sphere_is_in_front = u_dot_ray > 0.0
    dist = jnp.linalg.vector_norm(u1, axis=-1) - m
    dist = jnp.where(ray_intersects & sphere_is_in_front, dist, max_range)

    return dist


@eqx.filter_jit
@jaxtyped(typechecker=beartype)
def cast_ray_to_line(
    line: Entity,
    ray_origin: Array,
    ray_direction: Array,
    max_range: float,
):
    """
    Inspired by https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect/565282#565282
    Computes distance of ray originating from pos at angle to a line and sets distance to
    max_range if there is no intersection.
    """
    assert ray_origin.ndim == 2 and ray_direction.ndim == 1
    assert ray_origin.shape[0] == ray_direction.shape[0]
    assert isinstance(line.shape, Line)

    # Line segment vector
    p = line.state.pos
    r = (
        jnp.stack(
            [jnp.cos(line.state.rot.squeeze(1)), jnp.sin(line.state.rot.squeeze(1))],
            axis=-1,
        )
        * line.shape.length
    )

    # Ray vector
    q = ray_origin
    s = jnp.stack([jnp.cos(ray_direction), jnp.sin(ray_direction)], axis=-1)

    # Calculate intersection
    rxs = JaxUtils.cross(r, s)

    # Calculate intersection
    t = JaxUtils.cross(q - p, s / rxs)
    u = JaxUtils.cross(q - p, r / rxs)
    d = jnp.linalg.norm(u * s, axis=-1)
    perpendicular = rxs == 0.0
    above_line = t > 0.5
    below_line = t < -0.5
    behind_line = u < 0.0

    d = jnp.where(perpendicular.squeeze(-1), max_range, d)
    d = jnp.where(above_line.squeeze(-1), max_range, d)
    d = jnp.where(below_line.squeeze(-1), max_range, d)
    d = jnp.where(behind_line.squeeze(-1), max_range, d)

    return d


@chex.assert_max_traces(1)
@jaxtyped(typechecker=beartype)
def cast_rays_to_line(
    batch_dim: int,
    line_pos: Array,
    line_rot: Array,
    line_length: Array,
    ray_origin: Array,
    ray_direction: Array,
    max_range: float,
):
    """
    Inspired by https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect/565282#565282
    Computes distance of ray originating from pos at angle to a line and sets distance to
    max_range if there is no intersection.
    """
    batch_size = ray_origin.shape[:-1]
    assert batch_size[0] == batch_dim
    assert ray_origin.shape[-1] == 2  # ray_origin is [*batch_size, 2]
    assert (
        ray_direction.shape[:-1] == batch_size
    )  # ray_direction is [*batch_size, n_angles]
    assert line_pos.shape[:-2] == batch_size
    assert line_pos.shape[-1] == 2
    assert line_rot.shape[:-1] == batch_size
    assert line_length.shape[:-1] == batch_size

    num_angles = ray_direction.shape[-1]
    n_lines = line_pos.shape[-2]

    # Expand input to [*batch_size, n_lines, num_angles, 2]
    ray_origin = ray_origin[..., None, None, :]
    ray_origin = jnp.broadcast_to(ray_origin, (*batch_size, n_lines, num_angles, 2))

    line_pos_expanded = line_pos[..., None, :]
    line_pos_expanded = jnp.broadcast_to(
        line_pos_expanded, (*batch_size, n_lines, num_angles, 2)
    )

    # Expand input to [*batch_size, n_lines, num_angles]
    ray_direction = ray_direction[..., None, :]
    ray_direction = jnp.broadcast_to(ray_direction, (*batch_size, n_lines, num_angles))

    line_rot_expanded = line_rot[..., None]
    line_rot_expanded = jnp.broadcast_to(
        line_rot_expanded, (*batch_size, n_lines, num_angles)
    )

    line_length_expanded = line_length[..., None]
    line_length_expanded = jnp.broadcast_to(
        line_length_expanded, (*batch_size, n_lines, num_angles)
    )

    # Expand line state variables
    r = (
        jnp.stack(
            [
                jnp.cos(line_rot_expanded),
                jnp.sin(line_rot_expanded),
            ],
            axis=-1,
        )
        * line_length_expanded[..., None]
    )

    # Calculate q and s
    q = ray_origin
    s = jnp.stack(
        [
            jnp.cos(ray_direction),
            jnp.sin(ray_direction),
        ],
        axis=-1,
    )

    # Calculate rxs, t, u, and d
    rxs = JaxUtils.cross(r, s)
    t = JaxUtils.cross(q - line_pos_expanded, s / rxs)
    u = JaxUtils.cross(q - line_pos_expanded, r / rxs)
    d = jnp.linalg.norm(u * s, axis=-1)

    # Handle edge cases
    perpendicular = rxs == 0.0
    above_line = t > 0.5
    below_line = t < -0.5
    behind_line = u < 0.0
    d = jnp.where(perpendicular.squeeze(-1), max_range, d)
    d = jnp.where(above_line.squeeze(-1), max_range, d)
    d = jnp.where(below_line.squeeze(-1), max_range, d)
    d = jnp.where(behind_line.squeeze(-1), max_range, d)
    return d
