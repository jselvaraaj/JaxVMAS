#  Copyright (c) 2023-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.


import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxvmas.simulator.utils import JaxUtils

# Type dimensions
batch = "batch"
dim_p = "dim_p"
n = "n"


def _get_inner_point_box(
    outside_point: Float[Array, f"{batch} {dim_p}"],
    surface_point: Float[Array, f"{batch} {dim_p}"],
    box_pos: Float[Array, f"{batch} {dim_p}"],
) -> tuple[Float[Array, f"{batch} {dim_p}"], Float[Array, f"{batch}"]]:
    v = surface_point - outside_point
    u = box_pos - surface_point
    v_norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    x_magnitude = (v * u).sum(-1, keepdims=True) / v_norm
    x = (v / v_norm) * x_magnitude
    cond = v_norm == 0
    x = jnp.where(cond, surface_point, x)
    x_magnitude = jnp.where(cond, 0, x_magnitude)
    return surface_point + x, jnp.abs(x_magnitude.squeeze(-1))


def _get_closest_box_box(
    box_pos: Float[Array, f"{batch} {dim_p}"],
    box_rot: Float[Array, f"{batch} 1"],
    box_width: Float[Array, f"{batch}"] | float,
    box_length: Float[Array, f"{batch}"] | float,
    box2_pos: Float[Array, f"{batch} {dim_p}"],
    box2_rot: Float[Array, f"{batch} 1"],
    box2_width: Float[Array, f"{batch}"] | float,
    box2_length: Float[Array, f"{batch}"] | float,
) -> tuple[Float[Array, f"{batch} {dim_p}"], Float[Array, f"{batch} {dim_p}"]]:

    # Convert scalar inputs to arrays if needed
    if not isinstance(box_width, Array):
        box_width = jnp.full(box_pos.shape[0], box_width)
    if not isinstance(box_length, Array):
        box_length = jnp.full(box_pos.shape[0], box_length)
    if not isinstance(box2_width, Array):
        box2_width = jnp.full(box2_pos.shape[0], box2_width)
    if not isinstance(box2_length, Array):
        box2_length = jnp.full(box2_pos.shape[0], box2_length)

    # Get all lines for both boxes
    lines_pos, lines_rot, lines_length = _get_all_lines_box(
        jnp.stack([box_pos, box2_pos]),
        jnp.stack([box_rot, box2_rot]),
        jnp.stack([box_width, box2_width]),
        jnp.stack([box_length, box2_length]),
    )

    lines_a_pos, lines_b_pos = lines_pos[0], lines_pos[1]
    lines_a_rot, lines_b_rot = lines_rot[0], lines_rot[1]
    lines_a_length, lines_b_length = lines_length[0], lines_length[1]

    points_first, points_second = _get_closest_line_box(
        jnp.stack(
            [
                jnp.broadcast_to(box2_pos[None], lines_a_pos.shape),
                jnp.broadcast_to(box_pos[None], lines_b_pos.shape),
            ]
        ),
        jnp.stack(
            [
                jnp.broadcast_to(box2_rot[None], lines_a_rot.shape),
                jnp.broadcast_to(box_rot[None], lines_b_rot.shape),
            ]
        ),
        jnp.stack(
            [
                jnp.broadcast_to(box2_width[None], lines_a_length.shape),
                jnp.broadcast_to(box_width[None], lines_b_length.shape),
            ]
        ),
        jnp.stack(
            [
                jnp.broadcast_to(box2_length[None], lines_a_length.shape),
                jnp.broadcast_to(box_length[None], lines_b_length.shape),
            ]
        ),
        jnp.stack([lines_a_pos, lines_b_pos]),
        jnp.stack([lines_a_rot, lines_b_rot]),
        jnp.stack([lines_a_length, lines_b_length]),
    )

    points_box2_a, points_box_b = points_first[0], points_first[1]
    points_box_a, points_box2_b = points_second[0], points_second[1]

    # Concatenate all point pairs
    p1s = jnp.concatenate([points_box_a, points_box_b])
    p2s = jnp.concatenate([points_box2_a, points_box2_b])

    # Find the closest pair of points
    closest_point_1 = jnp.full_like(box_pos, jnp.inf)
    closest_point_2 = jnp.full_like(box_pos, jnp.inf)
    distance = jnp.full(box_pos.shape[:-1], jnp.inf)

    def update_closest(carry, x):
        closest_point_1, closest_point_2, distance = carry
        p1, p2 = x
        d = jnp.linalg.norm(p1 - p2, axis=-1)
        is_closest = d < distance
        closest_point_1 = jnp.where(is_closest[..., None], p1, closest_point_1)
        closest_point_2 = jnp.where(is_closest[..., None], p2, closest_point_2)
        distance = jnp.where(is_closest, d, distance)
        return (closest_point_1, closest_point_2, distance), None

    (closest_point_1, closest_point_2, _), _ = jax.lax.scan(
        update_closest, (closest_point_1, closest_point_2, distance), (p1s, p2s)
    )

    return closest_point_1, closest_point_2


def _get_line_extrema(
    line_pos: Float[Array, f"{batch} {dim_p}"],
    line_rot: Float[Array, f"{batch} 1"],
    line_length: Float[Array, f"{batch}"],
) -> tuple[Float[Array, f"{batch} {dim_p}"], Float[Array, f"{batch} {dim_p}"]]:
    line_length = line_length.reshape(line_rot.shape)
    x = (line_length / 2) * jnp.cos(line_rot)
    y = (line_length / 2) * jnp.sin(line_rot)
    xy = jnp.concatenate([x, y], axis=-1)

    point_a = line_pos + xy
    point_b = line_pos - xy

    return point_a, point_b


def _get_closest_points_line_line(
    line_pos: Float[Array, f"{batch} {dim_p}"],
    line_rot: Float[Array, f"{batch} 1"],
    line_length: Float[Array, f"{batch}"] | float,
    line2_pos: Float[Array, f"{batch} {dim_p}"],
    line2_rot: Float[Array, f"{batch} 1"],
    line2_length: Float[Array, f"{batch}"] | float,
) -> tuple[Float[Array, f"{batch} {dim_p}"], Float[Array, f"{batch} {dim_p}"]]:

    if not isinstance(line_length, Array):
        line_length = jnp.full(line_pos.shape[0], line_length)
    if not isinstance(line2_length, Array):
        line2_length = jnp.full(line2_pos.shape[0], line2_length)

    points_a, points_b = _get_line_extrema(
        jnp.stack([line_pos, line2_pos]),
        jnp.stack([line_rot, line2_rot]),
        jnp.stack([line_length, line2_length]),
    )
    point_a1, point_b1 = points_a[0], points_a[1]
    point_a2, point_b2 = points_b[0], points_b[1]

    point_i, d_i = _get_intersection_point_line_line(
        point_a1, point_a2, point_b1, point_b2
    )

    points = _get_closest_point_line(
        jnp.stack([line2_pos, line2_pos, line_pos, line_pos]),
        jnp.stack([line2_rot, line2_rot, line_rot, line_rot]),
        jnp.stack([line2_length, line2_length, line_length, line_length]),
        jnp.stack([point_a1, point_a2, point_b1, point_b2]),
    )

    point_a1_line_b, point_a2_line_b, point_b1_line_a, point_b2_line_a = points

    # Convert point pairs to arrays for jax.lax.scan
    p1s = jnp.stack([point_a1, point_a2, point_b1_line_a, point_b2_line_a])
    p2s = jnp.stack([point_a1_line_b, point_a2_line_b, point_b1, point_b2])

    closest_point_1 = jnp.full_like(line_pos, jnp.inf)
    closest_point_2 = jnp.full_like(line_pos, jnp.inf)
    min_distance = jnp.full(line_pos.shape[:-1], jnp.inf)

    def update_closest(carry, x):
        closest_point_1, closest_point_2, min_distance = carry
        p1, p2 = x
        d = jnp.linalg.norm(p1 - p2, axis=-1)
        is_closest = d < min_distance
        closest_point_1 = jnp.where(is_closest[..., None], p1, closest_point_1)
        closest_point_2 = jnp.where(is_closest[..., None], p2, closest_point_2)
        min_distance = jnp.where(is_closest, d, min_distance)
        return (closest_point_1, closest_point_2, min_distance), None

    (closest_point_1, closest_point_2, _), _ = jax.lax.scan(
        update_closest, (closest_point_1, closest_point_2, min_distance), (p1s, p2s)
    )

    cond = (d_i == 0)[..., None]
    closest_point_1 = jnp.where(cond, point_i, closest_point_1)
    closest_point_2 = jnp.where(cond, point_i, closest_point_2)

    return closest_point_1, closest_point_2


def _get_intersection_point_line_line(
    point_a1: Float[Array, f"{batch} {dim_p}"],
    point_a2: Float[Array, f"{batch} {dim_p}"],
    point_b1: Float[Array, f"{batch} {dim_p}"],
    point_b2: Float[Array, f"{batch} {dim_p}"],
) -> tuple[Float[Array, f"{batch} {dim_p}"], Float[Array, f"{batch}"]]:
    """
    Taken from:
    https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
    """
    r = point_a2 - point_a1
    s = point_b2 - point_b1
    p = point_a1
    q = point_b1
    cross_q_minus_p_r = JaxUtils.cross(q - p, r)
    cross_q_minus_p_s = JaxUtils.cross(q - p, s)
    cross_r_s = JaxUtils.cross(r, s)
    u = cross_q_minus_p_r / cross_r_s
    t = cross_q_minus_p_s / cross_r_s
    t_in_range = (0 <= t) & (t <= 1)
    u_in_range = (0 <= u) & (u <= 1)

    cross_r_s_is_zero = cross_r_s == 0

    distance = jnp.full(point_a1.shape[:-1], jnp.inf)
    point = jnp.full_like(point_a1, jnp.inf)

    condition = ~cross_r_s_is_zero & u_in_range & t_in_range
    condition_exp = jnp.broadcast_to(condition, point.shape)
    point = jnp.where(condition_exp, p + t * r, point)
    distance = jnp.where(condition.squeeze(-1), 0.0, distance)

    return point, distance


def _get_closest_point_box(
    box_pos: Float[Array, f"{batch} {dim_p}"],
    box_rot: Float[Array, f"{batch} 1"],
    box_width: Float[Array, f"{batch}"] | float,
    box_length: Float[Array, f"{batch}"] | float,
    test_point_pos: Float[Array, f"{batch} {dim_p}"],
) -> Float[Array, f"{batch} {dim_p}"]:

    if not isinstance(box_width, Array):
        box_width = jnp.full(box_pos.shape[0], box_width)
    if not isinstance(box_length, Array):
        box_length = jnp.full(box_pos.shape[0], box_length)

    closest_points = _get_all_points_box(
        box_pos, box_rot, box_width, box_length, test_point_pos
    )
    closest_point = jnp.full_like(box_pos, jnp.inf)
    distance = jnp.full(box_pos.shape[:-1], jnp.inf)

    def update_closest(carry, p):
        closest_point, distance = carry
        d = jnp.linalg.norm(test_point_pos - p, axis=-1)
        is_closest = d < distance
        closest_point = jnp.where(is_closest[..., None], p, closest_point)
        distance = jnp.where(is_closest, d, distance)
        return (closest_point, distance), None

    (closest_point, _), _ = jax.lax.scan(
        update_closest, (closest_point, distance), closest_points
    )

    return closest_point


def _get_all_lines_box(
    box_pos: Float[Array, f"{batch} {dim_p}"],
    box_rot: Float[Array, f"{batch} 1"],
    box_width: Float[Array, f"{batch}"],
    box_length: Float[Array, f"{batch}"],
) -> tuple[
    Float[Array, f"4 {batch} {dim_p}"],
    Float[Array, f"4 {batch} 1"],
    Float[Array, f"4 {batch}"],
]:
    # Rotate normal vector by the angle of the box
    rotated_vector = jnp.concatenate([jnp.cos(box_rot), jnp.sin(box_rot)], axis=-1)
    rot_2 = box_rot + jnp.pi / 2
    rotated_vector2 = jnp.concatenate([jnp.cos(rot_2), jnp.sin(rot_2)], axis=-1)

    expanded_half_box_length = box_length[..., None] / 2
    expanded_half_box_width = box_width[..., None] / 2

    # Middle points of the sides
    p1 = box_pos + rotated_vector * expanded_half_box_length
    p2 = box_pos - rotated_vector * expanded_half_box_length
    p3 = box_pos + rotated_vector2 * expanded_half_box_width
    p4 = box_pos - rotated_vector2 * expanded_half_box_width

    ps = jnp.stack([p1, p2, p3, p4])
    rots = jnp.stack([box_rot + jnp.pi / 2, box_rot + jnp.pi / 2, box_rot, box_rot])
    lengths = jnp.stack([box_width, box_width, box_length, box_length])

    return ps, rots, lengths


def _get_closest_line_box(
    box_pos: Float[Array, f"{batch} {dim_p}"],
    box_rot: Float[Array, f"{batch} 1"],
    box_width: Float[Array, f"{batch}"] | float,
    box_length: Float[Array, f"{batch}"] | float,
    line_pos: Float[Array, f"{batch} {dim_p}"],
    line_rot: Float[Array, f"{batch} 1"],
    line_length: Float[Array, f"{batch}"] | float,
) -> tuple[Float[Array, f"{batch} {dim_p}"], Float[Array, f"{batch} {dim_p}"]]:

    if not isinstance(box_width, Array):
        box_width = jnp.full(box_pos.shape[0], box_width)
    if not isinstance(box_length, Array):
        box_length = jnp.full(box_pos.shape[0], box_length)
    if not isinstance(line_length, Array):
        line_length = jnp.full(line_pos.shape[0], line_length)

    # Get all lines of the box
    lines_pos, lines_rot, lines_length = _get_all_lines_box(
        box_pos, box_rot, box_width, box_length
    )

    closest_point_1 = jnp.full_like(box_pos, jnp.inf)
    closest_point_2 = jnp.full_like(box_pos, jnp.inf)
    distance = jnp.full(box_pos.shape[:-1], jnp.inf)

    ps_box, ps_line = _get_closest_points_line_line(
        lines_pos,
        lines_rot,
        lines_length,
        jnp.broadcast_to(line_pos[None], lines_pos.shape),
        jnp.broadcast_to(line_rot[None], lines_rot.shape),
        jnp.broadcast_to(line_length[None], lines_length.shape),
    )

    # Find the closest pair of points
    def update_closest(carry, x):
        closest_point_1, closest_point_2, distance = carry
        p_box, p_line = x
        d = jnp.linalg.norm(p_box - p_line, axis=-1)
        is_closest = d < distance
        closest_point_1 = jnp.where(is_closest[..., None], p_box, closest_point_1)
        closest_point_2 = jnp.where(is_closest[..., None], p_line, closest_point_2)
        distance = jnp.where(is_closest, d, distance)
        return (closest_point_1, closest_point_2, distance), None

    (closest_point_1, closest_point_2, _), _ = jax.lax.scan(
        update_closest, (closest_point_1, closest_point_2, distance), (ps_box, ps_line)
    )

    return closest_point_1, closest_point_2


def _get_all_points_box(
    box_pos: Float[Array, f"{batch} {dim_p}"],
    box_rot: Float[Array, f"{batch} 1"],
    box_width: Float[Array, f"{batch}"],
    box_length: Float[Array, f"{batch}"],
    test_point_pos: Float[Array, f"{batch} {dim_p}"],
) -> Float[Array, f"4 {batch} {dim_p}"]:
    lines_pos, lines_rot, lines_length = _get_all_lines_box(
        box_pos, box_rot, box_width, box_length
    )

    closest_points = _get_closest_point_line(
        lines_pos,
        lines_rot,
        lines_length,
        jnp.broadcast_to(test_point_pos[None], lines_pos.shape),
    )

    return closest_points


def _get_closest_point_line(
    line_pos: Float[Array, f"{batch} {dim_p}"],
    line_rot: Float[Array, f"{batch} 1"],
    line_length: Float[Array, f"{batch}"] | float,
    test_point_pos: Float[Array, f"{batch} {dim_p}"],
    limit_to_line_length: bool = True,
) -> Float[Array, f"{batch} {dim_p}"]:
    if not isinstance(line_length, Array):
        line_length = jnp.full(line_rot.shape[0], line_length)

    # Rotate it by the angle of the line
    rotated_vector = jnp.concatenate([jnp.cos(line_rot), jnp.sin(line_rot)], axis=-1)
    # Get distance between line and sphere
    delta_pos = line_pos - test_point_pos
    # Dot product of distance and line vector
    dot_p = (delta_pos * rotated_vector).sum(-1, keepdims=True)
    # Coordinates of the closest point
    sign = jnp.sign(dot_p)
    distance_from_line_center = (
        jnp.minimum(
            jnp.abs(dot_p),
            (line_length / 2).reshape(dot_p.shape),
        )
        if limit_to_line_length
        else jnp.abs(dot_p)
    )
    closest_point = line_pos - sign * distance_from_line_center * rotated_vector
    return closest_point
