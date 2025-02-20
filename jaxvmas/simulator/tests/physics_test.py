import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float

from jaxvmas.simulator.physics import (
    _get_all_lines_box,
    _get_all_points_box,
    _get_closest_box_box,
    _get_closest_line_box,
    _get_closest_point_box,
    _get_closest_point_line,
    _get_closest_points_line_line,
    _get_inner_point_box,
    _get_intersection_point_line_line,
    _get_line_extrema,
)

# Type dimensions
batch = "batch"
dim_p = "dim_p"


class TestPhysics:
    @pytest.fixture
    def batch_size(self):
        return 2

    @pytest.fixture
    def box_data(self, batch_size):
        return {
            "pos": jnp.array(
                [[0.0, 0.0], [1.0, 1.0]]
            ),  # 2 boxes at different positions
            "rot": jnp.array(
                [[0.0], [jnp.pi / 4]]
            ),  # One aligned, one rotated 45 degrees
            "width": jnp.array([2.0, 1.0]),
            "length": jnp.array([2.0, 1.0]),
        }

    @pytest.fixture
    def line_data(self, batch_size):
        return {
            "pos": jnp.array([[3.0, 0.0], [2.0, 2.0]]),
            "rot": jnp.array([[0.0], [jnp.pi / 2]]),  # One horizontal, one vertical
            "length": jnp.array([1.0, 1.0]),
        }

    def test_get_inner_point_box(self, batch_size):
        # Test case 1: Normal case
        outside_point = jnp.array([[1.0, 1.0], [2.0, 2.0]])
        surface_point = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        box_pos = jnp.array([[0.0, 0.0], [0.0, 0.0]])
        inner_point, distance = _get_inner_point_box(
            outside_point, surface_point, box_pos
        )
        assert inner_point.shape == (batch_size, 2)
        assert distance.shape == (batch_size,)

        # Test case 2: Zero vector case
        outside_point = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        surface_point = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        box_pos = jnp.array([[0.0, 0.0], [0.0, 0.0]])
        inner_point, distance = _get_inner_point_box(
            outside_point, surface_point, box_pos
        )
        # When v_norm is zero, the function returns surface_point + surface_point
        expected_point = surface_point + surface_point
        assert jnp.allclose(inner_point, expected_point, rtol=1e-5, atol=1e-5)
        assert jnp.allclose(distance, jnp.zeros(batch_size), rtol=1e-5, atol=1e-5)

        # Test jit compatibility
        jitted_func = jax.jit(_get_inner_point_box)
        jit_inner_point, jit_distance = jitted_func(
            outside_point, surface_point, box_pos
        )
        assert jnp.allclose(jit_inner_point, inner_point, rtol=1e-5, atol=1e-5)
        assert jnp.allclose(jit_distance, distance, rtol=1e-5, atol=1e-5)

    def test_get_closest_box_box(self, box_data):
        # Test case 1: Non-overlapping boxes
        point1, point2 = _get_closest_box_box(
            box_data["pos"],
            box_data["rot"],
            box_data["width"],
            box_data["length"],
            box_data["pos"]
            + jnp.array([[5.0, 5.0], [5.0, 5.0]]),  # Shift second box away
            box_data["rot"],
            box_data["width"],
            box_data["length"],
        )
        assert point1.shape == (2, 2)
        assert point2.shape == (2, 2)

        # Test case 2: Overlapping boxes
        point1_overlap, point2_overlap = _get_closest_box_box(
            box_data["pos"],
            box_data["rot"],
            box_data["width"],
            box_data["length"],
            box_data["pos"],  # Same position
            box_data["rot"],
            box_data["width"],
            box_data["length"],
        )
        assert jnp.allclose(point1_overlap, point2_overlap)

        # Test jit compatibility
        @eqx.filter_jit
        def jitted_closest_box_box(
            box_pos: Float[Array, f"{batch} {dim_p}"],
            box_rot: Float[Array, f"{batch} 1"],
            box_width: Float[Array, f"{batch}"],
            box_length: Float[Array, f"{batch}"],
            box2_pos: Float[Array, f"{batch} {dim_p}"],
            box2_rot: Float[Array, f"{batch} 1"],
            box2_width: Float[Array, f"{batch}"],
            box2_length: Float[Array, f"{batch}"],
        ):
            return _get_closest_box_box(
                box_pos,
                box_rot,
                box_width,
                box_length,
                box2_pos,
                box2_rot,
                box2_width,
                box2_length,
            )

        jitted_point1, jitted_point2 = jitted_closest_box_box(
            box_data["pos"],
            box_data["rot"],
            box_data["width"],
            box_data["length"],
            box_data["pos"] + jnp.array([[5.0, 5.0], [5.0, 5.0]]),
            box_data["rot"],
            box_data["width"],
            box_data["length"],
        )
        assert jnp.allclose(jitted_point1, point1, rtol=1e-5, atol=1e-5)
        assert jnp.allclose(jitted_point2, point2, rtol=1e-5, atol=1e-5)

    def test_get_line_extrema(self, line_data):
        point_a, point_b = _get_line_extrema(
            line_data["pos"], line_data["rot"], line_data["length"]
        )
        assert point_a.shape == (2, 2)
        assert point_b.shape == (2, 2)

        # Test horizontal line
        expected_a = jnp.array([0.5, 0.0])  # Right endpoint
        expected_b = jnp.array([-0.5, 0.0])  # Left endpoint
        assert jnp.allclose(
            point_a[0] - line_data["pos"][0], expected_a, rtol=1e-5, atol=1e-5
        )
        assert jnp.allclose(
            point_b[0] - line_data["pos"][0], expected_b, rtol=1e-5, atol=1e-5
        )

        # Test vertical line
        expected_a_vert = jnp.array([0.0, 0.5])  # Top endpoint
        expected_b_vert = jnp.array([0.0, -0.5])  # Bottom endpoint
        assert jnp.allclose(
            point_a[1] - line_data["pos"][1], expected_a_vert, rtol=1e-5, atol=1e-5
        )
        assert jnp.allclose(
            point_b[1] - line_data["pos"][1], expected_b_vert, rtol=1e-5, atol=1e-5
        )

        # Test jit compatibility
        @eqx.filter_jit
        def jitted_line_extrema(
            line_pos: Float[Array, f"{batch} {dim_p}"],
            line_rot: Float[Array, f"{batch} 1"],
            line_length: Float[Array, f"{batch}"],
        ):
            return _get_line_extrema(line_pos, line_rot, line_length)

        jitted_a, jitted_b = jitted_line_extrema(
            line_data["pos"], line_data["rot"], line_data["length"]
        )
        assert jnp.allclose(jitted_a, point_a, rtol=1e-5, atol=1e-5)
        assert jnp.allclose(jitted_b, point_b, rtol=1e-5, atol=1e-5)

    def test_get_closest_points_line_line(self, line_data):
        # Test parallel lines
        line2_pos = line_data["pos"] + jnp.array([[0.0, 1.0], [1.0, 0.0]])
        point1, point2 = _get_closest_points_line_line(
            line_data["pos"],
            line_data["rot"],
            line_data["length"],
            line2_pos,
            line_data["rot"],
            line_data["length"],
        )
        assert point1.shape == (2, 2)
        assert point2.shape == (2, 2)

        # Test perpendicular lines
        perp_rot = line_data["rot"] + jnp.pi / 2
        point1_perp, point2_perp = _get_closest_points_line_line(
            line_data["pos"],
            line_data["rot"],
            line_data["length"],
            line2_pos,
            perp_rot,
            line_data["length"],
        )
        assert point1_perp.shape == (2, 2)
        assert point2_perp.shape == (2, 2)

        # Test jit compatibility
        @eqx.filter_jit
        def jitted_closest_points_line_line(
            line_pos: Float[Array, f"{batch} {dim_p}"],
            line_rot: Float[Array, f"{batch} 1"],
            line_length: Float[Array, f"{batch}"],
            line2_pos: Float[Array, f"{batch} {dim_p}"],
            line2_rot: Float[Array, f"{batch} 1"],
            line2_length: Float[Array, f"{batch}"],
        ):
            return _get_closest_points_line_line(
                line_pos, line_rot, line_length, line2_pos, line2_rot, line2_length
            )

        jitted_point1, jitted_point2 = jitted_closest_points_line_line(
            line_data["pos"],
            line_data["rot"],
            line_data["length"],
            line2_pos,
            perp_rot,
            line_data["length"],
        )
        assert jnp.allclose(jitted_point1, point1_perp, rtol=1e-5, atol=1e-5)
        assert jnp.allclose(jitted_point2, point2_perp, rtol=1e-5, atol=1e-5)

    def test_get_intersection_point_line_line(self, line_data):
        # Test intersecting lines
        point_a1 = jnp.array([[0.0, 0.0], [0.0, 0.0]])
        point_a2 = jnp.array([[1.0, 0.0], [1.0, 0.0]])
        point_b1 = jnp.array([[0.5, -1.0], [0.5, -1.0]])
        point_b2 = jnp.array([[0.5, 1.0], [0.5, 1.0]])

        intersection_point, distance = _get_intersection_point_line_line(
            point_a1, point_a2, point_b1, point_b2
        )
        assert intersection_point.shape == (2, 2)
        assert distance.shape == (2,)

        # Test intersection point
        expected_intersection = jnp.array([[0.5, 0.0], [0.5, 0.0]])
        assert jnp.allclose(
            intersection_point, expected_intersection, rtol=1e-5, atol=1e-5
        )
        assert jnp.allclose(distance, jnp.zeros(2), rtol=1e-5, atol=1e-5)

        # Test parallel lines
        point_b1_parallel = point_a1 + jnp.array([[0.0, 1.0], [0.0, 1.0]])
        point_b2_parallel = point_a2 + jnp.array([[0.0, 1.0], [0.0, 1.0]])
        intersection_point_parallel, distance_parallel = (
            _get_intersection_point_line_line(
                point_a1, point_a2, point_b1_parallel, point_b2_parallel
            )
        )
        assert jnp.all(jnp.isinf(distance_parallel))

        # Test jit compatibility
        @eqx.filter_jit
        def jitted_intersection_point(
            point_a1: Float[Array, f"{batch} {dim_p}"],
            point_a2: Float[Array, f"{batch} {dim_p}"],
            point_b1: Float[Array, f"{batch} {dim_p}"],
            point_b2: Float[Array, f"{batch} {dim_p}"],
        ):
            return _get_intersection_point_line_line(
                point_a1, point_a2, point_b1, point_b2
            )

        jitted_point, jitted_distance = jitted_intersection_point(
            point_a1, point_a2, point_b1, point_b2
        )
        assert jnp.allclose(jitted_point, intersection_point, rtol=1e-5, atol=1e-5)
        assert jnp.allclose(jitted_distance, distance, rtol=1e-5, atol=1e-5)

    def test_get_closest_point_box(self, box_data):
        # Test points outside box
        test_points = jnp.array([[3.0, 3.0], [4.0, 4.0]])
        closest_points = _get_closest_point_box(
            box_data["pos"],
            box_data["rot"],
            box_data["width"],
            box_data["length"],
            test_points,
        )
        assert closest_points.shape == (2, 2)

        # Test points inside box
        inside_points = box_data["pos"]
        inside_closest = _get_closest_point_box(
            box_data["pos"],
            box_data["rot"],
            box_data["width"],
            box_data["length"],
            inside_points,
        )
        assert inside_closest.shape == (2, 2)
        # Points inside box should be projected to the nearest edge
        assert jnp.all(jnp.isfinite(inside_closest))

        # Test jit compatibility
        @eqx.filter_jit
        def jitted_closest_point_box(
            box_pos: Float[Array, f"{batch} {dim_p}"],
            box_rot: Float[Array, f"{batch} 1"],
            box_width: Float[Array, f"{batch}"],
            box_length: Float[Array, f"{batch}"],
            test_point_pos: Float[Array, f"{batch} {dim_p}"],
        ):
            return _get_closest_point_box(
                box_pos, box_rot, box_width, box_length, test_point_pos
            )

        jitted_points = jitted_closest_point_box(
            box_data["pos"],
            box_data["rot"],
            box_data["width"],
            box_data["length"],
            test_points,
        )
        assert jnp.allclose(jitted_points, closest_points, rtol=1e-5, atol=1e-5)

    def test_get_closest_point_line(self, line_data):
        test_points = jnp.array([[1.0, 1.0], [2.0, 2.0]])
        closest_points = _get_closest_point_line(
            line_data["pos"],
            line_data["rot"],
            line_data["length"],
            test_points,
        )
        assert closest_points.shape == (2, 2)

        # Test point on line
        on_line_points = line_data["pos"]
        on_line_closest = _get_closest_point_line(
            line_data["pos"],
            line_data["rot"],
            line_data["length"],
            on_line_points,
        )
        assert jnp.allclose(on_line_closest, on_line_points)

        # Test jit compatibility
        @eqx.filter_jit
        def jitted_closest_point_line(
            line_pos: Float[Array, f"{batch} {dim_p}"],
            line_rot: Float[Array, f"{batch} 1"],
            line_length: Float[Array, f"{batch}"],
            test_point_pos: Float[Array, f"{batch} {dim_p}"],
        ):
            return _get_closest_point_line(
                line_pos, line_rot, line_length, test_point_pos
            )

        jitted_points = jitted_closest_point_line(
            line_data["pos"],
            line_data["rot"],
            line_data["length"],
            test_points,
        )
        assert jnp.allclose(jitted_points, closest_points, rtol=1e-5, atol=1e-5)

    def test_get_all_lines_box(self, box_data):
        # Test getting all lines of a box
        lines_pos, lines_rot, lines_length = _get_all_lines_box(
            box_data["pos"],
            box_data["rot"],
            box_data["width"],
            box_data["length"],
        )
        assert lines_pos.shape == (4, 2, 2)  # 4 lines, batch_size=2, dim_p=2
        assert lines_rot.shape == (4, 2, 1)  # 4 lines, batch_size=2, 1 rotation angle
        assert lines_length.shape == (4, 2)  # 4 lines, batch_size=2

        # Test that lengths match box dimensions
        assert jnp.allclose(lines_length[0], box_data["width"])  # Top line
        assert jnp.allclose(lines_length[1], box_data["width"])  # Bottom line
        assert jnp.allclose(lines_length[2], box_data["length"])  # Left line
        assert jnp.allclose(lines_length[3], box_data["length"])  # Right line

        # Test jit compatibility
        @eqx.filter_jit
        def jitted_all_lines_box(
            box_pos: Float[Array, f"{batch} {dim_p}"],
            box_rot: Float[Array, f"{batch} 1"],
            box_width: Float[Array, f"{batch}"],
            box_length: Float[Array, f"{batch}"],
        ):
            return _get_all_lines_box(box_pos, box_rot, box_width, box_length)

        jitted_pos, jitted_rot, jitted_length = jitted_all_lines_box(
            box_data["pos"],
            box_data["rot"],
            box_data["width"],
            box_data["length"],
        )
        assert jnp.allclose(jitted_pos, lines_pos, rtol=1e-5, atol=1e-5)
        assert jnp.allclose(jitted_rot, lines_rot, rtol=1e-5, atol=1e-5)
        assert jnp.allclose(jitted_length, lines_length, rtol=1e-5, atol=1e-5)

    def test_get_closest_line_box(self, box_data, line_data):
        # Test finding closest points between line and box
        point1, point2 = _get_closest_line_box(
            box_data["pos"],
            box_data["rot"],
            box_data["width"],
            box_data["length"],
            line_data["pos"],
            line_data["rot"],
            line_data["length"],
        )
        assert point1.shape == (2, 2)
        assert point2.shape == (2, 2)

        # Test with line intersecting box
        intersecting_line_pos = box_data["pos"]  # Line starting at box center
        point1_intersect, point2_intersect = _get_closest_line_box(
            box_data["pos"],
            box_data["rot"],
            box_data["width"],
            box_data["length"],
            intersecting_line_pos,
            line_data["rot"],
            jnp.array([3.0, 3.0]),
        )
        assert jnp.allclose(point1_intersect, point2_intersect)

        # Test jit compatibility
        @eqx.filter_jit
        def jitted_closest_line_box(
            box_pos: Float[Array, f"{batch} {dim_p}"],
            box_rot: Float[Array, f"{batch} 1"],
            box_width: Float[Array, f"{batch}"],
            box_length: Float[Array, f"{batch}"],
            line_pos: Float[Array, f"{batch} {dim_p}"],
            line_rot: Float[Array, f"{batch} 1"],
            line_length: Float[Array, f"{batch}"],
        ):
            return _get_closest_line_box(
                box_pos,
                box_rot,
                box_width,
                box_length,
                line_pos,
                line_rot,
                line_length,
            )

        jitted_point1, jitted_point2 = jitted_closest_line_box(
            box_data["pos"],
            box_data["rot"],
            box_data["width"],
            box_data["length"],
            line_data["pos"],
            line_data["rot"],
            line_data["length"],
        )
        assert jnp.allclose(jitted_point1, point1, rtol=1e-5, atol=1e-5)
        assert jnp.allclose(jitted_point2, point2, rtol=1e-5, atol=1e-5)

    def test_get_all_points_box(self, box_data):
        test_point = jnp.array([[3.0, 3.0], [4.0, 4.0]])
        points = _get_all_points_box(
            box_data["pos"],
            box_data["rot"],
            box_data["width"],
            box_data["length"],
            test_point,
        )

        points = jnp.stack(points)
        assert points.shape == (4, 2, 2)  # 4 points per box, batch_size=2, dim_p=2

        # Test with point at box center
        center_points = _get_all_points_box(
            box_data["pos"],
            box_data["rot"],
            box_data["width"],
            box_data["length"],
            box_data["pos"],
        )
        center_points = jnp.stack(center_points)
        assert center_points.shape == (4, 2, 2)

        # Test jit compatibility
        @eqx.filter_jit
        def jitted_all_points_box(
            box_pos: Float[Array, f"{batch} {dim_p}"],
            box_rot: Float[Array, f"{batch} 1"],
            box_width: Float[Array, f"{batch}"],
            box_length: Float[Array, f"{batch}"],
            test_point_pos: Float[Array, f"{batch} {dim_p}"],
        ):
            return _get_all_points_box(
                box_pos, box_rot, box_width, box_length, test_point_pos
            )

        jitted_points = jitted_all_points_box(
            box_data["pos"],
            box_data["rot"],
            box_data["width"],
            box_data["length"],
            test_point,
        )

        jitted_points = jnp.stack(jitted_points)

        assert jnp.allclose(jitted_points, points, rtol=1e-5, atol=1e-5)
