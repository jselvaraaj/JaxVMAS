#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from __future__ import annotations

import math
import os
import sys
from typing import Callable

import jax
import jax.numpy as jnp
import pyglet
from jaxtyping import Array, Float, Int

from jaxvmas.simulator.utils import x_to_rgb_colormap

# Type variables for dimensions
batch = "batch"
vertex = "vertex"
dim_2 = "dim_2"
dim_3 = "dim_3"
dim_4 = "dim_4"
n = "n"

try:
    from pyglet.gl import (
        GL_BLEND,
        GL_LINE_LOOP,
        GL_LINE_SMOOTH,
        GL_LINE_SMOOTH_HINT,
        GL_LINE_STIPPLE,
        GL_LINE_STRIP,
        GL_LINES,
        GL_NICEST,
        GL_ONE_MINUS_SRC_ALPHA,
        GL_POINTS,
        GL_POLYGON,
        GL_QUADS,
        GL_SRC_ALPHA,
        GL_TRIANGLES,
        glBegin,
        glBlendFunc,
        glClearColor,
        glColor4f,
        glDisable,
        glEnable,
        glEnd,
        glHint,
        glLineStipple,
        glLineWidth,
        glPopMatrix,
        glPushMatrix,
        glRotatef,
        glScalef,
        glTranslatef,
        gluOrtho2D,
        glVertex2f,
        glVertex3f,
    )
except ImportError:
    raise ImportError(
        "Error occurred while running `from pyglet.gl import *`. HINT: make sure you have OpenGL installed. "
        "On Ubuntu, you can run 'apt-get install python3-opengl'. If you're running on a server, you may need a "
        "virtual frame buffer; something like this should work: "
        "'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'"
    )

if "Apple" in sys.version:
    if "DYLD_FALLBACK_LIBRARY_PATH" in os.environ:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] += ":/usr/lib"

RAD2DEG = 57.29577951308232


def get_display(spec: None | str) -> None | pyglet.canvas.Display:
    """Convert a display specification into an actual Display object."""
    if spec is None:
        return None
    elif isinstance(spec, str):
        return pyglet.canvas.Display(spec)
    else:
        raise RuntimeError(
            f"Invalid display specification: {spec}. (Must be a string like :0 or None.)"
        )


class Viewer:
    def __init__(
        self,
        width: int,
        height: int,
        display: None | str = None,
        visible: bool = True,
    ):
        self.display = get_display(display)
        self.width = width
        self.height = height

        self.window = pyglet.window.Window(
            width=width, height=height, display=self.display, visible=visible
        )
        self.window.on_close = self.window_closed_by_user

        self.geoms: list[Geom] = []
        self.onetime_geoms: list[Geom] = []
        self.transform = Transform()
        self.bounds: None | Float[Array, f"{dim_4}"] = None

        glEnable(GL_BLEND)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glLineWidth(2.0)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def close(self) -> None:
        self.window.close()

    def window_closed_by_user(self) -> None:
        self.close()

    def set_bounds(
        self,
        left: Float[Array, ""],
        right: Float[Array, ""],
        bottom: Float[Array, ""],
        top: Float[Array, ""],
    ) -> None:
        assert right > left and top > bottom
        self.bounds = jnp.array([left, right, bottom, top])
        scalex = self.width / (right - left)
        scaley = self.height / (top - bottom)
        self.transform = Transform(
            translation=(-left * scalex, -bottom * scaley), scale=(scalex, scaley)
        )

    def add_geom(self, geom: Geom) -> None:
        self.geoms.append(geom)

    def add_onetime(self, geom: Geom) -> None:
        self.onetime_geoms.append(geom)

    def render(self, return_rgb_array: bool = False) -> None | Int[Array, f"{n} {n} 3"]:
        glClearColor(1, 1, 1, 1)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        self.transform.enable()
        for geom in self.geoms + self.onetime_geoms:
            geom.render()
        self.transform.disable()

        arr = None
        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            # Convert buffer to JAX array directly
            arr = jnp.frombuffer(image_data.get_data(), dtype=jnp.uint8)
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]

        self.window.flip()
        self.onetime_geoms = []
        return arr


class Geom:
    def __init__(self):
        self._color = Color((0, 0, 0, 1.0))
        self.attrs = [self._color]

    def render(self):
        for attr in reversed(self.attrs):
            attr.enable()
        self.render1()
        for attr in self.attrs:
            attr.disable()

    def render1(self):
        raise NotImplementedError

    def add_attr(self, attr: Attr) -> None:
        self.attrs.append(attr)

    def set_color(self, r: float, g: float, b: float, alpha: float = 1.0) -> None:
        self._color.vec4 = (r, g, b, alpha)


class Attr:
    def enable(self):
        raise NotImplementedError

    def disable(self):
        pass


class Transform(Attr):
    def __init__(
        self,
        translation: tuple[float, float] = (0.0, 0.0),
        rotation: float = 0.0,
        scale: tuple[float, float] = (1.0, 1.0),
    ):
        self.set_translation(*translation)
        self.set_rotation(rotation)
        self.set_scale(*scale)

    def enable(self):
        glPushMatrix()
        glTranslatef(self.translation[0], self.translation[1], 0)
        glRotatef(RAD2DEG * self.rotation, 0, 0, 1.0)
        glScalef(self.scale[0], self.scale[1], 1)

    def disable(self):
        glPopMatrix()

    def set_translation(self, newx: float, newy: float) -> None:
        self.translation = (float(newx), float(newy))

    def set_rotation(self, new: float) -> None:
        self.rotation = float(new)

    def set_scale(self, newx: float, newy: float) -> None:
        self.scale = (float(newx), float(newy))


class Color(Attr):
    def __init__(self, vec4: tuple[float, float, float, float]):
        self.vec4 = vec4

    def enable(self):
        glColor4f(*self.vec4)


class LineStyle(Attr):
    def __init__(self, style: int):
        self.style = style

    def enable(self):
        glEnable(GL_LINE_STIPPLE)
        glLineStipple(1, self.style)

    def disable(self):
        glDisable(GL_LINE_STIPPLE)


class LineWidth(Attr):
    def __init__(self, stroke: float):
        self.stroke = stroke

    def enable(self):
        glLineWidth(self.stroke)


class Point(Geom):
    def render1(self):
        glBegin(GL_POINTS)
        glVertex3f(0.0, 0.0, 0.0)
        glEnd()


class FilledPolygon(Geom):
    def __init__(
        self, vertices: Float[Array, f"{vertex} {dim_2}"], draw_border: bool = True
    ):
        super().__init__()
        self.vertices = vertices
        self.draw_border = draw_border

    def render1(self):
        if len(self.vertices) == 4:
            glBegin(GL_QUADS)
        elif len(self.vertices) > 4:
            glBegin(GL_POLYGON)
        else:
            glBegin(GL_TRIANGLES)

        for v in self.vertices:
            glVertex3f(v[0], v[1], 0)
        glEnd()

        if self.draw_border:
            color = tuple(c * 0.5 for c in self._color.vec4)
            glColor4f(*color)
            glBegin(GL_LINE_LOOP)
            for v in self.vertices:
                glVertex3f(v[0], v[1], 0)
            glEnd()


class PolyLine(Geom):
    def __init__(
        self,
        vertices: Float[Array, f"{vertex} {dim_2}"],
        close: bool,
        linewidth: float = 1,
    ):
        super().__init__()
        self.vertices = vertices
        self.close = close
        self.linewidth = LineWidth(linewidth)
        self.add_attr(self.linewidth)

    def render1(self):
        glBegin(GL_LINE_LOOP if self.close else GL_LINE_STRIP)
        for v in self.vertices:
            glVertex3f(v[0], v[1], 0)
        glEnd()

    def set_linewidth(self, x: float) -> None:
        self.linewidth.stroke = x


class Line(Geom):
    def __init__(
        self,
        start: tuple[float, float] = (0.0, 0.0),
        end: tuple[float, float] = (0.0, 0.0),
        width: float = 1,
    ):
        super().__init__()
        self.start = start
        self.end = end
        self.linewidth = LineWidth(width)
        self.add_attr(self.linewidth)

    def render1(self):
        glBegin(GL_LINES)
        glVertex2f(*self.start)
        glVertex2f(*self.end)
        glEnd()

    def set_linewidth(self, x: float) -> None:
        self.linewidth.stroke = x


def make_circle(
    radius: float = 10, res: int = 30, filled: bool = True, angle: float = 2 * math.pi
) -> FilledPolygon | PolyLine:
    points = []
    for i in range(res):
        ang = -angle / 2 + angle * i / res
        points.append((math.cos(ang) * radius, math.sin(ang) * radius))

    if angle % (2 * math.pi) != 0:
        points.append((0, 0))

    points = jnp.array(points)
    return FilledPolygon(points) if filled else PolyLine(points, True)


def make_polygon(
    vertices: Float[Array, f"{vertex} {dim_2}"],
    filled: bool = True,
    draw_border: bool = True,
) -> FilledPolygon | PolyLine:
    if filled:
        return FilledPolygon(vertices, draw_border=draw_border)
    else:
        return PolyLine(vertices, True)


def make_polyline(vertices: Float[Array, f"{vertex} {dim_2}"]) -> PolyLine:
    return PolyLine(vertices, False)


def make_capsule(length: float, width: float) -> Compound:
    l, r, t, b = 0, length, width / 2, -width / 2
    box = make_polygon(jnp.array([(l, b), (l, t), (r, t), (r, b)]))
    circ0 = make_circle(width / 2)
    circ1 = make_circle(width / 2)
    circ1.add_attr(Transform(translation=(length, 0)))
    return Compound([box, circ0, circ1])


class Compound(Geom):
    def __init__(self, gs: list[Geom]):
        super().__init__()
        self.gs = gs
        for g in self.gs:
            g.attrs = [a for a in g.attrs if not isinstance(a, Color)]

    def render1(self):
        for g in self.gs:
            g.render()


class Grid(Geom):
    def __init__(self, spacing: float = 0.1, length: float = 50, width: float = 0.5):
        super().__init__()
        self.spacing = spacing
        self.length = length
        self.linewidth = LineWidth(width)
        self.add_attr(self.linewidth)

    def render1(self):
        for point in jnp.arange(-self.length / 2, self.length / 2, self.spacing):
            glBegin(GL_LINES)
            glVertex2f(point, -self.length / 2)
            glVertex2f(point, self.length / 2)
            glEnd()
            glBegin(GL_LINES)
            glVertex2f(-self.length / 2, point)
            glVertex2f(self.length / 2, point)
            glEnd()

    def set_linewidth(self, x: float) -> None:
        self.linewidth.stroke = x


class Image(Geom):
    def __init__(
        self, img: Float[Array, f"{n} {n} 4"], x: float, y: float, scale: float
    ):
        super().__init__()
        self.x = x
        self.y = y
        self.scale = scale
        img_shape = img.shape
        # Convert to uint8 using JAX
        img = jnp.asarray(img, dtype=jnp.uint8)
        # Get device array for OpenGL
        img_data = jax.device_get(img.reshape(-1))
        tex_data = (pyglet.gl.GLubyte * len(img_data))(*img_data)
        pyg_img = pyglet.image.ImageData(
            img_shape[1],
            img_shape[0],
            "RGBA",
            tex_data,
            pitch=img_shape[1] * img_shape[2] * 1,
        )
        self.img = pyg_img
        self.sprite = pyglet.sprite.Sprite(
            img=self.img, x=self.x, y=self.y, subpixel=True
        )
        self.sprite.update(scale=self.scale)

    def render1(self):
        self.sprite.draw()


class TextLine(Geom):
    def __init__(
        self,
        text: str = "",
        font_size: int = 15,
        x: float = 0.0,
        y: float = 0.0,
    ):
        super().__init__()

        if pyglet.font.have_font("Courier"):
            font = "Courier"
        elif pyglet.font.have_font("Secret Code"):
            font = "Secret Code"
        else:
            font = None

        self.label = pyglet.text.Label(
            text,
            font_name=font,
            font_size=font_size,
            color=(0, 0, 0, 255),
            x=x,
            y=y,
            anchor_x="left",
            anchor_y="bottom",
        )

    def render1(self):
        if self.label is not None:
            self.label.draw()

    def set_text(self, text: str, font_size: None | int = None) -> None:
        self.label.text = text
        if font_size is not None:
            self.label.font_size = font_size


def render_function_util(
    f: Callable[
        [Float[Array, f"{n} {dim_2}"]], Float[Array, f"{n}"] | Float[Array, f"{n} 4"]
    ],
    plot_range: (
        float | tuple[float, float] | tuple[tuple[float, float], tuple[float, float]]
    ),
    precision: float = 0.01,
    cmap_range: None | tuple[float, float] = None,
    cmap_alpha: float = 1.0,
    cmap_name: str = "viridis",
) -> Image:
    """Utility function to render a function over a 2D grid.

    Args:
        f: Function that takes points and returns either scalar values or RGBA values
        plot_range: Range to plot over, can be single value, (x,y) or ((xmin,xmax), (ymin,ymax))
        precision: Grid spacing
        cmap_range: Optional range for colormap normalization
        cmap_alpha: Alpha value for colormap
        cmap_name: Name of colormap to use

    Returns:
        Image geometry containing rendered function
    """
    if isinstance(plot_range, (int, float)):
        x_min = -plot_range
        y_min = -plot_range
        x_max = plot_range
        y_max = plot_range
    elif len(plot_range) == 2:
        if isinstance(plot_range[0], (int, float)):
            x_min = -plot_range[0]
            y_min = -plot_range[1]
            x_max = plot_range[0]
            y_max = plot_range[1]
        else:
            x_min = plot_range[0][0]
            y_min = plot_range[1][0]
            x_max = plot_range[0][1]
            y_max = plot_range[1][1]

    xpoints = jnp.arange(x_min, x_max, precision)
    ypoints = jnp.arange(y_min, y_max, precision)

    ygrid, xgrid = jnp.meshgrid(ypoints, xpoints)
    pos = jnp.stack((xgrid, ygrid), axis=-1).reshape(-1, 2)
    pos_shape = pos.shape

    outputs = f(pos)

    assert isinstance(outputs, Array)
    assert outputs.shape[0] == pos_shape[0]
    assert outputs.ndim <= 2

    if outputs.ndim == 2 and outputs.shape[1] == 1:
        outputs = jnp.squeeze(outputs, axis=-1)
    elif outputs.ndim == 2:
        assert outputs.shape[1] == 4

    # Output is a scalar value per point - convert to RGBA using colormap
    if outputs.ndim == 1:
        if cmap_range is None:
            cmap_range = (None, None)
        outputs = x_to_rgb_colormap(
            outputs,
            low=cmap_range[0],
            high=cmap_range[1],
            alpha=cmap_alpha,
            cmap_name=cmap_name,
        )

    img = outputs.reshape(xgrid.shape[0], xgrid.shape[1], outputs.shape[-1])
    img = img * 255
    img = jnp.transpose(img, (1, 0, 2))

    return Image(img, x=x_min, y=y_min, scale=precision)


def make_ellipse(
    radius_x: float = 10,
    radius_y: float = 5,
    res: int = 30,
    filled: bool = True,
    angle: float = 2 * math.pi,
) -> FilledPolygon | PolyLine:
    """Create an ellipse geometry.

    Args:
        radius_x: X radius
        radius_y: Y radius
        res: Resolution (number of points)
        filled: Whether to fill the ellipse
        angle: Angle to sweep (default full ellipse)

    Returns:
        Ellipse geometry
    """
    points = []
    for i in range(res):
        ang = -angle / 2 + angle * i / res
        points.append((math.cos(ang) * radius_x, math.sin(ang) * radius_y))
    if angle % (2 * math.pi) != 0:
        points.append((0, 0))

    points = jnp.array(points)
    return FilledPolygon(points) if filled else PolyLine(points, True)
