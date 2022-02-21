from __future__ import annotations

__all__ = [
    "plot_system",
    "plot_M",
    "plot_V",
    "plot_R",
    "plot_phi",
    "plot_w",
    "plot_load",
    "plot_roller_support",
    "plot_fixed_support",
    "plot_hinged_support",
    "plot_w_0",
    "plot_cs",
]

import copy
import os

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox
from matplotlib.offsetbox import OffsetImage
from matplotlib.patches import PathPatch
from matplotlib.patches import Rectangle
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path
from PIL import Image

import numpy as np
import sympy as sym
import stanpy as stp

import importlib.resources as pkg_resources

# Markers ################################
verts = [(0, 0), (5, -5), (-5, -5), (0, 0)]
codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
festes_auflager = Path(verts, codes)

verts = [(0, 0), (5, -5), (-5, -5), (0, 0), (5, -7), (-5, -7)]
codes = [
    Path.MOVETO,
    Path.LINETO,
    Path.LINETO,
    Path.CLOSEPOLY,
    Path.MOVETO,
    Path.LINETO,
]
versch_auflager = Path(verts, codes)

verts = [(0, 5), (0, -5), (0.5, -5), (0.5, 5), (0, 5)]
codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
fixed_support_path_right = Path(verts, codes)

verts = [(0, 5), (0, -5), (-0.5, -5), (-0.5, 5), (0, 5)]
codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
fixed_support_path_left = Path(verts, codes)

distance_to_line = -6
verts = [
    (-12.5, distance_to_line),
    (-7.5, distance_to_line),
    (-2.5, distance_to_line),
    (2.5, distance_to_line),
    (7.5, distance_to_line),
    (12.5, distance_to_line),
]
codes = [Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO]
definitionsfaser = Path(verts, codes)

distance_to_line = 40
verts = [
    (0, distance_to_line + 40),
    (0, distance_to_line),
    (0 - 6, distance_to_line + 6),
    (0, distance_to_line),
    (0 + 6, distance_to_line + 6),
]
codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.MOVETO, Path.LINETO]
load_path = Path(verts, codes)


def watermark(ax, watermark_pos=4):
    # if "user_guide" in os.listdir():
    #     img = Image.open(os.path.join("user_guide", "static", "stanpy_logo_2-removebg.png"))
    # else:
    #     img = Image.open(os.path.join("src", "stanpy", "static", "stanpy_logo_2-removebg.png"))
    with pkg_resources.path(stp.static, 'stanpy_logo_2-removebg.png') as path:
        img = Image.open(path)

    width = ax.bbox.xmax - ax.bbox.xmin
    wm_width = int(width / 16)  # make the watermark 1/4 of the figure size
    scaling = wm_width / float(img.size[0])
    wm_height = int(float(img.size[1]) * float(scaling))
    img = img.resize((wm_width, wm_height), Image.ANTIALIAS)

    imagebox = OffsetImage(img, zoom=1, alpha=0.8)
    imagebox.image.axes = ax

    ao = AnchoredOffsetbox(watermark_pos, pad=0.01, borderpad=0, child=imagebox)
    ao.patch.set_alpha(0)
    ax.add_artist(ao)


def plot_beam(ax, xi, yi, xk, yk, **kwargs):
    df = kwargs.pop("df", True)  # definitionsfaser
    ax.plot(
        (xi, xk),
        (yi, yk),
        c=kwargs.pop("c", "black"),
        lw=kwargs.pop("lw", 2),
        fillstyle=kwargs.pop("fillstyle", "none"),
        zorder=kwargs.pop("zorder", 1),
        linestyle=kwargs.pop("linestyle", "-"),
        **kwargs,
    )
    if df:
        ax.scatter(
            [xi + np.abs(xi - xk) / 2],
            [yi],
            marker=definitionsfaser,
            s=20**2,
            edgecolors=kwargs.pop("c", "black"),
            facecolors=kwargs.pop("c", "black"),
            zorder=2,
            **kwargs,
        )


def plot_roller_support(ax, x, y, **kwargs):
    ax.scatter(
        [x],
        [y],
        marker=versch_auflager,
        s=30**2,
        edgecolors="black",
        facecolors="white",
    )  # auflager
    ax.scatter(
        [x],
        [y],
        marker="o",
        s=kwargs.get("hinge_size", 20),
        edgecolors="black",
        facecolors="white",
        zorder=2,
    )


def plot_roller_support_half(ax, x, y, **kwargs):
    ax.scatter(
        [x],
        [y],
        marker=versch_auflager,
        s=30**2,
        edgecolors="black",
        facecolors="white",
    )


def plot_fixed_support(ax, x, y, side, **kwargs):
    if side == "right":
        ax.scatter(
            [x],
            [y],
            marker=fixed_support_path_right,
            s=22**2,
            edgecolors="black",
            facecolors="black",
        )
    elif side == "left":
        ax.scatter(
            [x],
            [y],
            marker=fixed_support_path_left,
            s=22**2,
            edgecolors="black",
            facecolors="black",
        )


def plot_hinged_support(ax, x, y, **kwargs):
    ax.scatter(
        [x],
        [y],
        marker=festes_auflager,
        s=22**2,
        edgecolors="black",
        facecolors="white",
    )
    ax.scatter(
        [x],
        [y],
        marker="o",
        s=kwargs.get("hinge_size", 20),
        edgecolors="black",
        facecolors="white",
        zorder=2,
    )


def plot_hinge(ax, x, y, **kwargs):
    ax.scatter(
        [x],
        [y],
        marker="o",
        s=kwargs.get("hinge_size", 20),
        edgecolors="black",
        facecolors="white",
        zorder=2,
    )


def plot_uniformly_distributed_load(ax, xi, yi, xk, yk, **kwargs):
    x_arrows = np.linspace(xi, xk, 20)
    ax.scatter(
        x_arrows,
        [yi] * x_arrows.size,
        marker=load_path,
        s=(50) ** 2,
        edgecolors=kwargs.pop("c", "black"),
        facecolors="none",
        zorder=2,
    )


def plot_uniformly_distributed_load_patch(ax, xi, yi, xk, yk, q, **kwargs):
    l = np.abs(xk - xi)
    ax.add_patch(Rectangle((xi, 0.25), l, 0.2 * q, fill=None, hatch="|"))  # gleichlast


# def plot_point_load(ax,x,y,xj,P,**kwargs):
#   ax.scatter(x+xj,y, marker=load_path,
#   s=(50*P)**2,
#   edgecolors=kwargs.pop('c', "black"),
#   facecolors="none",
#   zorder=2)
# ax.scatter([xi],[yi], marker=uniformly_distributed_load_path, s=20**2,  edgecolors='black',facecolors='white', zorder=2)


def plot_point_load(ax, x, y, magnitude, **kwargs):
    dy_P = kwargs.pop("dy_P", 0)
    ax.arrow(
        x,
        y + magnitude / 2 + dy_P,
        0,
        -magnitude / 2,
        head_width=0.05,
        color=kwargs.get("c", "black"),
    )


def plot_point_load_H(ax, x, y, magnitude, **kwargs):
    dx_N = kwargs.pop("dx_N", 0.1)
    dy_N = kwargs.pop("dy_N", 0)
    scale = kwargs.pop("scale_N", 1)
    direction = kwargs.pop("direction", "left")
    if direction == "left":
        ax.arrow(
            x + scale + dx_N,
            y + dy_N,
            -scale,
            0,
            head_width=0.05,
            color=kwargs.get("c", "black"),
        )
    elif direction == "right":
        ax.arrow(
            x - scale - dx_N,
            y + dy_N,
            scale,
            0,
            head_width=0.05,
            color=kwargs.get("c", "black"),
        )


# def plot_slab(ax,xi,yi,**kwargs):
#   kwargs.pop('c', "black")
#   ax.plot(xi,yi, lw=3, fillstyle='none', zorder=1, linestyle="-", **kwargs)
#   ax.scatter([xi[0]+np.abs(xi[0]-xi[1])/2],[yi[0]], marker=definitionsfaser, s=20**2,  edgecolors='black',facecolors='white', zorder=2)


def plot_system(ax, *args, **kwargs):
    xi = 0
    yi = 0
    xk = 0
    yk = 0
    support = kwargs.pop("support", True)
    hinge_size = kwargs.pop("hinge_size", 22)
    watermark_pos = kwargs.pop("watermark_pos", 4)
    plot_watermark = kwargs.pop("watermark", True)
    for s in args:
        if "l" in s.keys():
            l: float = s["l"]
            xk = xi + l
            yk = yi
            plot_beam(ax, xi, yi, xk, yk, **kwargs)
        if support:
            if "bc_i" in s.keys():
                if all([boundary_condition in s["bc_i"].keys() for boundary_condition in ["w", "M", "H"]]):
                    plot_roller_support(ax, xi, yi, hinge_size=hinge_size)
                elif all([boundary_condition in s["bc_i"].keys() for boundary_condition in ["w", "M"]]):
                    plot_hinged_support(ax, xi, yi, hinge_size=hinge_size)
                elif all([boundary_condition in s["bc_i"].keys() for boundary_condition in ["w", "phi"]]):
                    plot_fixed_support(ax, xi, yi, "left")
                elif all([boundary_condition in s["bc_i"].keys() for boundary_condition in ["M", "V"]]):
                    pass
                elif all([boundary_condition in s["bc_i"].keys() for boundary_condition in ["w"]]):
                    plot_roller_support_half(ax, xi, yi, hinge_size=hinge_size)
                elif all([boundary_condition in s["bc_i"].keys() for boundary_condition in ["M"]]):
                    plot_hinge(ax, xi, yi, hinge_size=hinge_size)

            if "bc_k" in s.keys():
                if all([boundary_condition in s["bc_k"].keys() for boundary_condition in ["w", "M", "H"]]):
                    plot_roller_support(ax, xk, yk, hinge_size=hinge_size)
                elif all([boundary_condition in s["bc_k"].keys() for boundary_condition in ["w", "M"]]):
                    plot_hinged_support(ax, xk, yk, hinge_size=hinge_size)
                elif all([boundary_condition in s["bc_k"].keys() for boundary_condition in ["w", "phi"]]):
                    plot_fixed_support(ax, xk, yk, "right")
                elif all([boundary_condition in s["bc_k"].keys() for boundary_condition in ["M", "V"]]):
                    pass
                elif all([boundary_condition in s["bc_k"].keys() for boundary_condition in ["w"]]):
                    plot_roller_support_half(ax, xk, yk, hinge_size=hinge_size)
                elif all([boundary_condition in s["bc_k"].keys() for boundary_condition in ["M"]]):
                    plot_hinge(ax, xk, yk, hinge_size=hinge_size)

        xi = xk
        yi = yk

        ax.set_axisbelow(True)

        if plot_watermark:
            watermark(ax, watermark_pos)


def plot_load(ax, *args, **kwargs):
    xi = 0
    yi = 0
    xk = 0
    yk = 0

    q_offset = kwargs.pop("q_offset", 0)
    dy_P = kwargs.pop("dy_P", 0)
    dx_N = kwargs.pop("dx_N", 0.5)
    dy_N = kwargs.pop("dy_N", 0)
    P_scale = kwargs.pop("P_scale", 1)
    N_scale = kwargs.pop("N_scale", 1)

    offset0_positiv = offset_positiv = copy.copy(kwargs.pop("offset", 0.3))
    offset0_negativ = offset_negativ = -copy.copy(kwargs.pop("offset", 0.3))
    q_array = np.zeros(len(args))
    P_array_all = np.array([])
    for i, s in enumerate(args):
        if "q" in s.keys():
            q_array[i] = s["q"]
        P_array_all = np.append(P_array_all, stp.extract_P_from_beam(**s))
    P_array_all = P_array_all.flatten()
    for i, s in enumerate(args):
        P_array = stp.extract_P_from_beam(**s)
        N_array = stp.extract_N_from_beam(**s)
        l = s["l"]
        xk = xi + l
        yk = yi
        dl = 0.25
        if "q" in s.keys():
            x = np.linspace(xi, xk, int(l / dl))
            q_height = q_array[i] / np.max(np.abs(q_array)) / 4
            if q_array[i] > 0:
                plot_load_distribution(
                    ax,
                    x,
                    np.ones(x.size) * offset_positiv,
                    q_height,
                    q_offset=q_offset,
                    **kwargs,
                )
                offset_positiv += offset_positiv + q_height
            elif q_array[i] < 0:
                offset_negativ += offset_negativ - q_height
                plot_load_distribution(
                    ax,
                    x,
                    np.ones(x.size) * offset_negativ,
                    q_height,
                    q_offset=q_offset,
                    **kwargs,
                )

        if P_array.shape[0]:
            # plot_point_load(ax,xi,yi,s["P"][1],s["P"][0]/np.max(p_array))
            for i in range(P_array.shape[0]):
                if P_array[i][0] > 0:
                    plot_point_load(
                        ax,
                        xi + P_array[i][1],
                        offset_positiv,
                        P_array[i][0] / np.max(P_array_all) * P_scale,
                        dy_P=dy_P,
                        **kwargs,
                    )
                elif P_array[i][0] < 0:
                    plot_point_load(
                        ax,
                        xi + P_array[i][1],
                        offset_negativ,
                        -P_array[i][0] / np.max(P_array_all) * P_scale,
                        dy_P=dy_P,
                        **kwargs,
                    )

        if N_array.shape[0]:
            # plot_point_load(ax,xi,yi,s["P"][1],s["P"][0]/np.max(p_array))
            for i in range(P_array.shape[0]):
                if P_array[i][0] > 0:
                    plot_point_load_H(
                        ax,
                        xi + l,
                        0,
                        N_array[i] / 4 * N_scale,
                        dx_N=dx_N,
                        dy_N=dy_N,
                        **kwargs,
                    )
                elif P_array[i][0] < 0:
                    plot_point_load_H(
                        ax,
                        xi + l,
                        0,
                        N_array[i] / 4 * N_scale,
                        dx_N=dx_N,
                        dy_N=dy_N,
                        **kwargs,
                    )

        xi = xk
        yi = yk
        offset_positiv = offset0_positiv
        offset_negativ = offset0_negativ


def plot_load_distribution(ax, x, y, thickness, q_offset=0, **kwargs):
    lower_line = y + q_offset
    upper_line = y + q_offset + thickness

    size_arrow_head = thickness / 4

    code_arrows = np.zeros((x.size, 5, 1))
    code_arrows[:, :, 0] = np.array([Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO])
    code_arrows = code_arrows.flatten()
    arrows = np.zeros((x.size, 5, 2))
    arrows[:, 0, 0] = x
    arrows[:, 1, 0] = x
    arrows[:, 2, 0] = x + size_arrow_head / 3
    arrows[:, 3, 0] = x - size_arrow_head / 3
    arrows[:, -1, 0] = x
    arrows[:, 0, 1] = upper_line
    arrows[:, 1, 1] = lower_line
    arrows[:, -1, 1] = lower_line
    arrows[:, 2, 1] = lower_line + size_arrow_head
    arrows[:, 3, 1] = lower_line + size_arrow_head
    arrows = arrows.flatten().reshape(-1, 2)
    vertices = np.zeros((2 * x.size + arrows.shape[0], 2))
    vertices[: 2 * x.size, :] = np.block([[x, x], [lower_line, upper_line]]).T
    vertices[2 * x.size :, :] = arrows
    codes = np.zeros(2 * x.size + arrows.shape[0])
    codes[: 2 * x.size] = np.full(2 * x.size, Path.LINETO)
    codes[0] = codes[x.size] = Path.MOVETO
    codes[2 * x.size :] = code_arrows
    # codes = np.full(len(vertices), Path.LINETO)

    # path = Path(arrow, codes)
    path = Path(vertices, codes)
    ax.add_patch(PathPatch(path, facecolor=kwargs.pop("facecolor", "black"), **kwargs))


def plot_V(ax, x: np.ndarray = np.array([]), Vx: np.ndarray = np.array([]), **kwargs):
    plot_R(ax=ax, x=x, Rx=Vx, **kwargs)


def plot_R(ax, x: np.ndarray = np.array([]), Rx: np.ndarray = np.array([]), **kwargs):

    x_plot = np.zeros(x.size + 2)
    Rx_plot = np.zeros(x.size + 2)
    x_plot[1:-1] = x
    x_plot[-1] = x[-1]

    scale = kwargs.pop("scale", 1)
    Rx_plot[1:-1] = Rx / np.max(np.abs(Rx)) * scale

    hatch = kwargs.pop("hatch", "")
    fill_p = kwargs.pop("fill_p", None)
    fill_n = kwargs.pop("fill_n", None)
    fill = kwargs.pop("fill", None)
    alpha = kwargs.pop("alpha", 1)
    zorder = kwargs.pop("zorder", 1)
    c = kwargs.pop("c", "black")
    lw = kwargs.pop("lw", 1)

    annotate_x = kwargs.pop("annotate_x", np.array([]))
    annotate_text = kwargs.pop("annotate_text", np.array([]))
    annotate_round = kwargs.pop("annotate_round", 3)
    if isinstance(annotate_x, int) or isinstance(annotate_x, float):
        annotate_x = np.array([annotate_x]).flatten()
    if isinstance(annotate_text, list):
        annotate_x = np.array(annotate_text).flatten()

    if fill != None:
        ax.fill_between(
            x_plot,
            Rx_plot,
            where=Rx_plot > 0,
            y2=0,
            hatch=hatch,
            zorder=zorder,
            fc=fill,
            alpha=alpha,
        )
        ax.fill_between(
            x_plot,
            Rx_plot,
            where=Rx_plot < 0,
            y2=0,
            hatch=hatch,
            zorder=zorder,
            fc=fill,
            alpha=alpha,
        )
    else:
        if fill_p != None:
            ax.fill_between(
                x_plot,
                Rx_plot,
                where=Rx_plot > 0,
                y2=0,
                hatch=hatch,
                zorder=zorder,
                fc=fill_p,
                alpha=alpha,
            )
        if fill_n != None:
            ax.fill_between(
                x_plot,
                Rx_plot,
                where=Rx_plot < 0,
                y2=0,
                hatch=hatch,
                zorder=zorder,
                fc=fill_n,
                alpha=alpha,
            )
    ax.plot(x_plot, Rx_plot, zorder=zorder, lw=lw, c=c, **kwargs)

    for i, annotation in enumerate(annotate_x):
        if isinstance(annotation, list) or isinstance(annotation, tuple):
            annotation_list = []
            for annotation_i in sorted(annotation, reverse=True):
                mask_plot = x_plot == annotation_i
                mask = x == annotation_i
                if mask.any():
                    annotation_list.append(np.round(np.max(Rx[mask]), annotate_round))
                else:
                    raise ValueError("annotate_x value = {} is not in x".format(annotation_i))

                # annotation_list.append(np.round(np.max(Rx[mask]),annotate_round))
            ax.vlines(
                np.max(annotation),
                0,
                np.max(Rx_plot[mask_plot]),
                color=c,
                **kwargs,
                lw=1,
            )
            annotation_list = np.sort(np.array(annotation_list))

            annotation_string = np.array2string(annotation_list, separator=",", suppress_small=True)
            annotation_pos_y = np.max(Rx_plot[mask_plot])
            if annotation_pos_y > 0:
                ax.annotate(
                    annotation_string,
                    (np.max(annotation), annotation_pos_y + 0.1),
                    ha="center",
                    va="bottom",
                    rotation=90,
                    transform=ax.transAxes,
                    clip_on=True,
                    annotation_clip=True,
                )
            else:
                annotation_pos_y = np.min(Rx_plot[mask_plot])
                ax.annotate(
                    annotation_string,
                    (np.max(annotation), annotation_pos_y - 0.1),
                    ha="center",
                    va="top",
                    rotation=90,
                    transform=ax.transAxes,
                    clip_on=True,
                    annotation_clip=True,
                )

        else:
            mask_plot = x_plot == annotation
            mask = x == annotation

            if mask.any():
                ax.vlines(annotation, 0, np.max(Rx_plot[mask_plot]), color=c, **kwargs, lw=1)

                annotation_pos_y = np.max(Rx_plot[mask_plot])
                if annotation_pos_y > 0:
                    ax.annotate(
                        np.round(np.max(Rx[mask]), annotate_round),
                        (annotation, annotation_pos_y + 0.1),
                        ha="center",
                        va="bottom",
                        rotation=90,
                        transform=ax.transAxes,
                        clip_on=True,
                        annotation_clip=True,
                    )
                else:
                    annotation_pos_y = np.min(Rx_plot[mask_plot])
                    ax.annotate(
                        np.round(np.max(Rx[mask]), annotate_round),
                        (annotation, annotation_pos_y - 0.1),
                        ha="center",
                        va="top",
                        rotation=90,
                        transform=ax.transAxes,
                        clip_on=True,
                        annotation_clip=True,
                    )
            else:
                raise ValueError("annotate_x value = {} is not in x".format(annotation))


def plot_M(ax, *args, **kwargs):
    for s in args:
        l = s["l"]
        mx = s["M_sol"]
        x = np.linspace(0, l, mx.size)
        x_plot = np.zeros(x.size + 2)
        mx_plot = np.zeros(x.size + 2)
        x_plot[1:-1] = x
        x_plot[-1] = x[-1]
        mx_plot[1:-1] = -mx / np.max(np.abs(mx))

        hatch = kwargs.pop("hatch", "")
        fill_p = kwargs.pop("fill_p", None)
        fill_n = kwargs.pop("fill_n", None)
        fill = kwargs.pop("fill", None)
        alpha = kwargs.pop("alpha", 1)
        zorder = kwargs.pop("zorder", 1)
        c = kwargs.pop("c", "black")
        lw = kwargs.pop("lw", 1)
        if fill != None:
            ax.fill_between(
                x_plot,
                mx_plot,
                where=mx_plot < 0,
                y2=0,
                hatch=hatch,
                zorder=zorder,
                fc=fill,
                alpha=alpha,
            )
            ax.fill_between(
                x_plot,
                mx_plot,
                where=mx_plot > 0,
                y2=0,
                hatch=hatch,
                zorder=zorder,
                fc=fill,
                alpha=alpha,
            )
        else:
            if fill_p != None:
                ax.fill_between(
                    x_plot,
                    mx_plot,
                    where=mx_plot < 0,
                    y2=0,
                    hatch=hatch,
                    zorder=zorder,
                    fc=fill_p,
                    alpha=alpha,
                )
            if fill_n != None:
                ax.fill_between(
                    x_plot,
                    mx_plot,
                    where=mx_plot > 0,
                    y2=0,
                    hatch=hatch,
                    zorder=zorder,
                    fc=fill_n,
                    alpha=alpha,
                )
        ax.plot(x_plot, mx_plot, zorder=zorder, lw=lw, c=c, **kwargs)


def plot_M(ax, x: np.ndarray = np.array([]), Mx: np.ndarray = np.array([]), **kwargs):
    x_plot = np.zeros(x.size + 2)
    mx_plot = np.zeros(x.size + 2)
    x_plot[1:-1] = x
    x_plot[-1] = x[-1]

    scale = kwargs.pop("scale", 1)
    mx_plot[1:-1] = -Mx / np.max(np.abs(Mx)) * scale

    hatch = kwargs.pop("hatch", "")
    fill_p = kwargs.pop("fill_p", None)
    fill_n = kwargs.pop("fill_n", None)
    fill = kwargs.pop("fill", None)
    alpha = kwargs.pop("alpha", 1)
    zorder = kwargs.pop("zorder", 1)
    c = kwargs.pop("c", "black")
    lw = kwargs.pop("lw", 1)

    annotate_x = kwargs.pop("annotate_x", np.array([]))
    annotate_text = kwargs.pop("annotate_text", np.array([]))
    annotate_round = kwargs.pop("annotate_round", 3)
    if isinstance(annotate_x, int) or isinstance(annotate_x, float):
        annotate_x = np.array([annotate_x]).flatten()
    if isinstance(annotate_text, list):
        annotate_x = np.array(annotate_text).flatten()

    if fill != None:
        ax.fill_between(
            x_plot,
            mx_plot,
            where=mx_plot < 0,
            y2=0,
            hatch=hatch,
            zorder=zorder,
            fc=fill,
            alpha=alpha,
        )
        ax.fill_between(
            x_plot,
            mx_plot,
            where=mx_plot > 0,
            y2=0,
            hatch=hatch,
            zorder=zorder,
            fc=fill,
            alpha=alpha,
        )
    else:
        if fill_p != None:
            ax.fill_between(
                x_plot,
                mx_plot,
                where=mx_plot < 0,
                y2=0,
                hatch=hatch,
                zorder=zorder,
                fc=fill_p,
                alpha=alpha,
            )
        if fill_n != None:
            ax.fill_between(
                x_plot,
                mx_plot,
                where=mx_plot > 0,
                y2=0,
                hatch=hatch,
                zorder=zorder,
                fc=fill_n,
                alpha=alpha,
            )
    ax.plot(x_plot, mx_plot, zorder=zorder, lw=lw, c=c, **kwargs)

    for i, annotation in enumerate(annotate_x):
        if isinstance(annotation, list) or isinstance(annotation, tuple):
            annotation_list = []
            for annotation_i in sorted(annotation, reverse=True):
                mask_plot = x_plot == annotation_i
                mask = x == annotation_i

                if mask.any():
                    annotation_list.append(np.round(np.max(Mx[mask])))
                else:
                    raise ValueError("annotate_x value = {} is not in x".format(annotation_i))

            annotation_list = np.sort(np.array(annotation_list))
            annotation_string = np.array2string(annotation_list, separator=",", suppress_small=True)
            ax.vlines(
                np.max(annotation),
                0,
                np.max(mx_plot[mask_plot]),
                color=c,
                **kwargs,
                lw=1,
            )
            annotation_pos_y = np.max(mx_plot[mask_plot])
            if annotation_pos_y < 0:
                ax.annotate(
                    annotation_string,
                    (np.max(annotation), annotation_pos_y - 0.1),
                    ha="center",
                    va="top",
                    rotation=90,
                    transform=ax.transAxes,
                    clip_on=True,
                    annotation_clip=True,
                )
            else:
                ax.annotate(
                    annotation_string,
                    (np.max(annotation), annotation_pos_y + 0.1),
                    ha="center",
                    va="bottom",
                    rotation=90,
                    transform=ax.transAxes,
                    clip_on=True,
                    annotation_clip=True,
                )
        else:
            mask_plot = x_plot == annotation
            mask = x == annotation
            if mask.any():
                ax.vlines(annotation, 0, np.max(mx_plot[mask_plot]), color=c, **kwargs, lw=1)
                annotation_pos_y = np.max(mx_plot[mask_plot])
                if annotation_pos_y < 0:
                    ax.annotate(
                        np.round(np.max(Mx[mask]), annotate_round),
                        (annotation, np.max(mx_plot[mask_plot]) - 0.1),
                        ha="center",
                        va="top",
                        rotation=90,
                        transform=ax.transAxes,
                        clip_on=True,
                        annotation_clip=True,
                    )
                else:
                    ax.annotate(
                        np.round(np.max(Mx[mask]), annotate_round),
                        (annotation, np.max(mx_plot[mask_plot]) + 0.1),
                        ha="center",
                        va="bottom",
                        rotation=90,
                        transform=ax.transAxes,
                        clip_on=True,
                        annotation_clip=True,
                    )
            else:
                raise ValueError("annotate_x value = {} is not in x".format(annotation))


def plot_psi_0(ax, s, **kwargs):
    x = kwargs.pop("x", 0)
    offset = kwargs.pop("offset", 0)
    l = s.get("l")
    lw = kwargs.pop("lw", 1)
    c = kwargs.pop("c", "black")
    scale = kwargs.pop("scale", 1)
    xplot = np.linspace(x, x + l, 100)
    w_0 = 0
    psi_0 = s.get("psi_0", 0)
    phivi = psi_0 + 4 * w_0 / l
    kappav = 8 * w_0 / l**2
    # phiv = phivi-x*kappav
    wv = xplot * phivi - xplot**2 / 2 * kappav
    ax.plot(xplot, np.ones(xplot.size) * offset, lw=lw, c=c, **kwargs)
    ax.plot(xplot, offset - wv / np.max(wv) * 0.5 * scale, c=c)


def plot_cs(ax, b_vec, h_vec, ysi=np.array([]), zsi=np.array([]), **kwargs):

    watermark_pos = kwargs.pop("watermark_pos", 4)
    plot_watermark = kwargs.pop("watermark", True)
    edgecolor = kwargs.pop("edgecolor", "black")
    facecolor = kwargs.pop("facecolor", "#111111")
    alpha = kwargs.pop("alpha", 0.3)
    hatch = kwargs.pop("hatch", None)
    lw = kwargs.pop("lw", 1)
    dy = kwargs.pop("dy", 0)
    dz = kwargs.pop("dz", 0)

    if isinstance(ysi, list):
        ysi = np.array(ysi).astype(float)
    if isinstance(zsi, list):
        zsi = np.array(zsi).astype(float)

    if isinstance(b_vec, (int, float, sym.core.numbers.Float)) and isinstance(
        h_vec, (int, float, sym.core.numbers.Float)
    ):
        ax.add_patch(
            Rectangle(
                (0 + dy, 0 + dz),
                b_vec,
                h_vec,
                facecolor=facecolor,
                edgecolor=edgecolor,
                lw=lw,
                hatch=hatch,
                alpha=alpha,
                **kwargs,
            )
        )
    else:
        for b, h, ys, zs in zip(b_vec, h_vec, ysi, zsi):
            ax.add_patch(
                Rectangle(
                    (ys - b / 2 + dy, zs - h / 2 + dz),
                    b,
                    h,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    lw=lw,
                    hatch=hatch,
                    alpha=alpha,
                    **kwargs,
                )
            )

    if plot_watermark:
        watermark(ax, watermark_pos)

    ax.set_axisbelow(True)


def plot_w_0(ax, s, **kwargs):
    x = kwargs.pop("x", 0)
    offset = kwargs.pop("dy", 0)
    l = s.get("l")
    lw = kwargs.pop("lw", 1)
    c = kwargs.pop("c", "black")
    scale = kwargs.pop("scale", 1)
    xplot = np.linspace(x, x + l, 100)
    w_0 = s.get("w_0", 0)
    psi_0 = 0
    phivi = psi_0 + 4 * w_0 / l
    kappav = 8 * w_0 / l**2
    # phiv = phivi-x*kappav
    wv = xplot * phivi - xplot**2 / 2 * kappav
    ax.plot(xplot, np.ones(xplot.size) * offset, lw=lw, c=c, **kwargs)
    ax.plot(xplot, offset - wv / np.max(np.abs(wv)) * 0.25 * scale, c=c)


def plot_w(ax, x: np.ndarray = np.array([]), wx: np.ndarray = np.array([]), **kwargs):
    x_plot = np.zeros(x.size + 2)
    wx_plot = np.zeros(x.size + 2)
    x_plot[1:-1] = x
    x_plot[-1] = x[-1]
    scale = kwargs.pop("scale", 1)
    wx_plot[1:-1] = -wx / np.max(np.abs(wx)) * scale
    zorder = kwargs.pop("zorder", 1)
    lw = kwargs.pop("lw", 2)
    c = kwargs.pop("c", "black")
    linestyle = kwargs.pop("linestyle", "-")
    ax.plot(x_plot, wx_plot, zorder=zorder, lw=lw, c=c, linestyle=linestyle, **kwargs)


def plot_phi(ax, x: np.ndarray = np.array([]), phix: np.ndarray = np.array([]), **kwargs):
    x_plot = np.zeros(x.size + 2)
    wx_plot = np.zeros(x.size + 2)
    x_plot[1:-1] = x
    x_plot[-1] = x[-1]
    scale = kwargs.pop("scale", 1)
    wx_plot[1:-1] = -phix / np.max(np.abs(phix)) * scale
    zorder = kwargs.pop("zorder", 2)
    lw = kwargs.pop("lw", 1)
    c = kwargs.pop("c", "black")
    ax.plot(x_plot, wx_plot, zorder=zorder, lw=lw, c=c, **kwargs)


if __name__ == "__main__":

    import numpy as np
    import sympy as sym
    import stanpy as stp

    x = sym.Symbol("x")
    l = 4  # m
    s, t = 0.012, 0.02  # m
    ba, bb, ha, hb = 0.3, 0.4, 0.3, 0.4  # m
    hx = ha + (hb - ha) / l * x  # m
    bx = ba + (bb - ba) / l * x  # m

    b_v = np.array([t, hx - 2 * t, t])
    h_v = np.array([bx, s, bx])
    zsi_v = stp.AH_z.dot(h_v)  # von OK
    ysi_v = stp.AH_y.dot(b_v)  # von Links

    cs_props = stp.cs(b=b_v, h=h_v, y_si=ysi_v, z_si=zsi_v)

    print(cs_props)
    print("I_y(0) = ", cs_props["I_y"](0))
    print("I_y(l) = ", cs_props["I_y"](l))

    b_v_fun = sym.lambdify(x, b_v, 'numpy')
    h_v_fun = sym.lambdify(x, h_v, 'numpy')
    zsi_v_fun = sym.lambdify(x, zsi_v, 'numpy')
    ysi_v_fun = sym.lambdify(x, ysi_v, 'numpy')

    offset = 0.2
    fig, ax = plt.subplots(1)
    stp.plot_cs(ax, b_v_fun(0), h_v_fun(0), ysi_v_fun(0), zsi_v_fun(0))
    stp.plot_cs(ax, b_v_fun(l), h_v_fun(l), ysi_v_fun(l), zsi_v_fun(l), dy=0.4)

    ax.grid(linestyle=":")
    ax.axis('equal')

    plt.show()
