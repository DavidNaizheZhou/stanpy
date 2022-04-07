import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
import stanpy as stp

x_sym = sym.Symbol("x")
E = 3e7  # kN/m2
b = 0.2  # m
ha = hb = 0.3  # m
hc = 0.4  # m
l1 = 4  # m
l2 = 3  # m
hx = ha + (hc - hb) / l2 * x_sym

cs_props1 = stp.cs(b=b, h=ha)
s1 = {"E": E, "cs": cs_props1, "q": 10, "l": l1, "bc_i": {"w": 0, "M": 0, "H": 0}}

cs_props2 = stp.cs(b=b, h=hx)
s2 = {"E": E, "cs": cs_props2, "q": 10, "l": l2, "bc_k": {"w": 0, "phi": 0}}

s = [s1,s2]

fig, ax = plt.subplots(figsize=(12, 5))
stp.plot_system(ax, *s, render=True, facecolor="gray", alpha=0.5, render_scale=0.4)
stp.plot_load(ax, *s)
ax.grid(linestyle=":")
ax.set_axisbelow(True)
ax.set_ylim(-0.75, 1.0)
plt.show()