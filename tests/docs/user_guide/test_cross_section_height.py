def test_ex_01():

    import sympy as sym
    import stanpy as stp
    import matplotlib.pyplot as plt

    x = sym.Symbol("x")
    l = 4  # m
    b, ha, hb = 0.2, 0.3, 0.4  # m
    hx = ha + (hb - ha) / l * x  # m

    cs_pros = stp.cs(b=b, h=hx)

    print(cs_pros)

    fig, ax = plt.subplots(figsize=(8, 2))

    s = {"l": l}
    stp.plot_system(ax, s)
    plt.show()


if __name__ == "__main__":
    test_ex_01()
