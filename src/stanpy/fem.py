import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
import stanpy as stp


def extract_nodes(**s):
    nodes_array = np.array([value for key, value in s.items() if 'X' in key])
    return nodes_array  # column 0: X, column 1: Y, column 2: Z


def assembly(*s_list, deg_free_node=2):
    number_elements = np.sum([s.get("num_elements", 1) for s in s_list])
    nodes_per_element = extract_nodes(**s_list[0]).shape[0]  # todo: make variable
    deg_free_element = nodes_per_element * deg_free_node
    nodes = np.zeros((number_elements * nodes_per_element, 3))
    index = 0
    # create nodes array
    for s in s_list:
        num_elements = s.get("num_elements", 1) + 1
        lspace = np.linspace(s["Xi"], s["Xk"], num_elements, axis=0)
        nodes_element = np.empty((lspace.shape[0] + lspace[1:-1].shape[0], 3), dtype=lspace.dtype)
        nodes_element[0::2] = lspace[:-1]
        nodes_element[1:-1:2] = lspace[1:-1]
        nodes_element[-1] = lspace[-1]
        nodes[index : index + nodes_element.shape[0]] = nodes_element
        index += nodes_element.shape[0] + 1
    global_nodes = np.unique(nodes, axis=0)

    num_global_nodes = global_nodes.shape[0]
    a_mat = np.zeros((number_elements, deg_free_element, num_global_nodes * deg_free_node))

    for element in range(number_elements):
        for node in range(nodes_per_element):
            index = element * nodes_per_element + node
            pos = np.array(np.where(np.equal(nodes[index], global_nodes).all(axis=1)), dtype=int).flatten().item() * 2
            a_mat[
                element, node * nodes_per_element : (1 + node) * nodes_per_element, pos : pos + deg_free_node
            ] = np.eye(deg_free_node, deg_free_node)

    return a_mat


def length_from_nodes(**s):
    vec_i = np.array(s["Xi"])
    vec_k = np.array(s["Xk"])
    vec_R = vec_k - vec_i
    return np.linalg.norm(vec_R)


def interpolation_function_deg4(x, **s):
    num_elements = s.get("num_elements", 1)
    l = length_from_nodes(**s) / num_elements
    return np.array(
        [
            [
                1 / 2 - 3 * x / (2 * l) + 2 * x**3 / l**3,
                -l / 8 + x / 4 + x**2 / (2 * l) - x**3 / l**2,
                1 / 2 + 3 * x / (2 * l) - 2 * x**3 / l**3,
                l / 8 + x / 4 - x**2 / (2 * l) - x**3 / l**2,
            ]
        ]
    )


def stiffness_matrix_element_rect(**s):
    b = s.get("b")
    h = s.get("h")
    num_elements = s.get("num_elements", 1)
    l = length_from_nodes(**s) / num_elements
    return np.array(
        [
            [b * h**3 / l**3, -b * h**3 / (2 * l**2), -b * h**3 / l**3, -b * h**3 / (2 * l**2)],
            [-b * h**3 / (2 * l**2), b * h**3 / (3 * l), b * h**3 / (2 * l**2), b * h**3 / (6 * l)],
            [-b * h**3 / l**3, b * h**3 / (2 * l**2), b * h**3 / l**3, b * h**3 / (2 * l**2)],
            [-b * h**3 / (2 * l**2), b * h**3 / (6 * l), b * h**3 / (2 * l**2), b * h**3 / (3 * l)],
        ]
    )


def system_eqation(**s):
    a = stp.assembly(s, deg_free_node=2)
    aT = np.transpose(a, axes=(0, 2, 1))
    K_local = stp.stiffness_matrix_element_rect(**s)
    K_global = np.sum(np.matmul(aT.dot(K_local), a), axis=0)
    p_local = stp.node_load_vector(**s)
    p_global = np.sum(aT.dot(p_local), axis=0)
    bc_i = s.get("bc_i")
    bc_k = s.get("bc_k")
    if bc_i.get("w", None) == 0:
        K_global[0, :] = 0
        K_global[0, 0] = 1
        p_global[0] = 0
    if bc_i.get("phi", None) == 0:
        K_global[1, :] = 0
        K_global[1, 1] = 1
        p_global[1] = 0
    if bc_k.get("w", None) == 0:
        K_global[-2, :] = 0
        K_global[-2, -2] = 1
        p_global[-2] = 0
    if bc_k.get("phi", None) == 0:
        K_global[-1, :] = 0
        K_global[-1, -1] = 1
        p_global[-1] = 0

    return K_global, p_global


def node_load_vector(**s):
    q = s.get("q")
    num_elements = s.get("num_elements", 1)
    l = length_from_nodes(**s) / num_elements
    return np.array([l * q / 2, -(l**2) * q / 12, l * q / 2, l**2 * q / 12])


if __name__ == "__main__":
    np.set_printoptions(precision=6, linewidth=500)
    node1 = (0, 0, 0)
    node2 = (3, 0, 0)
    # node3 = (3, 0, 0)
    s = {
        "b": 1,
        "h": 1,
        "q": 1,
        "Xi": node1,
        "Xk": node2,
        "num_elements": 3,
        "bc_i": {"w": 0, "phi": 0},
        "bc_k": {"w": 0, "M": 0},
    }

    a = stp.assembly(s, deg_free_node=2)
    K, p = system_eqation(**s)
    q = np.linalg.pinv(K).dot(p).round(10)
    N = interpolation_function_deg4(sym.Symbol("x"), **s)
    print(N.dot(a[0].dot(q)))
    print(N.dot(a[1].dot(q)))
    print(N.dot(a[2].dot(q)))
    # print(N[0].dot(a[0].dot(q)))
    # print()
