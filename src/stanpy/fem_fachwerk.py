import numpy as np
import matplotlib.pyplot as plt
import stanpy as stp

np.set_printoptions(precision=5, linewidth=500)


def assembly(*s_list, deg_freedom=3):
    number_elements = len(s_list)
    nodes = np.zeros((2 * number_elements, 3))
    nodes[0::2, :] = np.array([s["Xi"] for s in s_list]).astype(int)
    nodes[1::2, :] = np.array([s["Xk"] for s in s_list]).astype(int)
    global_nodes = np.unique(nodes, axis=0)
    num_global_nodes = global_nodes.shape[0]
    indices = (np.arange(num_global_nodes) * deg_freedom).astype(int)
    a = np.zeros((number_elements, 2, num_global_nodes))
    a_full = np.zeros((number_elements, 2 * deg_freedom, num_global_nodes * deg_freedom))
    for i, node in enumerate(nodes.reshape(number_elements, -1, 3)):
        a[i, 0] = (global_nodes == node[0]).all(axis=1).astype(int)
        a[i, 1] = (global_nodes == node[1]).all(axis=1).astype(int)
        mask = a[i, 0] == 1
        a_full[i, 0:deg_freedom, indices[mask].item() : indices[mask].item() + deg_freedom] = np.eye(
            deg_freedom, deg_freedom
        )
        mask = a[i, 1] == 1
        a_full[
            i,
            deg_freedom : 2 * deg_freedom,
            indices[mask].item() : indices[mask].item() + deg_freedom,
        ] = np.eye(deg_freedom, deg_freedom)

    return a_full


# def assembly_univ(*elements, deg_freedom=3):
#     number_elements = len(elements)
#     nodes = np.zeros((2 * number_elements, 3)) # 3Dimensional
#     nodes[0::2, :] = np.array([s["Xi"] for s in elements])
#     nodes[1::2, :] = np.array([s["Xk"] for s in elements])
#     global_nodes = np.unique(nodes, axis=0)
#     num_global_nodes = global_nodes.shape[0]
#     indices = (np.arange(num_global_nodes) * deg_freedom).astype(int)
#     a = np.zeros((number_elements, 2, num_global_nodes))
#     a_full = np.zeros((number_elements, 2 * deg_freedom, num_global_nodes * deg_freedom))
#     for i, node in enumerate(nodes.reshape(number_elements, -1, 3)):
#         a[i, 0] = (global_nodes == node[0]).all(axis=1).astype(int)
#         a[i, 1] = (global_nodes == node[1]).all(axis=1).astype(int)
#         mask = a[i, 0] == 1
#         a_full[i, 0:deg_freedom, indices[mask].item() : indices[mask].item() + deg_freedom] = np.eye(
#             deg_freedom, deg_freedom
#         )
#         mask = a[i, 1] == 1
#         a_full[
#             i,
#             deg_freedom : 2 * deg_freedom,
#             indices[mask].item() : indices[mask].item() + deg_freedom,
#         ] = np.eye(deg_freedom, deg_freedom)

#     return a_full


def element_stiffness_matrix(**s):
    R = rotation_matrix(**s)
    vec_i = np.array(s["Xi"])
    vec_k = np.array(s["Xk"])
    vec_R = vec_k - vec_i
    QeT = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]).dot(R)
    l = np.linalg.norm(vec_R).item()
    K = s["EA"] / l * QeT.T.dot(np.array([[1, -1], [-1, 1]])).dot(QeT)
    return K


def system_stiffness_matrix(*s_list):
    a = assembly(*s_list)
    K = a[0].T.dot(element_stiffness_matrix(**s_list[0])).dot(a[0])
    for i, s in enumerate(s_list):
        if i == 0:
            pass
        else:
            K += a[i].T.dot(element_stiffness_matrix(**s_list[i])).dot(a[i])
    K[0, :] = 0
    diag = np.copy(np.diag(K))
    diag[diag == 0] = 1
    np.fill_diagonal(K, diag)
    return K


def rotation_matrix(**s):
    vec_i = np.array(s["Xi"])
    vec_k = np.array(s["Xk"])
    vec_R = vec_k - vec_i
    norm_R = np.linalg.norm(vec_R)

    theta = np.radians(90)
    c, s = np.cos(theta), np.sin(theta)
    rot_mat = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))

    e1 = vec_R / norm_R
    e2 = rot_mat.dot(e1)
    e3 = np.cross(e1, e2)

    e_global = np.eye(3, 3)

    R = np.zeros((6, 6))
    R[0, :3] = e1.dot(e_global)
    R[1, :3] = e2.dot(e_global)
    R[2, :3] = e3.dot(e_global)
    R[3, 3:] = e1.dot(e_global)
    R[4, 3:] = e2.dot(e_global)
    R[5, 3:] = e3.dot(e_global)

    return R


def cosine_alpha(nod):
    pass


if __name__ == "__main__":
    L = 1

    roller = {"w": 0, "M": 0, "H": 0}
    hinged = {"w": 0, "M": 0}

    node1 = (0, 0, 0)
    node2 = (L, 0, 0)
    node3 = (2 * L, 0, 0)
    node4 = (3 * L, 0, 0)

    s1 = {"EA": 1, "Xi": node1, "Xk": node2, "bc_i": hinged}
    s2 = {"EA": 1, "Xi": node2, "Xk": node3, "bc_i": roller}
    s3 = {"EA": 1, "Xi": node3, "Xk": node4, "bc_i": roller, "bc_k": roller}

    s = [s1, s2, s3]

    R = rotation_matrix(**s1)
    K = system_stiffness_matrix(*s)

    P = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]).reshape(-1, 1)
    q = np.linalg.pinv(K).dot(P).round(10)

    print(q)

    # fig, ax = plt.subplots()
    # stp.plot_fachwerk(ax, s1, s2, s3, df=False)
    # ax.set_aspect("equal")
    # ax.set_ylim(-2, 2)
    # plt.show()
