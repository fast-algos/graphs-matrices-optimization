import math
import scipy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import typing as th
from tqdm import tqdm

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


class TwiceRamanujan:
    def __init__(
        self,
        graph: th.Union[nx.Graph, np.array],
        d: int,
        eps: float = 1e-2,
        fast=False,
        verbose=0,
    ):

        if not isinstance(graph, nx.Graph):
            graph = nx.from_numpy_matrix(graph)

        # The epsilon value used for numerical stuff
        self.eps = eps
        self.fast_settings = fast

        self.graph = graph

        # setup the other values
        self.d = d
        self.n = len(self.graph.nodes)
        self.verbose = verbose

        self.sparsified_graph = None

    def do_reduction(self):
        # get the laplacian of self.graph
        self.L = nx.laplacian_matrix(self.graph).todense()

        # get the pseudo-inverse of the Laplacian
        self.L_pinv = scipy.linalg.pinv(self.L)

        # do an eigendecomposition of the Laplacian
        eig_vals, eig_vecs = np.linalg.eigh(self.L_pinv)
        eig_vals[np.abs(eig_vals) < self.eps] = 0
        self.L_pinv_sqrt = eig_vecs @ np.diag(np.sqrt(eig_vals)) @ eig_vecs.T

        # Reduce to the matrix problem:

        # set the list of vectors
        self.edge_vectors = []
        self.edges = []
        self.Pi = np.zeros((self.n, self.n))
        # iterate over all the edges with weights
        for (a, b, c) in self.graph.edges.data("weight"):
            # set the weight to 1 if it is not set
            if c is None:
                self.graph[a][b]["weight"] = 1

            # set the weight
            w = self.graph[a][b]["weight"]

            # create a vector with one at a and -1 at b
            L_ab = np.zeros((self.n, 1))
            L_ab[a][0], L_ab[b][0] = 1, -1
            # add the corresponding vector to the list
            self.edge_vectors.append(math.sqrt(w) * self.L_pinv_sqrt @ L_ab)
            self.Pi += self.edge_vectors[-1] @ self.edge_vectors[-1].T
            self.edges.append((a, b))

    def upper_bound_function(self, A, u, delta_u, v):
        if self.fast_settings:
            inv_diff = scipy.linalg.pinv(self.Pi * (u + delta_u) - A)
            pot_diff = self.potential_upper(A, u) - self.potential_upper(A, u + delta_u)
            ret = v.T @ inv_diff @ inv_diff @ v / pot_diff - v.T @ inv_diff @ v
            ret = 1 / ret.item()
            return ret
        else:

            def check(s):
                crit1 = np.max(np.linalg.eigh(A + s * v @ v.T)[0]) < u + delta_u
                crit2 = self.potential_upper(
                    A + s * v @ v.T, u + delta_u
                ) <= self.potential_upper(A, u)
                return crit1 and crit2

            s = 1
            while check(s):
                s *= 2

            s_L, s_R = 0, s
            for _ in range(100):
                s = (s_L + s_R) / 2
                if check(s):
                    s_L = s
                else:
                    s_R = s
            return s_L

    def lower_bound_function(self, A, l, delta_l, v):
        if self.fast_settings:
            inv_diff = scipy.linalg.pinv(A - self.Pi * (l + delta_l))
            pot_diff = self.potential_lower(A, l + delta_l) - self.potential_lower(A, l)
            ret = (
                pot_diff
                / (v.T @ (inv_diff @ inv_diff - pot_diff * inv_diff) @ v).item()
            )
            return ret
        else:

            def check(s):
                crit1 = np.sort(np.linalg.eigh(A + s * v @ v.T)[0])[1] > l + delta_l
                crit2 = self.potential_lower(
                    A + s * v @ v.T, l + delta_l
                ) <= self.potential_lower(A, l)
                return crit1 and crit2

            s = 1
            while not check(s):
                s *= 2
                if s > 1e10:
                    return math.inf
            # print(s)
            # print(np.min(np.linalg.eigh(A + s * v @ v.T)[0]))
            # print(l + delta_l)
            s_L, s_R = 0, s
            for _ in range(100):
                s = (s_L + s_R) / 2
                if not check(s):
                    s_L = s
                else:
                    s_R = s
            return s_R

        # print("Delta lower: ", pot_diff)

        # lower_bound = (v.T @ (inv_diff @ inv_diff) @ v) / pot_diff - v.T @ (
        #     inv_diff
        # ) @ v
        # return lower_bound.item()

    def potential_upper(self, L, u: float):
        eigenValues, _ = np.linalg.eigh(L)
        ret = 0
        for val in eigenValues:
            ret += 1 / (u - val)
        # one of the eigenvalues would have been zero the following offsets
        ret -= 1.0 / (u - 0)
        return ret

    def potential_lower(self, L, l):
        eigenValues, _ = np.linalg.eigh(L)
        lower = 0
        for val in eigenValues:
            lower += 1 / (val - l)
        # one of the eigenvalues would have been zero the following offsets
        lower -= 1.0 / (0 - l)
        return lower

    def sanity_check(self, A, l, u):
        ret = True
        # check if the eigenvalues of A are between l and u
        eigenValues, _ = np.linalg.eigh(A)
        for val in eigenValues:
            if val < l or val > u:
                print(f"eigval {val} illegal and not between {l} and {u}")
                ret = False
        return ret

    def _check_in_range(self, A, B, C):
        a1 = np.min(np.linalg.eigh(B - A)[0])
        b1 = np.min(np.linalg.eigh(C - B)[0])
        if self.verbose >= 1:
            print("checking:")
            print(f"\tminimum eigenvalue of LHS: {a1}")
            print(f"\tminimum eigenvalue of RHS: {b1}")
        return a1 >= 0 and b1 >= 0

    def sparsify(self):
        self.do_reduction()

        # setup all the parameters
        d, n = self.d, self.n
        d_root = math.sqrt(d)
        delta_l = 1
        delta_u = (d_root + 1) / (d_root - 1)
        epsilon_l = 1 / d_root
        epsilon_u = (d_root - 1) / (d + d_root)
        l = -n / epsilon_l
        u = n / epsilon_u

        # A would be the idempotent matrix with eigenvalues equal to 1 or 0
        # print the eigenvalues of self.Pi

        A = np.zeros((n, n))

        # this scaling will ensure A is positive in all the other eigenvalues and
        # is strictly sandwiched between l and u

        self.sparsified_graph = nx.Graph()

        if self.verbose == 2:
            iterable = tqdm(range(d * n))
        else:
            iterable = range(d * n)

        scaling_factor = (math.sqrt(d) - 1) / (n * (d + 1) * math.sqrt(d))
        for _ in iterable:
            cand = (_, _, _, _, -math.inf)
            for e, v in zip(self.edges, self.edge_vectors):
                lb = self.lower_bound_function(A, l, delta_l, v)
                ub = self.upper_bound_function(A, u, delta_u, v)
                if ub - lb > 0 and ub - lb > cand[2]:
                    cand = (v, e, lb, ub, ub - lb)
                    if self.fast_settings:
                        # if set on fast setting then just pick the first edge
                        break
            if cand[-1] < 0:
                raise Exception("Cannot pick an edge!")
            else:
                v, e, lb, ub, _ = cand
                if lb <= ub:
                    s = (lb + ub) / 2
                    A = A + s * v @ v.T

                    a, b = e
                    w = self.graph[a][b]["weight"] * s * scaling_factor

                    # if edge already exists, add the weight
                    if self.sparsified_graph.has_edge(a, b):
                        self.sparsified_graph[a][b]["weight"] += w
                    else:
                        self.sparsified_graph.add_edge(a, b, weight=w)

                    if self.verbose > 2:
                        print(
                            f"picked the edge between {a} and {b} with weight {w:.2f}"
                        )
            l += delta_l
            u += delta_u

        # calculate the pseudo-inverse of self.L
        eig_vals, eig_vecs = np.linalg.eigh(self.L)
        eig_vals[eig_vals < self.eps] = 0
        L_sqrt = eig_vecs @ np.diag(np.sqrt(eig_vals)) @ eig_vecs.T
        self.accurate_laplacian = L_sqrt @ (scaling_factor * A) @ L_sqrt

        return self.sparsified_graph

    def juxtapose(self, with_verify=False):
        pos = nx.spring_layout(self.graph)
        subax1 = plt.subplot(121)
        nx.draw(self.graph, pos, with_labels=True, font_weight="bold")
        for edge in self.graph.edges(data="weight"):
            nx.draw_networkx_edges(
                self.graph, pos, edgelist=[edge], width=2.2 * edge[2]
            )
        if self.sparsified_graph is None:
            raise ValueError("Sparsify the graph first")
        subax2 = plt.subplot(122)
        nx.draw(self.sparsified_graph, pos, with_labels=True, font_weight="bold")

        for edge in self.sparsified_graph.edges(data="weight"):
            nx.draw_networkx_edges(
                self.sparsified_graph, pos, edgelist=[edge], width=2.2 * edge[2]
            )

        plt.show()

    def verify(self, eps=1e-3, use_accurate_laplacian=True):
        # verify that the sparsified graph indeed approximates the laplacian
        # get the laplacian of self.graph
        L = nx.laplacian_matrix(self.graph).toarray()

        # get the laplacian of self.sparsified_graph
        if use_accurate_laplacian:
            L_sparsified = self.accurate_laplacian
        else:
            L_sparsified = nx.laplacian_matrix(self.sparsified_graph).todense()

        a1 = np.min(np.linalg.eigh(L_sparsified - (1 - eps) * L)[0])
        b1 = np.min(np.linalg.eigh((1 + eps) * L - L_sparsified)[0])
        if self.verbose >= 1:
            print(f"eps = 2sqrt(d)/(d+1) = {eps:.4f}")
            print(
                "LHS (1 - eps) * L_G <= L_H : we check the minimum eigenvalue of the difference:"
            )
            print(f"Min eigenvalue of [L_H - (1 - eps) L_G] = {a1:.2f} >= 0")
            print(
                "RHS L_H <= (1 + eps) * L_G : we check the minimum eigenvalue of the difference:"
            )
            print(f"Min eigenvalue of [(1 + eps) L_G - L_H] = {b1:.2f} >= 0")

        return a1 > -self.eps and b1 > -self.eps

    # def draw_graph(self, L):
    #     A = self.adjac(L)
    #     G = nx.from_numpy_matrix(A)
    #     A1 = self.adjac(self.L)
    #     G2 = nx.from_numpy_matrix(A1)
    #     pos = nx.spring_layout(G)
    #     subax1 = plt.subplot(121)
    #     nx.draw(G2, pos, with_labels=True, font_weight="bold")
    #     for edge in G2.edges(data="weight"):
    #         nx.draw_networkx_edges(G2, pos, edgelist=[edge], width=0.1 * edge[2])
    #     subax2 = plt.subplot(122)
    #     nx.draw(G, pos, with_labels=True, font_weight="bold")
    #     for edge in G.edges(data="weight"):
    #         nx.draw_networkx_edges(G, pos, edgelist=[edge], width=0.1 * edge[2])
    #     plt.show()

    # def ellipse(self):
    # eigenValues,eigenVectors =scipy.linalg.eig(self.L)
    # idx = eigenValues.argsort()
    # eigenValues = eigenValues[idx]
    # eigenVectors = eigenVectors[:,idx]
    # eigen=list(map(lambda x: 1/x,eigenValues[1:]))
    # for i in range(len(eigen)):
    # globals()["x_"+str(i)] = sympy.symbols('x'+str(i))
    # var=np.array([globals()["x_"+str(i)] for i in range(len(eigen))]).reshape(-1,1)
    # expr=0
    # for i in range(len(eigen)):
    # expr+=eigen[i]*(var[i][0]**2)
    # eq1 = sympy.Eq(expr, 1)
    # x=[globals()["x_"+str(i)] for i in range(len(eigen))]
    # sol = sympy.solve([eq1], x,dict=True)
    # print(sol)
    # print(len(eigen))
    # sympy.calculus.util.continuous_domain(sol, x[1:], sympy.S.Reals)
    # p=None
    # for solu in sol:
    # if p is None:
    # p=sympy.plot(solu[x_0](-10,10),show=False)
    # else:
    # t=sympy.plot(solu[x_0],show=False)
    # p.extend(t)
    # p.show()
    # def ellipse_analytic(self):
    # globals()["x_"+str(i)] = sympy.symbols('x'+str(i))
    # var=np.array([[x],[y]])
    # temp
    # m=var.T@self.L@var
    # eq1 = Eq(m[0][0], 1)
    # sol = solve([eq1], [x, y])
    # for solu in sol:
    # plot(solu[0],solu[1])


if "__main__" == __name__:
    # L=Clique(3).laplacian()
    d = 2
    # g = nx.barbell_graph(5, 0)
    g = nx.complete_graph(7)
    TR = TwiceRamanujan(g, d=d, verbose=2)
    L_s = TR.sparsify()
    # TR.ellipse()
    TR.juxtapose(with_verify=True)
    TR.verify(eps=2 * math.sqrt(d) / (d + 1), use_accurate_laplacian=False)
