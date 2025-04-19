import numpy as np
from .cython_funcs import dominance_check, concatable, UB_gen

class ESPPRC:
    """
    ESPPRC (Elementary Shortest Path Problem with Resource Constraints)
    solver using a bidirectional label-setting algorithm with Cython-accelerated components.
    """

    def __init__(self,
                 r_max: np.ndarray,
                 r: np.ndarray,
                 max_customer=None) -> None:
        """
        Initialize the ESPPRC instance with resource matrices, duals, and limits.
        """
        self.MAX = 1000000  # Max number of labels
        self.n = r.shape[0]  # Number of nodes
        self.n_res = r.shape[-1] - 1  # Number of resources (excluding cost)
        self.r_max = r_max
        self.r = r
        self.cg_dual = cg_dual
        self.wh_dual = wh_dual
        self.wh_pi = wh_pi
        self.max_customer = max_customer if max_customer is not None else self.n

        # Pre-allocated label containers
        self.Reachable = np.ones((self.MAX, self.n), dtype=np.bool_)
        self.Direction = np.ones(self.MAX, dtype=np.bool_)
        self.Resource = np.zeros((self.MAX, self.n_res + 1), dtype=np.double)
        self.Lb = np.ones(self.MAX, dtype=np.double) * -np.inf
        self.Fertile = np.zeros(self.MAX, dtype=np.bool_)
        self.Empty = np.ones(self.MAX, dtype=np.bool_)
        self.Halved = np.zeros(self.MAX, dtype=np.bool_)
        self.Vertex = -np.ones(self.MAX, dtype=np.int32)
        self.Path = np.zeros((self.MAX, self.n), dtype=np.uint8)
        self.Index = -np.ones(self.MAX, dtype=np.int8)
        self.visit = np.zeros((self.MAX, self.n), dtype=np.uint8)
        self.Parent = -np.ones(self.MAX, dtype=np.int8)
        self.Ever_Cnctd = np.zeros(self.MAX, dtype=np.bool_)
        self.Ever_Domchkd = np.zeros(self.MAX, dtype=np.bool_)

        self.best_path = []
        self.best_cost = []

        self.initlize()
        self.initilize_UB()

    def initlize(self) -> None:
        """
        Set up initial labels for forward and backward search.
        """
        p = [0, 1]
        self.Fertile[p] = True
        self.Vertex[p] = 0
        self.Halved[p] = False
        self.Reachable[0, :] = (self.r[0, :, 1:] <= self.r_max).all(axis=1)
        self.Reachable[1, :] = (self.r_max - self.r[:, 0, 1:] >= 0).all(axis=1)
        self.Reachable[:, 0] = False
        self.Empty[p] = False
        self.Index[p] = 0
        self.Direction[1] = False
        self.visit[p, 0] = 1

        self.Lb[0] = ((self.Reachable[0, :, None] * self.r[:, :, 0][None, :, :]).min(axis=2)).sum(axis=1).sum()
        self.Lb[1] = ((self.Reachable[1, :, None] * (self.r[:, :, 0].T)[None, :, :]).min(axis=2)).sum(axis=1).sum()

    def initilize_UB(self) -> None:
        """
        Initialize upper bound using greedy path heuristic.
        """
        self.UB, p, c = UB_gen(self.r, self.r_max)
        self.best_path += p
        self.best_cost += c

    def generate(self) -> None:
        """
        Generate children labels from current fertile parents.
        """
        p = np.where(self.Fertile)[0]
        if p.size == 0:
            return

        self.Fertile[p] = False
        n_child = self.Reachable[p, :].sum(axis=1).astype(int)
        num_child = int(n_child.sum())
        new_child_index = self.Empty.nonzero()[0][:num_child]
        parent_index = np.repeat(p, n_child).astype(int)
        vertex_child = np.where(self.Reachable[p, :])[1]
        vertex_parent = self.Vertex[parent_index]
        direction = self.Direction[parent_index]

        # Partition based on direction
        p_i_f, p_i_b = parent_index[direction], parent_index[~direction]
        v_p_f, v_p_b = vertex_parent[direction], vertex_parent[~direction]
        v_c_f, v_c_b = vertex_child[direction], vertex_child[~direction]
        c_i_f, c_i_b = new_child_index[direction], new_child_index[~direction]

        R_f = self.Resource[p_i_f, :] + self.r[v_p_f, v_c_f, :]
        R_b = self.Resource[p_i_b, :] + self.r[v_c_b, v_p_b, :]
        Reach_f = self.Reachable[p_i_f, :] & (((R_f[:, None, 1:] + self.r[v_c_f, :, 1:]) <= self.r_max[None, None, :]).all(axis=2))
        Reach_b = self.Reachable[p_i_b, :] & (((R_b[:, None, 1:] + self.r[:, v_c_b, 1:].transpose(1, 0, 2)) <= self.r_max[None, None, :]).all(axis=2))
        Reach_f[np.arange(v_c_f.size), v_c_f] = False
        Reach_b[np.arange(v_c_b.size), v_c_b] = False

        self.Direction[new_child_index] = direction
        self.visit[new_child_index] = self.visit[parent_index]
        self.visit[new_child_index, vertex_child] = 1
        self.Empty[new_child_index] = False

        self.Lb[c_i_f] = ((Reach_f[:, :, None] * self.r[:, :, 0][None, :, :]).min(axis=2)).sum(axis=1) + R_f[:, 0]
        self.Lb[c_i_b] = ((Reach_b[:, :, None] * self.r[:, :, 0].T[None, :, :]).min(axis=2)).sum(axis=1) + R_b[:, 0]

        self.Fertile[c_i_f] = (R_f[:, 1:] <= self.r_max / 2).all(axis=1) & (self.visit[c_i_f, :].sum(axis=1) <= self.max_customer / 2)
        self.Fertile[c_i_b] = (R_b[:, 1:] <= self.r_max / 2).all(axis=1) & (self.visit[c_i_b, :].sum(axis=1) <= self.max_customer / 2)

        self.Halved[new_child_index] = ~self.Fertile[new_child_index]
        self.Parent[new_child_index] = vertex_parent
        self.Vertex[new_child_index] = vertex_child
        self.Resource[c_i_f] = R_f
        self.Resource[c_i_b] = R_b
        self.Reachable[c_i_f] = Reach_f
        self.Reachable[c_i_b] = Reach_b
        self.Index[new_child_index] = self.Index[parent_index] + 1
        self.Path[new_child_index] = self.Path[parent_index]
        self.Path[new_child_index, self.Index[new_child_index]] = vertex_child

        self.drop_labels(p)

    def drop_labels(self, idx):
        """
        Deactivate labels by resetting their properties.
        """
        self.Direction[idx] = True
        self.Empty[idx] = True
        self.Vertex[idx] = -1
        self.Parent[idx] = -1
        self.Resource[idx, :] = 0
        self.Reachable[idx, :] = True
        self.Fertile[idx] = False
        self.Halved[idx] = False
        self.visit[idx, :] = 0
        self.Path[idx, :] = 0
        self.Index[idx] = -1
        self.Lb[idx] = -np.inf
        self.Ever_Cnctd[idx] = False
        self.Ever_Domchkd[idx] = False

    def sort_labels(self) -> None:
        """
        Reorders all active labels lexicographically by cost, vertex, and direction.
        """
        idx = np.where(~self.Empty)[0]
        idx = idx[np.lexsort((self.Resource[idx, 0], self.Vertex[idx], 1 - self.Direction[idx]))]
        i = idx.size

        self.Resource[:i] = self.Resource[idx]
        self.Resource[i:] = 0
        self.Reachable[:i] = self.Reachable[idx]
        self.Reachable[i:] = True
        self.Direction[:i] = self.Direction[idx]
        self.Direction[i:] = True
        self.Empty[:i] = False
        self.Empty[i:] = True
        self.Halved[:i] = self.Halved[idx]
        self.Halved[i:] = False
        self.Vertex[:i] = self.Vertex[idx]
        self.Vertex[i:] = -1
        self.Parent[:i] = self.Parent[idx]
        self.Parent[i:] = -1
        self.Fertile[:i] = self.Fertile[idx]
        self.Fertile[i:] = False
        self.Path[:i] = self.Path[idx]
        self.Path[i:] = 0
        self.Index[:i] = self.Index[idx]
        self.Index[i:] = -1
        self.visit[:i] = self.visit[idx]
        self.visit[i:] = 0
        self.Lb[:i] = self.Lb[idx]
        self.Lb[i:] = -np.inf
        self.Ever_Cnctd[:i] = self.Ever_Cnctd[idx]
        self.Ever_Cnctd[i:] = False
        self.Ever_Domchkd[:i] = self.Ever_Domchkd[idx]
        self.Ever_Domchkd[i:] = False

    def domination(self) -> None:
        """
        Applies dominance rules to prune dominated labels.
        """
        self.sort_labels()
        keep = np.where(~self.Empty)[0]
        keep_ = dominance_check(np.hstack((self.Resource[keep], self.visit[keep])).astype(np.double),
                                self.Vertex[keep],
                                self.Direction[keep], self.Ever_Domchkd[keep])
        self.drop_labels(keep[keep_])

    def solve(self) -> None:
        """
        Main ESPPRC solver loop.
        """
        while np.any(self.Fertile):
            self.generate()
            self.domination()
            self.concat()
        self.best_path = [[int(j) for j in p] for p in self.best_path]
        self.best_cost = [float(c) for c in self.best_cost]

    def concat(self) -> None:
        """
        Joins forward and backward paths and updates the upper bound.
        """
        self.sort_labels()
        keep = np.where(~self.Empty)[0]
        idx = np.where(self.Direction[keep])[0][-1] + 1
        mask, cost, ub = concatable(np.hstack((self.Resource[keep], self.visit[keep])).astype(np.double),
                                    idx, self.Vertex[keep], self.Ever_Cnctd[keep],
                                    self.UB, self.r_max.astype(float), self.cg_dual, self.wh_dual, self.wh_pi)
        self.Ever_Cnctd[keep] = True
        i_, j_ = np.where(mask)
        cost = cost[i_, j_]
        if i_.size == 0:
            return
        i_, j_ = keep[i_], keep[idx + j_]
        i_s = cost.argmin()
        c_s = cost[i_s]
        i_s, j_s = i_[i_s], j_[i_s]
        if c_s < self.UB:
            self.UB = c_s
            path_f = self.Path[i_s][:self.Index[i_s] + 1].tolist()
            path_b = self.Path[j_s][:self.Index[j_s]].tolist()[::-1]
            self.best_path.append(path_f + path_b)
            self.best_cost.append(float(c_s))

