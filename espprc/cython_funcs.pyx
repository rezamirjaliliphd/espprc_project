### cython_funcs.pyx
# Cython-accelerated helper functions for ESPPRC solver
# Provides high-performance routines for dominance checking, label concatenation, and UB initialization

# Cython directives
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False

import numpy as np
cimport numpy as np
from cython.parallel import prange
from libcpp.vector cimport vector
from libc.math cimport floor


cdef bint dominance_check_1d(const double* a, const double* b, int size):
    """Returns True if vector `a` dominates `b` (component-wise <=)."""
    cdef int i
    for i in range(size):
        if a[i] > b[i]:
            return False
    return True

def dominance_check(double[:, :] Res, int[:] Vertex, unsigned char[:] drc, unsigned char[:] Ever_Domchkd):
    """
    Apply pairwise dominance checking for labels with the same vertex and direction.
    Labels that are dominated will be marked and removed.
    """
    cdef int size = Res.shape[0]
    cdef int dim = Res.shape[1]
    cdef int k, j
    cdef unsigned char[:] dominance = np.zeros(size, dtype=np.uint8)
    for k in range(size - 1):
        if not dominance[k]:
            for j in range(k + 1, size):
                if drc[k] != drc[j] or Vertex[k] != Vertex[j]:
                    break
                if Ever_Domchkd[k] and Ever_Domchkd[j]:
                    continue
                if not dominance[j]:
                    if dominance_check_1d(&Res[k, 0], &Res[j, 0], dim):
                        dominance[j] = True
    return np.asarray(dominance, dtype=bool)

def concatable(double[:,:] Res, int idx, int[:] Vertex, unsigned char[:] Ever_cnctd,
               double UB, double[:] r_max, double[:] cg_duals, double[:] wh_duals, double[:] wh_pi):
    """
    Check for forward-backward label pairs that can be concatenated.
    Returns:
      - A mask indicating valid concatenations
      - A cost matrix
      - The updated upper bound (min cost found)
    """
    cdef int num_labels = Res.shape[0]
    cdef int num_res = r_max.shape[0]
    cdef int n = Res.shape[1] - num_res - 1
    cdef int num_wh = wh_duals.shape[0]
    cdef int num_cg = cg_duals.shape[0]
    cdef vector[int] index_f, index_b
    cdef int vertex, j, k, f_idx, b_idx, i
    cdef double ub = UB, cost
    cdef double f_val, b_val

    cdef np.uint8_t[:, :] concat = np.zeros((idx, num_labels - idx), dtype=np.uint8)
    cdef np.double_t[:, :] cost_concat = np.zeros((idx, num_labels - idx), dtype=np.double)

    for vertex in range(1, n + 1):
        index_f = find_index_vertex(&Vertex[0], num_labels, idx, vertex, True)
        index_b = find_index_vertex(&Vertex[0], num_labels, idx, vertex, False)
        if index_f.size() == 0 or index_b.size() == 0:
            continue

        for k in range(index_f.size()):
            f_idx = index_f[k]
            f_val = Res[f_idx, 0]
            if f_val + Res[index_b[0], 0] >= ub:
                break
            for j in range(index_b.size()):
                b_idx = index_b[j]
                b_val = Res[b_idx, 0]
                if Ever_cnctd[f_idx] and Ever_cnctd[b_idx]:
                    continue
                if f_val + b_val >= ub:
                    break
                if concatable_1d(&Res[f_idx, 0], &Res[b_idx, 0], Res.shape[1], vertex, UB, &r_max[0], num_res):
                    concat[f_idx, b_idx - idx] = 1
                    cost = f_val + b_val
                    for i in range(3, 3 + num_cg):
                        cost += cg_duals[i - 3] * floor(Res[f_idx, i] + Res[b_idx, i])
                    for i in range(num_wh):
                        cost += wh_duals[i] * floor((Res[f_idx, 1] + Res[b_idx, 1]) * wh_pi[i])
                    cost_concat[f_idx, b_idx - idx] = cost
                    if ub > cost:
                        ub = cost
    return np.asarray(concat, dtype=bool), np.asarray(cost_concat, dtype=np.double), ub

cdef vector[int] find_index_vertex(const int* Vertex, int num_labels, int idx, int vertex, bint direction):
    """
    Returns a list of label indices matching a given vertex.
    Direction = True: forward labels [0:idx]
    Direction = False: backward labels [idx:num_labels]
    """
    cdef vector[int] index = vector[int]()
    cdef int i
    if direction:
        for i in range(idx):
            if Vertex[i] > vertex:
                break
            elif Vertex[i] == vertex:
                index.push_back(i)
    else:
        for i in range(idx, num_labels):
            if Vertex[i] > vertex:
                break
            elif Vertex[i] == vertex:
                index.push_back(i)
    return index

cdef int concatable_1d(const double* f_res, const double* b_res, int res_length, int vertex,
                       double UB, const double* r_max, int num_res):
    """
    Check if two labels are compatible in terms of:
    - Resource feasibility
    - Visit feasibility (binary indicators)
    """
    cdef int i
    for i in range(1, num_res + 1):
        if f_res[i] + b_res[i] > r_max[i - 1]:
            return False
    for i in range(num_res + 2, res_length):
        if i - num_res - 1 == vertex:
            continue
        if f_res[i] + b_res[i] > 1:
            return False
    return True

def UB_gen(double[:, :, :] r, double[:] r_max):
    """
    Heuristic upper bound generator by enumerating short paths (3-customer tours).
    Returns:
      - The upper bound
      - The list of feasible paths
      - Their costs
    """
    cdef int n = r.shape[0]
    cdef int num_res = r_max.shape[0]
    cdef int i, j, k, l
    cdef bint ind
    cdef double ub = 1000, cost, res
    cdef vector[vector[int]] Path = vector[vector[int]]()
    cdef vector[double] Cost = vector[double]()
    cdef vector[int] p = vector[int]()

    for i in range(1, n):
        for j in range(1, n):
            for k in range(1, n):
                if i < j and j < k:
                    ind = True
                    cost = r[0, i, 0] + r[i, j, 0] + r[j, k, 0] + r[k, 0, 0]
                    for l in range(1, num_res + 1):
                        res = r[0, i, l] + r[i, j, l] + r[j, k, l] + r[k, 0, l]
                        if res > r_max[l - 1]:
                            ind = False
                            break
                    if ind and cost < 0 and cost < ub:
                        p.clear()
                        p.push_back(0)
                        p.push_back(i)
                        p.push_back(j)
                        p.push_back(k)
                        p.push_back(0)
                        Path.push_back(p)
                        Cost.push_back(cost)
                        ub = cost
    return ub, [list(Path[i]) for i in range(Path.size())], list(Cost)

