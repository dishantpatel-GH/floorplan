"""Place every photo in one metric frame from feature matches + per-photo metric depth (decisions P-1, P-5).

Each photo is a levelled metric point cloud (views.py). Two photos that share verified feature matches give 3-D/3-D
correspondences; a robust gravity-constrained similarity (yaw, shift, scale) between them is an EDGE. Edges between
photos of different rooms are the cross-room links (doorway "looking out" shots are designed to create them).

Global solve:
  1. a maximum spanning tree (edges weighted by inlier count) gives initial poses,
  2. robust least squares over ALL edges (soft-L1) refines yaw, position and a per-photo scale correction.
Per-photo scale correction: learned depth has a few-percent scale error that differs between photos. Edges measure
the RATIO of two photos' scales, so the graph pulls every photo to a common scale; the overall scale is then set by
the scale cues (scale.py), not by any single photo.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy.optimize import least_squares


@dataclass
class Edge:
    a: str
    b: str
    yaw: float          # x_a = s R_y(yaw) x_b + t
    t: np.ndarray
    log_s: float
    inliers: int
    matches: int
    rmse_m: float
    cross_room: bool
    violation: float = 0.0     # worst-direction free-space violation fraction (dense check)
    agreement: float = 0.0
    # v2 (D-017): edges that are not feature matches carry explicit 1-sigma uncertainties
    kind: str = "features"     # "features" | "spin_prior" | "doorway_pair" | "layout"
    sig_t: float = 0.0         # metres (priors only)
    sig_yaw_deg: float = 0.0   # degrees (priors only)
    sig_s: float = 0.0         # log-scale (priors only)
    note: str = ""

    @property
    def tree_weight(self) -> float:
        """Spanning-tree preference, in inlier-count units: a doorway pair (same spot, by protocol) beats a weak
        feature link; a spin prior (Manhattan yaw, shared centre) is the weakest."""
        return {"features": self.inliers, "doorway_pair": 30.0, "layout": 18.0}.get(self.kind, 8.0)

    def weights(self) -> tuple[float, float, float]:
        """Residual weights (yaw per rad, translation per m, log-scale) in the solver's units (1 sigma = 0.1)."""
        if self.kind == "features":
            w = float(np.sqrt(self.inliers / 20.0))
            return 2.0 * w, w, 2.0 * w
        return 0.1 / np.radians(self.sig_yaw_deg), 0.1 / self.sig_t, 0.1 / self.sig_s


def _ry(th: float) -> np.ndarray:
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _sim_fit(A: np.ndarray, B: np.ndarray, with_scale: bool = True):
    """Least-squares gravity-constrained similarity B -> A: yaw about +y, scale, shift (closed form)."""
    ca, cb = A.mean(0), B.mean(0)
    a, b = A - ca, B - cb
    # yaw: maximise sum a . R_y b over the x-z components; R_y(th) b = (c bx + s bz, by, -s bx + c bz)
    sc = (a[:, 0] * b[:, 0] + a[:, 2] * b[:, 2]).sum()
    ss = (a[:, 0] * b[:, 2] - a[:, 2] * b[:, 0]).sum()
    th = float(np.arctan2(ss, sc))
    Rb = b @ _ry(th).T
    s = float((a * Rb).sum() / max((Rb ** 2).sum(), 1e-12)) if with_scale else 1.0
    t = ca - s * _ry(th) @ cb
    return th, s, t


def pair_edge(A: np.ndarray, B: np.ndarray, depth: np.ndarray, rel_thr: float, abs_thr: float,
              iters: int = 1000, seed: int = 0):
    """RANSAC over 2-point minimal sets (2 points fix yaw, scale and shift), then refit on inliers."""
    k = len(A)
    if k < 6:
        return None
    rng = np.random.default_rng(seed)
    thr = abs_thr + rel_thr * depth
    best_in = np.zeros(k, bool)
    for _ in range(iters):
        i, j = rng.choice(k, 2, replace=False)
        if np.linalg.norm((B[i] - B[j])[[0, 2]]) < 0.2:
            continue
        th, s, t = _sim_fit(A[[i, j]], B[[i, j]])
        if not 0.6 < s < 1.6:
            continue
        r = np.linalg.norm(s * B @ _ry(th).T + t - A, axis=1)
        inl = r < thr
        if inl.sum() > best_in.sum():
            best_in = inl
    if best_in.sum() < 6:
        return None
    for _ in range(3):
        th, s, t = _sim_fit(A[best_in], B[best_in])
        r = np.linalg.norm(s * B @ _ry(th).T + t - A, axis=1)
        best_in = r < thr
        if best_in.sum() < 6:
            return None
    return th, s, t, best_in, float(np.sqrt((r[best_in] ** 2).mean()))


def build_edges(views: dict, matches: dict, params, log=print) -> list[Edge]:
    edges, rejected = [], dict(scale=0, free_space=0)
    for (na, nb), (xa, xb) in matches.items():
        if na not in views or nb not in views:
            continue
        A, va = views[na].lift(xa)
        B, vb = views[nb].lift(xb)
        ok = va & vb
        if ok.sum() < params.edge_min_inliers:
            continue
        depth = np.maximum(np.linalg.norm(A[ok], axis=1), np.linalg.norm(B[ok], axis=1))
        r = pair_edge(A[ok], B[ok], depth, params.edge_rel_thr, params.edge_abs_thr, seed=params.seed)
        if r is None:
            continue
        th, s, t, inl, rmse = r
        n_in = int(inl.sum())
        if n_in < params.edge_min_inliers or n_in < params.edge_min_ratio * ok.sum():
            continue
        if abs(np.log(s)) > np.log(params.edge_max_scale_ratio):
            rejected["scale"] += 1          # two photos' learned depths disagree too much: likely a false match
            continue
        v_ab = free_space_violation(views[na], views[nb], th, s, t)
        inv_t = -_ry(-th) @ t / s
        v_ba = free_space_violation(views[nb], views[na], -th, 1 / s, inv_t)
        viol = max(v_ab[0], v_ba[0])
        if viol > params.edge_max_violation:
            rejected["free_space"] += 1
            continue
        edges.append(Edge(na, nb, th, t, float(np.log(s)), n_in, int(ok.sum()), rmse,
                          views[na].room != views[nb].room, viol, min(v_ab[1], v_ba[1])))
    log(f"[photo/link] {len(edges)} metric edges from {len(matches)} matched pairs "
        f"({sum(e.cross_room for e in edges)} cross-room); rejected {rejected}")
    return edges


def _compose(pose, e: Edge, forward: bool):
    """pose = (yaw, t, log_s) of the known photo; returns the pose of the other end of the edge."""
    th, t, ls = pose
    if forward:     # known a, unknown b:  x_w = s_a R_a (s_ab R_ab x_b + t_ab) + t_a
        return th + e.yaw, t + np.exp(ls) * _ry(th) @ e.t, ls + e.log_s
    # known b, unknown a: invert the edge, x_b = (1/s_ab) R_ab^T (x_a - t_ab)
    inv_t = -np.exp(-e.log_s) * _ry(-e.yaw) @ e.t
    return th - e.yaw, t + np.exp(ls) * _ry(th) @ inv_t, ls - e.log_s


def solve_component(nodes: list[str], edges: list[Edge], loss_scale_m: float = 0.1) -> dict[str, tuple]:
    """Spanning-tree initialisation, then robust least squares over all edges. Gauge: the best-connected photo
    is fixed at yaw 0, origin, and the mean log-scale correction is pinned to 0 afterwards (scale cues set it)."""
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for k, e in enumerate(edges):
        if not G.has_edge(e.a, e.b) or G[e.a][e.b]["w"] < e.tree_weight:
            G.add_edge(e.a, e.b, w=e.tree_weight, k=k)
    root = max(nodes, key=lambda n: G.degree(n, weight="w"))
    T = nx.maximum_spanning_tree(G, weight="w")
    pose = {root: (0.0, np.zeros(3), 0.0)}
    for u, v in nx.bfs_edges(T, root):
        e = edges[T[u][v]["k"]]
        pose[v] = _compose(pose[u], e, forward=(e.a == u))
    idx = {n: i for i, n in enumerate(nodes)}

    def unpack(x):
        x = x.reshape(-1, 5)
        return x[:, 0], x[:, 1:4], x[:, 4]

    def resid(x):
        th, t, ls = unpack(x)
        out = []
        for e in edges:
            i, j = idx[e.a], idx[e.b]
            pth, pt, pls = _compose((th[i], t[i], ls[i]), e, True)
            wy, wt, ws = e.weights()                                           # features: 1 rad ~ 2 m of residual
            out += [wy * np.angle(np.exp(1j * (th[j] - pth))),
                    *(wt * (t[j] - pt)),
                    ws * (ls[j] - pls)]
        r = idx[root]
        out += [100 * th[r], *(100 * t[r])]                                    # gauge
        return np.array(out)

    x0 = np.concatenate([[pose[n][0], *pose[n][1], pose[n][2]] for n in nodes])
    if len(edges) > len(nodes) - 1:
        x0 = least_squares(resid, x0, loss="soft_l1", f_scale=loss_scale_m).x
    th, t, ls = unpack(x0)
    g = ls.mean()                  # pin the mean scale correction to 1: a global similarity, so positions scale too
    ls, t = ls - g, t * np.exp(-g)
    return {n: (float(th[i]), t[i].copy(), float(ls[i])) for n, i in idx.items()}


def edge_residuals(poses: dict, edges: list[Edge]) -> list[dict]:
    """How well each edge agrees with the solved poses (loop consistency): translation and yaw disagreement."""
    out = []
    for e in edges:
        if e.a not in poses or e.b not in poses:
            continue
        pth, pt, _ = _compose(poses[e.a], e, True)
        th, t, _ = poses[e.b]
        out.append(dict(a=e.a, b=e.b, cross_room=e.cross_room, inliers=e.inliers,
                        dt_m=float(np.linalg.norm(t - pt)), dyaw_deg=float(np.degrees(abs(np.angle(np.exp(1j * (th - pth))))))))
    return out


def prune_inconsistent(names: list[str], edges: list[Edge], max_dt_m: float, max_dt_rel: float,
                       max_dyaw_deg: float, log=print) -> tuple[list[Edge], list[dict]]:
    """Loop-consistency pruning: solve, find the edge that disagrees most with the solution, drop it, repeat.

    A false edge inside a cycle pulls the solution away from the true edges around it, so it ends up with the
    largest residual. (A false edge that is the ONLY link between two groups cannot be detected this way; the
    dense free-space check is the guard against those.)"""
    edges = list(edges)
    removed = []
    while True:
        worst, worst_bad = None, 0.0
        for c in components(names, edges):
            ce = [e for e in edges if e.a in c and e.b in c]
            if len(ce) < len(c):                 # a tree: no cycle, nothing to check
                continue
            poses = solve_component(c, ce)
            for e, r in zip(ce, edge_residuals(poses, ce)):
                if e.kind == "features":
                    bad = max(r["dt_m"] / max(max_dt_m, max_dt_rel * np.linalg.norm(e.t)), r["dyaw_deg"] / max_dyaw_deg)
                else:                            # a prior is inconsistent beyond 3 of its own sigmas
                    bad = max(r["dt_m"] / (3 * e.sig_t), r["dyaw_deg"] / max(3 * e.sig_yaw_deg, max_dyaw_deg))
                if bad > 1 and bad > worst_bad:
                    worst, worst_bad = e, bad
        if worst is None:
            break
        edges.remove(worst)
        removed.append(dict(a=worst.a, b=worst.b, kind=worst.kind, inliers=worst.inliers, badness=round(worst_bad, 2)))
    log(f"[photo/link] loop-consistency pruning removed {len(removed)} edge(s)")
    return edges, removed


def drop_weak_bridges(edges: list[Edge], min_inliers: int, log=print) -> tuple[list[Edge], list[dict]]:
    """Remove weak edges that are the ONLY connection between two groups of photos (graph bridges).

    Loop consistency cannot check a bridge, and the dense check has a blind spot (a wrong edge that slides along a
    wall still agrees). Measured on the sample: every false edge that survived the dense check had <= 19 inliers,
    so a bridge must carry >= min_inliers matches; weaker bridges leave the two groups unlinked (honest) rather
    than linked wrongly (confident garbage)."""
    removed = []
    while True:
        G = nx.Graph()
        for e in edges:
            G.add_edge(e.a, e.b)
        weak = [e for e in edges if e.kind == "features" and e.inliers < min_inliers and (e.a, e.b) in set(nx.bridges(G)) | {
            (b, a) for a, b in nx.bridges(G)}]
        if not weak:
            break
        w = min(weak, key=lambda e: e.inliers)
        edges = [e for e in edges if e is not w]
        removed.append(dict(a=w.a, b=w.b, inliers=w.inliers, reason="weak_bridge"))
    log(f"[photo/link] removed {len(removed)} weak bridge edge(s)")
    return edges, removed


def components(names: list[str], edges: list[Edge]) -> list[list[str]]:
    G = nx.Graph()
    G.add_nodes_from(names)
    G.add_edges_from((e.a, e.b) for e in edges)
    return [sorted(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]


def free_space_violation(va, vb, th: float, s: float, t: np.ndarray, stride: int = 4,
                         agree_rel: float = 0.08, violate_rel: float = 0.15,
                         violate_abs: float = 0.0) -> tuple[float, float, int]:
    """Dense check of an edge: move photo a's whole depth map into photo b and compare with b's depth.

    Feature matches can be geometrically consistent and still wrong (repeated doors, identical cabinets, mirrors):
    the matched points agree, but the REST of the two depth maps does not. If a point of a lands clearly IN FRONT of
    the surface that b observed along the same ray, b would have seen it: a contradiction ("free-space violation").
    Points behind b's surface are merely occluded and are not counted against the edge.
    Returns (violation fraction, agreement fraction, number of comparable points)."""
    Pc, _ = va.points_cam(stride)
    Xa = Pc @ va.R_lev.T                                    # a levelled
    Xb = (Xa - t) @ _ry(th) / s                             # inverse of x_a = s R x_b + t  (R^T x == x @ R)
    Pb = Xb @ vb.R_lev                                      # b camera (R_lev^T x == x @ R_lev)
    z = Pb[:, 2]
    ok = z > 0.1
    u = vb.K[0, 0] * Pb[ok, 0] / z[ok] + vb.K[0, 2]
    v = vb.K[1, 1] * Pb[ok, 1] / z[ok] + vb.K[1, 2]
    h, w = vb.depth.shape
    inside = (u >= 0) & (u < w - 0.5) & (v >= 0) & (v < h - 0.5)
    zb = vb.depth[np.round(v[inside]).astype(int), np.round(u[inside]).astype(int)]
    za = z[ok][inside]
    m = zb > 0
    if m.sum() < 200:
        return 0.0, 0.0, int(m.sum())
    rel = (za[m] - zb[m]) / zb[m]
    viol = (rel < -violate_rel) & (za[m] < zb[m] - violate_abs)     # violate_abs: v2 room-level check tolerance
    return float(viol.mean()), float((np.abs(rel) < agree_rel).mean()), int(m.sum())
