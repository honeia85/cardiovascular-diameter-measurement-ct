import heapq
import json
import math
import os
import sys

if sys.version_info[:2] != (3, 12):
    raise RuntimeError(
        "This release requires Python 3.12.x; create a Python 3.12 environment "
        "before importing the measurement modules."
    )

import cv2
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema, find_peaks
from skimage.measure import find_contours
from skimage.morphology import skeletonize
from typing import Tuple, Optional

# math / os / skeletonize are not used in this file directly, but measure_vessel
# picks them up via `from utils import *`, so keep them.


def rc_to_xy(p):
    return (int(p[1]), int(p[0]))

def load_nii(path: str) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    img = nib.load(path)
    if len(img.shape) != 3:
        raise ValueError(f"NIfTI image must be a 3D volume; got {img.shape}: {path}")
    if not np.all(np.isfinite(img.affine)):
        raise ValueError(f"NIfTI affine contains non-finite values: {path}")
    axis_codes = tuple(nib.aff2axcodes(img.affine))
    if axis_codes != ("L", "P", "S"):
        raise ValueError(
            f"unsupported NIfTI orientation {axis_codes} for {path}; "
            "this implementation requires LPS+ voxel orientation"
        )
    linear = np.asarray(img.affine[:3, :3], dtype=float)
    column_norms = np.linalg.norm(linear, axis=0)
    if not np.all(np.isfinite(column_norms)) or np.any(column_norms <= 0):
        raise ValueError(f"NIfTI affine has invalid spatial axes: {path}")
    directions = linear / column_norms
    if not np.allclose(
        directions, np.diag([-1.0, -1.0, 1.0]), rtol=0.0, atol=1e-5
    ):
        raise ValueError(
            f"NIfTI affine must be axis-aligned LPS+ for {path}; "
            "oblique/sheared input is unsupported"
        )
    spacing = np.asarray(img.header.get_zooms()[:3], dtype=float)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError(f"invalid voxel spacing {spacing.tolist()} for {path}")
    if not np.isclose(spacing[0], spacing[1], rtol=1e-5, atol=1e-6):
        raise ValueError(
            f"in-plane spacing must be isotropic for {path}; got "
            f"{spacing[0]:.9g} x {spacing[1]:.9g} mm"
        )
    data = img.get_fdata(dtype=np.float32)
    data = np.rint(data).astype(np.int32)
    return data, img.affine, img.header

def pick_volume3d(data: np.ndarray) -> np.ndarray:
    if data.ndim == 4:
        return data[..., 0]
    if data.ndim == 3:
        return data
    raise ValueError(f"unsupported shape: {data.shape}")


def keep_only_component_with_point(mask, p_rc, connectivity=4):
    h, w = mask.shape
    pr, pc = p_rc
    if not (0 <= pr < h and 0 <= pc < w):
        raise ValueError("p_rc is outside the mask bounds.")
    if mask[pr, pc] == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    if mask.dtype != bool and mask.max() > 1:
        return (mask == mask[pr, pc]).astype(np.uint8)
    struct = ndi.generate_binary_structure(2, 1 if connectivity == 4 else 2)
    labeled, _ = ndi.label(mask, structure=struct)
    label_at_p = labeled[pr, pc]
    if label_at_p == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    return (labeled == label_at_p).astype(np.uint8)


def slice_by_axis(vol: np.ndarray, axis: str, index: Optional[int]):
    axis = axis.lower()
    X, Y, Z = vol.shape
    if axis == 'axial':
        if index is None: index = Z // 2
        if not (0 <= index < Z): raise IndexError(f"axial index {index} / Z={Z}")
        return vol[:, :, index]
    elif axis == 'coronal':
        if index is None: index = Y // 2
        if not (0 <= index < Y): raise IndexError(f"coronal index {index} / Y={Y}")
        return vol[:, index, :]
    elif axis == 'sagittal':
        if index is None: index = X // 2
        if not (0 <= index < X): raise IndexError(f"sagittal index {index} / X={X}")
        return vol[index, :, :]
    else:
        raise ValueError("axis must be one of 'axial' | 'coronal' | 'sagittal'.")

def parabolic_refine(y_minus, y0, y_plus):
    denom = (y_minus - 2 * y0 + y_plus)
    if denom == 0:
        return 0.0
    return 0.5 * (y_minus - y_plus) / denom

def wrap_indices(n, i0, i1, include_end=False):
    if include_end:
        if i1 >= i0:
            return np.arange(i0, i1 + 1, dtype=int)
        else:
            return np.concatenate([np.arange(i0, n, dtype=int), np.arange(0, i1 + 1, dtype=int)])
    else:
        if i1 >= i0:
            return np.arange(i0, i1, dtype=int)
        else:
            return np.concatenate([np.arange(i0, n, dtype=int), np.arange(0, i1, dtype=int)])

def extract_main_contour(mask):
    m = (mask > 0).astype(float)
    curves = find_contours(m, level=0.5)
    if len(curves) == 0:
        raise ValueError("no contour found.")
    curves.sort(key=lambda c: len(c), reverse=True)
    return curves[0]

def split_by_ridge_maxima(contour_rc, center_rc, smooth_sigma=5, min_sep_ratio=0.18):
    N = len(contour_rc)
    r, c = contour_rc[:, 0], contour_rc[:, 1]
    d = np.hypot(r - center_rc[0], c - center_rc[1])
    d_s = gaussian_filter1d(d, sigma=smooth_sigma, mode='wrap')
    mins = find_mins(d_s)
    if len(mins) <= 2:
        min_sep = max(2, int(N * min_sep_ratio))
        min_peaks, min_props = find_peaks(-d_s, distance=min_sep,
                                          prominence=np.ptp(d_s) * 0.02)
        if len(min_peaks) < 3:
            mins = np.argsort(d_s)[:3]
        else:
            order = np.argsort(min_props["prominences"])[::-1]
            mins = min_peaks[order[:3]]
    mins = np.sort(mins)
    cuts = []
    for a, b in zip(mins, np.roll(mins, -1)):
        seg = wrap_indices(N, a, b, include_end=True)
        if len(seg) > 6:
            seg = seg[3:-3]
        k = seg[np.argmax(d_s[seg])]
        cuts.append(int(k))
    cuts = np.array(sorted(cuts))
    i0, i1, i2 = cuts
    arcs = [
        wrap_indices(N, i0, i1, include_end=False),
        wrap_indices(N, i1, i2, include_end=False),
        wrap_indices(N, i2, i0, include_end=False),
    ]
    return cuts, mins, d_s, arcs

def find_mins(d_s, k=3, order=1):
    y = np.asarray(d_s)
    x = np.arange(len(y))
    min_idx = argrelextrema(y, np.less, order=order)[0]
    if min_idx.size == 0:
        return np.array([], dtype=int)
    top = min_idx[np.argsort(y[min_idx])[:k]]
    return x[top[np.argsort(top)]]

def gradient_descent_on_arc(d_s, arc_indices, max_iter=200):
    if not isinstance(arc_indices, (list, tuple, np.ndarray)):
        arc_indices = list(arc_indices)
    arc_set = set(arc_indices)
    if len(arc_indices) == 0:
        raise ValueError("arc_indices is empty.")
    N = len(d_s)
    i = arc_indices[len(arc_indices) // 2]
    for _ in range(max_iter):
        im1 = (i - 1) % N
        ip1 = (i + 1) % N
        g = 0.5 * (d_s[ip1] - d_s[im1])
        if abs(g) < 1e-8:
            break
        step = -1 if g > 0 else 1
        cand = (i + step) % N
        if cand in arc_set:
            if d_s[cand] > d_s[i]:
                break
            i = cand
        else:
            break
    im1 = (i - 1) % N
    ip1 = (i + 1) % N
    delta = parabolic_refine(d_s[im1], d_s[i], d_s[ip1])
    return i, float(delta)


def mopology_noise_remover(m, k=5, iter=3):
    m = m * 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dst = cv2.erode(m, k, iterations=iter)
    blur = cv2.GaussianBlur(dst, ksize=(5, 5), sigmaX=0)
    dst = cv2.dilate(blur, k, iterations=iter)
    return (dst > 0).astype(np.uint8)

def pick_indices(df: pd.DataFrame):
    head_index = df['y'].idxmin()
    rem = df.drop(index=head_index)
    left_index = rem['x'].idxmin()
    right_index = rem.index.difference([left_index])[0]
    return head_index, left_index, right_index

def two_nearest_indices(df: pd.DataFrame, p) -> tuple:
    r0, c0 = float(p[0]), float(p[1])
    d2 = (df['y'] - r0)**2 + (df['x'] - c0)**2
    return d2.nsmallest(2).index.tolist()

def keep_only_inside_contour(label_mask: np.ndarray, contour_rc: np.ndarray) -> np.ndarray:
    h, w = label_mask.shape[:2]
    poly = np.round(contour_rc[:, [1, 0]]).astype(np.int32).reshape(-1, 1, 2)
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(poly_mask, [poly], 1)
    if label_mask.dtype == bool:
        return label_mask & (poly_mask.astype(bool))
    else:
        return label_mask * poly_mask.astype(label_mask.dtype)

def _foreground_rc(mask: np.ndarray) -> np.ndarray:
    fg = mask > 0 if mask.dtype != np.bool_ else mask
    rs, cs = np.nonzero(fg)
    if len(cs) < 6:
        raise ValueError("too few foreground pixels (>=6 required).")
    return np.column_stack([rs.astype(float), cs.astype(float)])


def partial_mask(mask, result, nearest_neck_index, neck, center, i=0, ridge=None):
    m = (mask > 0).astype(np.uint8)
    m = mopology_noise_remover(m)
    contour_rc = extract_main_contour(m)
    N = len(contour_rc)
    min_indices = np.array(result['min_indices'])
    i0 = min_indices[nearest_neck_index[0]]
    i1 = min_indices[nearest_neck_index[1]]
    p_mask = np.ones(len(min_indices), dtype=bool)
    p_mask[nearest_neck_index] = False
    i2 = min_indices[p_mask][0]

    def find_branch(m, i1, i2):
        i1, i2 = sorted([i1, i2])
        arc = wrap_indices(N, i1, i2, include_end=False)
        partial_contour = contour_rc[arc]
        m1 = keep_only_inside_contour(m, partial_contour)
        rr, rc = np.int32(center)
        if m1[rr, rc] > 0:
            i1, i2 = (i2, i1)
            arc = wrap_indices(N, i1, i2, include_end=False)
            partial_contour = contour_rc[arc]
            m1 = keep_only_inside_contour(m, partial_contour)
        return m1

    m1 = find_branch(m, i0, i2)
    m2 = find_branch(m, i1, i2)
    m3 = cv2.bitwise_or(m1, m2)
    m4 = cv2.bitwise_xor(m, m3)
    r0, c0 = np.mean(neck, axis=0)
    return keep_only_component_with_point(m4, (int(r0), int(c0)))


def skel_neighbors(x, y, skel, allow_diag=True):
    H, W = skel.shape
    nbrs = ([(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]
            if allow_diag else [(0,-1),(-1,0),(1,0),(0,1)])
    return [(x+dx, y+dy) for dx, dy in nbrs
            if 0 <= x+dx < W and 0 <= y+dy < H and skel[y+dy, x+dx]]

def astar_on_skel(skel, start, goal, allow_diag=True):
    skel = (skel > 0)
    H, W = skel.shape

    def h(p):
        return np.hypot(p[0] - goal[0], p[1] - goal[1])

    openpq = []
    heapq.heappush(openpq, (h(start), 0.0, start))
    came = {start: None}
    gscore = {start: 0.0}
    closed = set()

    while openpq:
        _, g, curr = heapq.heappop(openpq)
        if curr in closed:
            continue
        closed.add(curr)
        if curr == goal:
            path = []
            p = curr
            while p is not None:
                path.append(p); p = came[p]
            return np.asarray(path[::-1], dtype=int)
        for nb in skel_neighbors(curr[0], curr[1], skel, allow_diag):
            if nb in closed:
                continue
            ng = g + np.hypot(nb[0] - curr[0], nb[1] - curr[1])
            if nb not in gscore or ng < gscore[nb]:
                gscore[nb] = ng
                came[nb] = curr
                heapq.heappush(openpq, (ng + h(nb), ng, nb))
    raise RuntimeError("A* failed: no path between projected endpoints")


def estimate_direction(path, at_head=True, k=5):
    if len(path) < k + 1:
        k = len(path) - 1
    if k <= 0:
        return None
    v = path[0] - path[k] if at_head else path[-1] - path[-k-1]
    n = np.linalg.norm(v)
    return None if n < 1e-6 else v / n

def _u_samples_away_from_center(u_start, parab, center_rc, du, count,
                                 u_bounds=None, ridge=(0, 0)):
    coeffs = np.asarray(parab["coeffs"], float)
    R, mu = parab["R"], parab["mu"]
    CRc = np.array([float(center_rc[1]), float(center_rc[0])], dtype=float)
    u_plus, u_minus = u_start + du, u_start - du
    cr_plus  = R @ np.array([u_plus,  np.polyval(coeffs, u_plus)])  + mu
    cr_minus = R @ np.array([u_minus, np.polyval(coeffs, u_minus)]) + mu
    r_ref, c_ref = float(ridge[0]), float(ridge[1])
    dist_plus  = np.hypot(float(cr_plus[1])  - r_ref, float(cr_plus[0])  - c_ref)
    dist_minus = np.hypot(float(cr_minus[1]) - r_ref, float(cr_minus[0]) - c_ref)
    sgn = 1.0 if dist_plus <= dist_minus else -1.0
    us = [u_start + sgn * k * du for k in range(0, count + 1)]
    if u_bounds is not None:
        lo, hi = u_bounds
        us = [u for u in us if lo <= u <= hi]
    return np.array(us, dtype=float)

def _normal_at_u(parab, u):
    coeffs = np.asarray(parab["coeffs"], float)
    R, mu = parab["R"], parab["mu"]
    v = float(np.polyval(coeffs, u))
    CR = (R @ np.array([u, v])) + mu
    c0, r0 = float(CR[0]), float(CR[1])
    dv_du = float(np.polyval(np.polyder(coeffs), u))
    t_CR = R @ np.array([1.0, dv_du])
    t_rc = np.array([float(t_CR[1]), float(t_CR[0])], dtype=float)
    t_norm = np.linalg.norm(t_rc)
    if t_norm == 0:
        raise RuntimeError("Zero tangent norm at u.")
    t_rc /= t_norm
    n_rc = np.array([-t_rc[1], t_rc[0]], dtype=float)
    n_rc /= np.linalg.norm(n_rc)
    return (r0, c0), t_rc, n_rc

from scipy.ndimage import map_coordinates

def _dist_point_to_segment(px, py, x1, y1, x2, y2):
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    seg_len2 = vx*vx + vy*vy
    if seg_len2 == 0:
        return np.hypot(px - x1, py - y1)
    t = np.clip((wx*vx + wy*vy) / seg_len2, 0.0, 1.0)
    return np.hypot(px - (x1 + t*vx), py - (y1 + t*vy))

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) - (B[1] - A[1]) * (C[0] - A[0])

def segments_intersect(p1, p2, p3, p4):
    A, B, C, D = p1, p2, p3, p4
    ccw1, ccw2 = ccw(A, B, C), ccw(A, B, D)
    ccw3, ccw4 = ccw(C, D, A), ccw(C, D, B)
    if ccw1 * ccw2 < 0 and ccw3 * ccw4 < 0:
        return True
    def on_seg(P, Q, R):
        return (min(P[0], R[0]) <= Q[0] <= max(P[0], R[0]) and
                min(P[1], R[1]) <= Q[1] <= max(P[1], R[1]))
    if ccw1 == 0 and on_seg(A, C, B): return True
    if ccw2 == 0 and on_seg(A, D, B): return True
    if ccw3 == 0 and on_seg(C, A, D): return True
    if ccw4 == 0 and on_seg(C, B, D): return True
    return False

def intersects_with_neck(r1, c1, r2, c2, neck):
    return segments_intersect((r1, c1), (r2, c2), neck[0], neck[1])

def _clip_normal_on_mask(r0, c0, neck, n_rc, length, mask, step=0.25, thresh=None):
    H, W = mask.shape
    if thresh is None:
        thresh = 0.5 if mask.max() <= 1 else 127.5
    half = length * 0.5
    N = max(2, int(np.ceil(length/step)) + 1)
    ts = np.linspace(-half, half, N)
    rs = r0 + n_rc[0]*ts
    cs = c0 + n_rc[1]*ts
    vals = map_coordinates(mask.astype(float), [rs, cs], order=1,
                           mode='constant', cval=0.0)
    inside = vals >= thresh
    if not np.any(inside):
        return [], 0.0

    def interp_edge(i0, i1):
        v0, v1 = vals[i0], vals[i1]
        t0, t1 = ts[i0], ts[i1]
        if v1 == v0:
            return 0.5 * (t0 + t1)
        alpha = np.clip((thresh - v0) / (v1 - v0), 0.0, 1.0)
        return t0 + alpha*(t1 - t0)

    jump = np.flatnonzero(inside[1:] != inside[:-1])
    starts, ends = [], []
    if inside[0]:
        starts.append(0)
    for j in jump:
        if not inside[j] and inside[j+1]:
            starts.append(j)
        elif inside[j] and not inside[j+1]:
            ends.append(j)
    if inside[-1]:
        ends.append(N-2)

    segments = []
    for s, e in zip(starts, ends):
        t1 = ts[0] if (s == 0 and inside[0]) else interp_edge(s, s+1)
        t2 = ts[-1] if (e == N-2 and inside[-1]) else interp_edge(e+1, e)
        t_min, t_max = (t1, t2) if t1 <= t2 else (t2, t1)
        r1 = r0 + n_rc[0]*t_min; c1 = c0 + n_rc[1]*t_min
        r2 = r0 + n_rc[0]*t_max; c2 = c0 + n_rc[1]*t_max
        if intersects_with_neck(r1, c1, r2, c2, neck):
            continue
        segments.append((r1, c1, r2, c2, t_max - t_min))

    if len(segments) > 1:
        dists = [_dist_point_to_segment(r0, c0, seg[0], seg[1], seg[2], seg[3])
                 for seg in segments]
        best_seg = segments[int(np.argmin(dists))]
        return [best_seg], best_seg[4]
    return (segments, segments[0][4]) if segments else ([], 0.0)

def u_range_from_mask(mask, R, mu, margin_ratio=0.10):
    rc = _foreground_rc(mask)
    CR = np.column_stack([rc[:, 1], rc[:, 0]])
    UV = uv_from_cr(CR, R, mu)
    u = UV[:, 0]
    umin, umax = float(u.min()), float(u.max())
    pad = (umax - umin) * margin_ratio if umax > umin else 1.0
    return umin - pad, umax + pad

def uv_from_cr(CR, R, mu):
    return (R.T @ (CR - mu).T).T


def dist_yxz(p, q, spacing):
    sx, sy, sz = spacing
    dy = (p[0] - q[0]) * sy
    dx = (p[1] - q[1]) * sx
    return np.sqrt(dy*dy + dx*dx)

def draw_clipped_normals_and_measure(ax, parab, start_r, start_c, mask,
                                     du=2.0, count_each_side=20, length=28.0,
                                     color="tab:blue", linewidth=2.0,
                                     center_rc=None, only_inside=True,
                                     step_along_normal=0.25, return_details=True,
                                     skel_label=[], ratio_dict={}, slice_idx=0,
                                     spacing=(1.0, 1.0, 1.0), global_center=(0, 0),
                                     dist=(0, 0), ridge=(0, 0), neck=[]):
    u_star, (r0, c0), _, _ = normal_at_rc(parab, r=start_r, c=start_c, mask=mask)
    u_bounds = u_range_from_mask(mask, parab["R"], parab["mu"], margin_ratio=0.10)

    if center_rc is None:
        us = [u_star]
        for k in range(1, count_each_side + 1):
            us.extend([u_star + k*du, u_star - k*du])
        us = np.array(sorted(us))
        lo, hi = u_bounds
        us = us[(us >= lo) & (us <= hi)]
    else:
        us = _u_samples_away_from_center(
            u_start=u_star, parab=parab, center_rc=center_rc,
            du=du, count=count_each_side, u_bounds=u_bounds, ridge=ridge)

    lengths, details, ratios = [], [], []
    for u in us:
        try:
            (r_c, c_c), _, n_rc = _normal_at_u(parab, u)
            if mask[int(r_c), int(c_c)] == 0:
                continue
            d = dist_yxz(global_center, (r_c, c_c), spacing)
            if d <= dist[0] or d >= dist[1]:
                continue
            segs, total_len = _clip_normal_on_mask(
                r_c, c_c, neck, n_rc, length=length, mask=mask,
                step=step_along_normal, thresh=None)
            lengths.append(total_len)
            if return_details:
                details.append((u, (r_c, c_c), segs, total_len, 0))
        except Exception:
            continue

    return ((np.array(us), np.array(lengths), details, ratios)
            if return_details else (np.array(us), np.array(lengths)))

def _u_bounds_from_mask(mask, R, mu, margin_ratio=0.10):
    rs, cs = np.nonzero(mask > 0)
    if rs.size == 0:
        raise ValueError("Foreground not found.")
    CR = np.column_stack([cs, rs]).astype(float)
    UV = (R.T @ (CR - mu).T).T
    u = UV[:, 0]
    umin, umax = float(u.min()), float(u.max())
    pad = (umax - umin) * margin_ratio
    return umin - pad, umax + pad

def _closest_u_to_point_poly(u0, v0, coeffs, u_bounds=None, n_samples=200):
    if u_bounds is None:
        umin, umax = u0 - 1.0, u0 + 1.0
    else:
        umin, umax = u_bounds
    us = np.linspace(umin, umax, n_samples)
    vs = np.polyval(coeffs, us)
    d2 = (us - u0)**2 + (vs - v0)**2
    return float(us[np.argmin(d2)])

def normal_at_rc(parab, r, c, mask=None):
    coeffs = np.asarray(parab["coeffs"], float)
    R, mu = parab["R"], parab["mu"]
    CRp = np.array([c, r], dtype=float)
    UVp = R.T @ (CRp - mu)
    u0, v0 = float(UVp[0]), float(UVp[1])
    u_bounds = _u_bounds_from_mask(mask, R, mu) if mask is not None else None
    u_star = _closest_u_to_point_poly(u0, v0, coeffs, u_bounds=u_bounds)
    v_star = np.polyval(coeffs, u_star)
    CRs = R @ np.array([u_star, v_star]) + mu
    c0_star, r0_star = float(CRs[0]), float(CRs[1])
    dv_du = np.polyval(np.polyder(coeffs), u_star)
    t_CR = R @ np.array([1.0, dv_du])
    t_rc = np.array([float(t_CR[1]), float(t_CR[0])], dtype=float)
    t_norm = np.linalg.norm(t_rc)
    if t_norm == 0:
        raise RuntimeError("Zero tangent norm at u*.")
    t_rc /= t_norm
    n_rc = np.array([-t_rc[1], t_rc[0]], dtype=float)
    n_rc /= np.linalg.norm(n_rc)
    return u_star, (r0_star, c0_star), t_rc, n_rc


def get_two_largest_components(mask, connectivity=1):
    mask = mask.astype(bool)
    structure = ndi.generate_binary_structure(2, connectivity)
    labeled, num = ndi.label(mask, structure=structure)
    if num == 0:
        empty = np.zeros_like(mask, dtype=bool)
        return empty, empty
    sizes = np.array(ndi.sum(mask, labeled, index=range(1, num + 1)))
    top_k = min(2, num)
    top_labels = np.argsort(sizes)[-top_k:] + 1
    mask1 = (labeled == top_labels[-1])
    mask2 = (labeled == top_labels[-2]) if top_k == 2 else np.zeros_like(mask, dtype=bool)
    return mask1, mask2

def measure_width_along_perp(mask, centroid, perp_unit):
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return 0.0, (np.nan, np.nan), (np.nan, np.nan)
    pts = np.column_stack([ys, xs]).astype(float)
    c = np.array(centroid, dtype=float)
    t = (pts - c) @ perp_unit
    t_min, t_max = t.min(), t.max()
    return float(t_max - t_min), tuple(c + perp_unit*t_min), tuple(c + perp_unit*t_max)

def analyze_two_masks_from_single_mask(mask, spacing=1.0, connectivity=1,
                                       do_plot=False):
    mask1, mask2 = get_two_largest_components(mask, connectivity=connectivity)
    cy1, cx1 = ndi.center_of_mass(mask1)
    cy2, cx2 = ndi.center_of_mass(mask2)
    c1, c2 = (cy1, cx1), (cy2, cx2)
    dy, dx = cy2 - cy1, cx2 - cx1
    v_norm = np.hypot(dy, dx)
    if v_norm == 0:
        perp_unit = np.array([0.0, 1.0])
    else:
        perp = np.array([-dx, dy], dtype=float)
        perp_unit = perp / np.hypot(perp[0], perp[1])
    w1, p1_min, p1_max = measure_width_along_perp(mask1, c1, perp_unit)
    w2, p2_min, p2_max = measure_width_along_perp(mask2, c2, perp_unit)
    return {
        "centroid1": c1, "centroid2": c2,
        "width1_px": w1, "width2_px": w2,
        "width1_phys": w1 * spacing, "width2_phys": w2 * spacing,
        "perp_unit": tuple(perp_unit),
        "segment1_endpoints": (p1_min, p1_max),
        "segment2_endpoints": (p2_min, p2_max),
    }

def save_line_markup_json(filename, pos1, pos2, length_value):
    pos1 = np.asarray(pos1, dtype=float)
    pos2 = np.asarray(pos2, dtype=float)
    length_value = float(length_value)
    if pos1.shape != (3,) or pos2.shape != (3,):
        raise ValueError("markup endpoints must each contain three coordinates")
    if not np.all(np.isfinite(pos1)) or not np.all(np.isfinite(pos2)):
        raise ValueError("markup endpoints must be finite")
    endpoint_distance = float(np.linalg.norm(pos2 - pos1))
    if not np.isfinite(endpoint_distance) or endpoint_distance <= 0:
        raise ValueError("markup endpoints must define a positive finite line")
    if not np.isfinite(length_value) or length_value <= 0:
        raise ValueError(f"measurement length must be finite and > 0; got {length_value}")

    data = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": [{
            "type": "Line",
            "coordinateSystem": "LPS",
            "coordinateUnits": "mm",
            "locked": False,
            "fixedNumberOfControlPoints": False,
            "labelFormat": "%N-%d",
            "lastUsedControlPointNumber": 2,
            "controlPoints": [
                {"id": "1", "label": "L-1", "description": "",
                 "associatedNodeID": "vtkMRMLScalarVolumeNode1",
                 "position": pos1.tolist(),
                 "orientation": [-1.0,-0.0,-0.0,-0.0,-1.0,-0.0,0.0,0.0,1.0],
                 "selected": True, "locked": False, "visibility": True,
                 "positionStatus": "defined"},
                {"id": "2", "label": "L-2", "description": "",
                 "associatedNodeID": "vtkMRMLScalarVolumeNode1",
                 "position": pos2.tolist(),
                 "orientation": [-1.0,-0.0,-0.0,-0.0,-1.0,-0.0,0.0,0.0,1.0],
                 "selected": True, "locked": False, "visibility": True,
                 "positionStatus": "defined"}
            ],
            "measurements": [{"name": "length", "enabled": True,
                               "value": length_value, "units": "mm",
                               "printFormat": "%-#4.4gmm"}],
            "display": {
                "visibility": True, "opacity": 1.0,
                "color": [0.4, 1.0, 1.0],
                "selectedColor": [1.0, 0.5000076295109483, 0.5000076295109483],
                "activeColor": [0.4, 1.0, 0.0],
                "propertiesLabelVisibility": True, "pointLabelsVisibility": False,
                "textScale": 3.0, "glyphType": "Sphere3D", "glyphScale": 1.0,
                "glyphSize": 5.0, "useGlyphScale": True, "sliceProjection": False,
                "sliceProjectionUseFiducialColor": True,
                "sliceProjectionOutlinedBehindSlicePlane": False,
                "sliceProjectionColor": [1.0, 1.0, 1.0], "sliceProjectionOpacity": 0.6,
                "lineThickness": 0.2, "lineColorFadingStart": 1.0,
                "lineColorFadingEnd": 10.0, "lineColorFadingSaturation": 1.0,
                "lineColorFadingHueOffset": 0.0, "handlesInteractive": False,
                "translationHandleVisibility": True, "rotationHandleVisibility": True,
                "scaleHandleVisibility": False, "interactionHandleScale": 3.0,
                "snapMode": "toVisibleSurface"
            }
        }]
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, allow_nan=False)

def voxel_yxz_to_LPS(p_yxz, affine):
    y, x, z = map(float, p_yxz)
    ijk_h = np.array([x, y, z, 1.0], dtype=float)
    ras = affine @ ijk_h
    R, A, S = ras[:3]
    return np.array([-R, -A, S], dtype=float)


def fit_spline_from_skeleton(pts_xy, smooth=2.0):
    k = min(3, len(pts_xy) - 1)
    x, y = pts_xy[:, 0], pts_xy[:, 1]
    tck, _ = splprep([x, y], s=smooth, k=k)
    return tck

def spline_to_uv(tck, n=800):
    t = np.linspace(0, 1, n)
    from scipy.interpolate import splev
    x, y = splev(t, tck)
    pts = np.column_stack([x, y])
    d = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
    s = np.concatenate([[0], np.cumsum(d)])
    s /= s[-1]
    mu = pts.mean(axis=0)
    X = pts - mu
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    R = Vt[:2].T
    uv = (R.T @ X.T).T
    return uv[:, 0], uv[:, 1], R, mu

def fit_poly_aic(u, v, max_degree=3, criterion="AIC"):
    n = len(u)
    best = None
    for d in range(1, max_degree + 1):
        coeffs = np.polyfit(u, v, d)
        rss = np.sum((v - np.polyval(coeffs, u))**2)
        k = d + 1
        score = (n * np.log(rss / n) + 2 * k if criterion == "AIC"
                 else n * np.log(rss / n) + k * np.log(n))
        if best is None or score < best["score"]:
            best = {"degree": d, "coeffs": coeffs, "score": score}
    return best


def extract_centerline_from_mask(mask, center, num_samples=200,
                                  slice_half_width=20, drop_tail_ratio=0.2):
    mask = mask.astype(bool)
    center = np.asarray(center, dtype=float)
    ys, xs = np.where(mask)
    if len(xs) < 20:
        raise RuntimeError("Mask too small")
    coords = np.stack([xs, ys], axis=1)
    mu = coords.mean(axis=0)
    _, _, Vt = np.linalg.svd(coords - mu, full_matrices=False)
    axis = Vt[0]
    proj = (coords - center) @ axis
    coords_pos = coords[proj > 0]
    coords_neg = coords[proj < 0]
    if len(coords_pos) < 10 or len(coords_neg) < 10:
        raise RuntimeError("One side too small")
    v_pos = (coords_pos.mean(axis=0) - center)
    v_neg = (coords_neg.mean(axis=0) - center)
    v_pos /= np.linalg.norm(v_pos)
    v_neg /= np.linalg.norm(v_neg)

    def trace_one_side(v_main):
        v_perp = np.array([-v_main[1], v_main[0]])
        proj = (coords - center) @ v_main
        t_max = proj.max()
        if t_max <= 1: return []
        ts = np.linspace(0, t_max, num_samples)
        line = []
        H, W = mask.shape
        for t in ts:
            base = center + t * v_main
            samples = []
            for s in range(-slice_half_width, slice_half_width + 1):
                p = base + s * v_perp
                x, y = int(round(p[0])), int(round(p[1]))
                if 0 <= x < W and 0 <= y < H and mask[y, x]:
                    samples.append([x, y])
            if len(samples) < 2: break
            line.append(np.asarray(samples, dtype=float).mean(axis=0))
        return line

    line_pos = trace_one_side(v_pos)
    line_neg = trace_one_side(v_neg)
    if len(line_pos) < 3 or len(line_neg) < 3:
        raise RuntimeError("Centerline extraction failed")
    line_neg = np.asarray(line_neg)[::-1]
    line_pos = np.asarray(line_pos)
    centerline = np.vstack([line_neg[:-1], line_pos])
    k = int(len(centerline) * drop_tail_ratio)
    if k > 0:
        centerline = centerline[k:-k]
    return centerline

def fit_centerline_polynomial_rc(centerline, max_degree=3, criterion="AIC"):
    pts_xy = centerline
    _, idx = np.unique(pts_xy, axis=0, return_index=True)
    pts_xy = pts_xy[np.sort(idx)]
    tck = fit_spline_from_skeleton(pts_xy)
    u, v, R, mu = spline_to_uv(tck)
    best = fit_poly_aic(u, v, max_degree=max_degree, criterion=criterion)
    coeffs = best["coeffs"]
    poly = np.poly1d(coeffs)

    def eval_rc(u_grid):
        u_grid = np.asarray(u_grid, float)
        v_grid = poly(u_grid)
        xy = (R @ np.column_stack([u_grid, v_grid]).T).T + mu
        return np.column_stack([xy[:, 1], xy[:, 0]])

    eq_terms = []
    for c, p in zip(coeffs, range(len(coeffs) - 1, -1, -1)):
        if abs(c) < 1e-12: continue
        eq_terms.append(f"{c:.6g}" if p == 0 else
                        f"{c:.6g} u" if p == 1 else f"{c:.6g} u^{p}")
    eq = "v = " + " + ".join(eq_terms).replace("+ -", "- ")
    return {"degree": best["degree"], "coeffs": coeffs, "R": R, "mu": mu,
            "eval_rc": eval_rc, "equation_uv_str": eq, "criterion": criterion}


def pca_direction_on_path(best_path):
    if best_path is None or len(best_path) < 5: return None
    pts = np.asarray(best_path, dtype=float)
    mu = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - mu)
    v = Vt[0]
    return v / (np.linalg.norm(v) + 1e-6)

def angle_between_dirs(v1, v2):
    if v1 is None or v2 is None: return np.pi
    dot = np.clip(abs(np.dot(v1, v2)), -1.0, 1.0)
    return np.arccos(dot)


