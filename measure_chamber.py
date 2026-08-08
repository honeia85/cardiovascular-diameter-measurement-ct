"""
Cardiac chamber (LA/LV/RA/RV) slice selection + diameter measurement.

The fixed slice-selection weights below were optimized against R1 slice indices in 59
development examinations by minimizing mean squared slice-index error. The LA and RA
second-stage corrections used the same development data. The resulting implementation
was then applied without tuning to the independent evaluation cohort.
"""

from utils import *
from sklearn.svm import SVC
import numpy as np


def sample_points_on_svm_line(w, b, mask, step=5):
    H, W = mask.shape
    wx, wy = w
    pts = []
    for y in range(0, H, step):
        if abs(wx) < 1e-6:
            continue
        x = -(wy * y + b) / wx
        x = int(round(x))
        if 0 <= x < W:
            pts.append((y, x))
    return pts


def sample_horizontal_line(r0, mask, step=5):
    H, W = mask.shape
    return [(r0, c) for c in range(0, W, step)]


def bias_updater(mask, w):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return 0.0
    y, x = ys.mean(), xs.mean()
    wx, wy = w
    return -x * wx - y * wy


def intersect_line_with_mask(r0, c0, n_rc, mask, max_length=200.0, step=0.25):
    H, W  = mask.shape
    ts    = np.arange(-max_length, max_length + step, step)
    hits  = []
    for t in ts:
        r, c   = r0 + n_rc[0] * t, c0 + n_rc[1] * t
        ir, ic = int(round(r)), int(round(c))
        hits.append(t if (0 <= ir < H and 0 <= ic < W and mask[ir, ic] > 0) else None)

    segments = []
    in_seg, t_start = False, None
    for t, hit in zip(ts, hits):
        if hit is not None and not in_seg:
            in_seg, t_start = True, t
        elif hit is None and in_seg:
            segments.append((t_start, t - step))
            in_seg = False
    if in_seg:
        segments.append((t_start, ts[-1]))

    best_seg, best_len, best_half = None, -np.inf, -np.inf
    for t1, t2 in segments:
        length = t2 - t1
        half   = abs(t2)
        if length <= best_len:
            continue
        r1, c1 = r0 + n_rc[0] * t1, c0 + n_rc[1] * t1
        r2, c2 = r0 + n_rc[0] * t2, c0 + n_rc[1] * t2
        best_len = length
        best_seg = (r1, c1, r2, c2, length)
        best_half = half
    return best_seg, best_len, best_half


def find_max_area_slice(vol, cls):
    areas = np.sum(vol == cls, axis=(0, 1))
    if np.all(areas == 0):
        raise ValueError(f"class {cls} not found")
    idx = int(np.argmax(areas))
    return (vol[:, :, idx] == cls).astype(np.uint8).T, idx


def find_weighted_max_area_slice_bonus(vol, cls, slice_indices, bonus=0.4):
    areas   = np.sum(vol == cls, axis=(0, 1)).astype(np.float32)
    weights = np.ones_like(areas)
    idx     = np.asarray(slice_indices, dtype=np.int32)
    idx     = idx[(0 <= idx) & (idx < vol.shape[2])]
    weights[idx] = 1.0 + bonus
    best = int(np.argmax(areas * weights))
    return (vol[:, :, best] == cls).astype(np.uint8).T, best


def find_left_atrium_slice(vol, cls, width_ratio=0.2, angle_thresh=0.1, bonus=0.4, plot=False):
    _, _, Z  = vol.shape
    mask3d   = (vol == cls)
    target_slices = []
    for z in range(Z):
        try:
            sl = mask3d[:, :, z].astype(np.uint8).T
            if sl.sum() == 0:
                continue
            contour = extract_main_contour(sl)
            r0, c0  = np.mean(contour, axis=0)
            width   = contour[:, 1].max() - contour[:, 1].min()
            m_col   = np.abs(contour[:, 1] - c0) <= width * width_ratio
            m_row   = contour[:, 0] < r0
            pts     = np.asarray(contour[m_col & m_row], dtype=np.float32)
            if len(pts) < 2:
                continue
            m = pts.mean(axis=0)
            _, _, vt = np.linalg.svd(pts - m, full_matrices=False)
            d = vt[0] / (np.linalg.norm(vt[0]) + 1e-12)
            if angle_between_dirs(d, [0, 1]) < angle_thresh:
                target_slices.append(z)
        except Exception:
            continue
    if target_slices:
        return find_weighted_max_area_slice_bonus(vol, cls, target_slices, bonus=bonus)
    return find_max_area_slice(vol, cls)


def find_max_weighted_slice(vol, cls1, cls2, cls3, w1=0.5, w2=0.25, w3=0.25):
    scores = (w1 * np.sum(vol == cls1, axis=(0, 1)) +
              w2 * np.sum(vol == cls2, axis=(0, 1)) +
              w3 * np.sum(vol == cls3, axis=(0, 1)))
    if np.all(scores == 0):
        raise ValueError("classes not found")
    idx = int(np.argmax(scores))
    return (vol[:, :, idx] == cls1).astype(np.uint8).T, idx


# ══════════════════════════════════════════════════════════════════════
# Slice selection parameters
# ══════════════════════════════════════════════════════════════════════
# Stage 1 weights
RA_W_3VAR = [0.676763816994483, 0.16081028540415993, 0.34479596581007654]   # RA, LA, RV
LA_W_FLAT = [0.15876489113791403, 0.2619876786666036, 0.16598341840945016]  # width_ratio, angle_thresh, bonus
LV_W_3VAR = [0.9993815876718859, 0.011786941704419247, 0.4574948263364531]   # LV, RV, LA
RV_W_3VAR = [0.0010522683787815123, 0.39614827320034884, 0.8127172666682201] # RV, LV, RA

# ──────────────────────────────────────────────────────────────────────
# Stage 2 correction (normalized + 1-slice iteration)
#   Shared method: at the Stage 1 slice (z0), evaluate two normalized indicators,
#   [area fraction] and [normalized LV rate of change]. While both stay above
#   threshold, step one slice cranially (+z) at a time.
#   - Both indicators are normalized for body size and voxel spacing.
#   - Each step is exactly one adjacent slice (resolution); the cumulative limit
#     is bounded in mm (win_mm).
#   LA uses its forward relative area change and chamber-area fraction.
#   RA uses forward relative area changes for the RA and adjacent LV.
# ──────────────────────────────────────────────────────────────────────
LA_CORR = dict(
    thr_rel_dla_fwd = -0.024239, # LA relative rate of change (forward diff), >=
    thr_la_frac     = 0.457699,  # la_frac (LA fraction within the slice), <=
    area_min        = 50.0,      # guards the relative-rate denominator: LA area floor (voxels)
    win_mm          = 10.0,
)
# LA: rel_dla_fwd >= AND la_frac <= ("LA still growing + still small" -> go up).
# This is a self-signal rule; LV is not used.
RA_CORR = dict(
    thr_rel_dra_fwd = -0.010438, # RA relative rate of change (forward diff), >=
    thr_rel_dlv_fwd = -0.016810, # LV relative rate of change (forward diff), >=
    area_min        = 50.0,      # guards the relative-rate denominator (RA/LV area floor)
    win_mm          = 10.0,
)
# RA: rel_dra_fwd >= AND rel_dlv_fwd >= ("RA not shrinking + LV remaining =
# AV junction not yet exited" -> go up). This uses the RA and adjacent LV signal.


def _rel_fwd_profile(vol, cd, spacing_z, area_min=50.0):
    """Per-slice relative rate of change (forward diff, normalized by own area).
       rel_d{X}_fwd = (A_X[z+1]-A_X[z])/spacing_z / A_X[z]  (monotonic, safe as a stop signal)
       plus area fractions (la_frac, ra_frac). If area < area_min the rate is 0
       (guards against denominator blow-up)."""
    A = {c: np.sum(vol == cd[c], axis=(0, 1)).astype(float) for c in ["LA", "LV", "RA", "RV"]}
    Z = len(A["RA"])
    total = A["LA"] + A["LV"] + A["RA"] + A["RV"] + 1e-8

    def fwd(area):
        d = np.zeros_like(area)
        d[:-1] = (area[1:] - area[:-1]) / spacing_z
        return d

    def rel(area):
        return np.where(area >= area_min, fwd(area) / (area + 1e-6), 0.0)

    return dict(
        rel_dla_fwd = rel(A["LA"]),
        rel_dra_fwd = rel(A["RA"]),
        rel_dlv_fwd = rel(A["LV"]),
        la_frac     = A["LA"] / total,
        ra_frac     = A["RA"] / total,
    ), Z


# ══════════════════════════════════════════════════════════════════════
# Slice selection  (Stage 1 + Stage 2 relative-rate iterative correction)
# ══════════════════════════════════════════════════════════════════════
def find_la_slice(vol, cd, spacing_z):
    """
    LA slice selection.
    Stage 1: find_left_atrium_slice (flat-contour method)
    Stage 2: relative rate of change (forward diff) + area fraction, iterative
             1-slice cranial correction
             condition: rel_dla_fwd >= thr_rel_dla_fwd AND la_frac <= thr_la_frac
             ("LA still growing (max cross-section not reached) + still small"
              -> correct cranially)
    Parameters were fixed using the development cohort. LV is unused.
    """
    p = LA_CORR
    _, z0 = find_left_atrium_slice(
        vol, cd["LA"],
        width_ratio=LA_W_FLAT[0], angle_thresh=LA_W_FLAT[1], bonus=LA_W_FLAT[2]
    )
    Z = vol.shape[2]
    rp, _ = _rel_fwd_profile(vol, cd, spacing_z, p["area_min"])
    max_steps = max(1, int(round(p["win_mm"] / spacing_z)))

    z = int(z0)
    for _ in range(max_steps):
        if rp["rel_dla_fwd"][z] >= p["thr_rel_dla_fwd"] and rp["la_frac"][z] <= p["thr_la_frac"] \
           and z + 1 <= Z - 1:
            z += 1
        else:
            break
    return z


def find_lv_slice(vol, cd, spacing_z):
    """LV slice selection using the fixed three-variable Optuna weights."""
    w1, w2, w3 = LV_W_3VAR
    a_lv = np.sum(vol == cd["LV"], axis=(0, 1)).astype(float)
    a_rv = np.sum(vol == cd["RV"], axis=(0, 1)).astype(float)
    a_la = np.sum(vol == cd["LA"], axis=(0, 1)).astype(float)
    return int(np.argmax(w1*a_lv + w2*a_rv + w3*a_la))


def find_ra_slice(vol, cd, spacing_z):
    """
    RA slice selection.
    Stage 1: 3-var Optuna (weighted sum of RA, LA, RV areas)
    Stage 2: relative rate of change (forward diff), iterative 1-slice cranial correction
             condition: rel_dra_fwd >= thr_rel_dra_fwd AND rel_dlv_fwd >= thr_rel_dlv_fwd
             ("RA not shrinking + LV remaining = AV junction not yet exited"
              -> correct cranially)
    Parameters were fixed using the development cohort.
    """
    p = RA_CORR
    w1, w2, w3 = RA_W_3VAR
    a_ra = np.sum(vol == cd["RA"], axis=(0, 1)).astype(float)
    a_la = np.sum(vol == cd["LA"], axis=(0, 1)).astype(float)
    a_rv = np.sum(vol == cd["RV"], axis=(0, 1)).astype(float)
    z0 = int(np.argmax(w1 * a_ra + w2 * a_la + w3 * a_rv))
    Z = len(a_ra)
    rp, _ = _rel_fwd_profile(vol, cd, spacing_z, p["area_min"])
    max_steps = max(1, int(round(p["win_mm"] / spacing_z)))

    z = int(z0)
    for _ in range(max_steps):
        if rp["rel_dra_fwd"][z] >= p["thr_rel_dra_fwd"] \
           and rp["rel_dlv_fwd"][z] >= p["thr_rel_dlv_fwd"] \
           and z + 1 <= Z - 1:
            z += 1
        else:
            break
    return z


def find_rv_slice(vol, cd, spacing_z):
    """RV slice selection using the fixed three-variable Optuna weights."""
    w1, w2, w3 = RV_W_3VAR
    a_rv = np.sum(vol == cd["RV"], axis=(0, 1)).astype(float)
    a_lv = np.sum(vol == cd["LV"], axis=(0, 1)).astype(float)
    a_ra = np.sum(vol == cd["RA"], axis=(0, 1)).astype(float)
    return int(np.argmax(w1*a_rv + w2*a_lv + w3*a_ra))


# ══════════════════════════════════════════════════════════════════════
# Measurement direction (SVM-based)
# ══════════════════════════════════════════════════════════════════════
def _svm_direction(pts_a, pts_b, mask, mode='normal'):
    if len(pts_a) < 3:
        ys, xs = np.where(mask > 0)
        pts_a = np.stack([xs, ys], axis=1).astype(float)
    if len(pts_b) < 3:
        return np.array([0.0, 1.0]), None
    X = np.vstack([pts_a, pts_b])
    y = np.hstack([np.zeros(len(pts_a)), np.ones(len(pts_b))])
    clf = SVC(kernel="linear", C=0.1, class_weight="balanced")
    clf.fit(X, y)
    w = clf.coef_[0]; b = clf.intercept_[0]
    n = w / np.linalg.norm(w)
    n_rc = np.array([n[1], n[0]])
    if mode == 'tangent':
        n_rc = np.array([-n_rc[1], n_rc[0]]); n_rc /= np.linalg.norm(n_rc)
    return n_rc, (w, b)


def _measure_la(vol, cd, axis, slice_idx, spacing):
    im   = slice_by_axis(vol, axis, slice_idx).T
    mask = (im == cd["LA"]).astype(np.uint8)
    if not np.any(mask):
        raise ValueError("LA mask empty")
    r0     = int(np.mean(np.where(mask > 0)[0]))
    n_rc   = np.array([1.0, 0.0])
    pts_rc = sample_horizontal_line(r0, mask, step=1)
    return mask, pts_rc, n_rc


def _measure_lv(vol, cd, axis, slice_idx, spacing):
    im    = slice_by_axis(vol, axis, slice_idx).T
    mask  = (im == cd["LV"]).astype(np.uint8)
    maskB = (im == cd["RV"]).astype(np.uint8)
    ys, xs = np.where(mask  > 0); ptsA = np.stack([xs, ys], axis=1).astype(float)
    ys, xs = np.where(maskB > 0); ptsB = np.stack([xs, ys], axis=1).astype(float)
    X = np.vstack([ptsA, ptsB]); y = np.hstack([np.zeros(len(ptsA)), np.ones(len(ptsB))])
    clf = SVC(kernel="linear", C=0.1, class_weight="balanced"); clf.fit(X, y)
    w = clf.coef_[0]
    n = w / np.linalg.norm(w); n_rc = np.array([n[1], n[0]])
    new_b = bias_updater(mask, w)
    pts_rc = sample_points_on_svm_line(w, new_b, mask, step=1)
    return mask, pts_rc, n_rc


def _measure_ra(vol, cd, axis, slice_idx, spacing):
    im    = slice_by_axis(vol, axis, slice_idx).T
    mask  = (im == cd["RA"]).astype(np.uint8)
    maskA = ((im == cd["RA"]) | (im == cd["RV"])).astype(np.uint8)
    maskB = ((im == cd["LV"]) | (im == cd["LA"])).astype(np.uint8)
    if not np.any(mask):
        raise ValueError("RA mask empty")
    ys, xs = np.where(maskA > 0); ptsA = np.stack([xs, ys], axis=1).astype(float)
    ys, xs = np.where(maskB > 0); ptsB = np.stack([xs, ys], axis=1).astype(float)
    n_rc, (w, b) = _svm_direction(ptsA, ptsB, mask, mode='normal')
    if n_rc[1] < 0: n_rc = -n_rc
    pts_rc = sample_points_on_svm_line(w, b, mask, step=1)
    return mask, pts_rc, n_rc


def _measure_rv(vol, cd, axis, slice_idx, spacing):
    im    = slice_by_axis(vol, axis, slice_idx).T
    mask  = (im == cd["RV"]).astype(np.uint8)
    maskB = (im == cd["LV"]).astype(np.uint8)
    ys, xs = np.where(mask  > 0); ptsA = np.stack([xs, ys], axis=1).astype(float)
    ys, xs = np.where(maskB > 0); ptsB = np.stack([xs, ys], axis=1).astype(float)
    n_rc, (w, b) = _svm_direction(ptsA, ptsB, mask, mode='normal')
    if n_rc[1] < 0: n_rc = -n_rc
    pts_rc = sample_points_on_svm_line(w, b, mask, step=1)
    return mask, pts_rc, n_rc


def _compute_measure(pts_rc, n_rc, mask, spacing, cls_name):
    lengths, segments, half_lens = [], [], []
    for (r, c) in pts_rc:
        seg, total_len, half_len = intersect_line_with_mask(
            r, c, n_rc=n_rc, mask=mask, max_length=200, step=0.25)
        if total_len > 0:
            lengths.append(total_len); segments.append(seg); half_lens.append(half_len)
    if not lengths:
        raise ValueError(f"{cls_name}: no valid measurement")
    l = len(lengths); l_ratio = max(1, int(l * 0.2)); l_m = l // 2
    lengths   = lengths  [l_m - l_ratio: l_m + l_ratio]
    segments  = segments [l_m - l_ratio: l_m + l_ratio]
    half_lens = half_lens[l_m - l_ratio: l_m + l_ratio]
    if cls_name == "LV":
        idx = int(np.argmin(half_lens))
        measure_coords = segments[idx][:4]; measure_value = lengths[idx] * spacing[0]
    else:
        idx = int(np.argmax(lengths))
        measure_coords = segments[idx][:4]; measure_value = lengths[idx] * spacing[0]
    return measure_coords, measure_value


def _process_chamber(args):
    cls_name, vol, cd, axis, spacing, affine, output_path = args
    sz_z = float(spacing[2])
    try:
        slice_fns = {
            "LA": lambda: find_la_slice(vol, cd, sz_z),
            "LV": lambda: find_lv_slice(vol, cd, sz_z),
            "RA": lambda: find_ra_slice(vol, cd, sz_z),
            "RV": lambda: find_rv_slice(vol, cd, sz_z),
        }
        measure_fns = {
            "LA": _measure_la, "LV": _measure_lv,
            "RA": _measure_ra, "RV": _measure_rv,
        }
        slice_idx = slice_fns[cls_name]()
        mask, pts_rc, n_rc = measure_fns[cls_name](vol, cd, axis, slice_idx, spacing)
        measure_coords, measure_value = _compute_measure(
            pts_rc, n_rc, mask, spacing, cls_name
        )
        p1 = voxel_yxz_to_LPS(np.float64(list(measure_coords[:2]) + [slice_idx]), affine)
        p2 = voxel_yxz_to_LPS(np.float64(list(measure_coords[2:]) + [slice_idx]), affine)
        save_line_markup_json(output_path + f"/{cls_name}.mrk.json", p1, p2, measure_value)
        return cls_name, True, None
    except Exception as e:
        return cls_name, False, str(e)


def run(input_path, output_path, targets=None, cd=None):
    """If cd is not given, labels are inferred with np.unique(vol)[3:] as before.
       That inference is only correct when labels 0-6 are all present (if any
       structure is missing the mapping shifts), so pipelines with a fixed label
       scheme should pass cd explicitly."""
    from concurrent.futures import ThreadPoolExecutor
    axis = 'axial'
    data, affine, header = load_nii(input_path)
    vol         = pick_volume3d(data)
    ALL_CLS     = ["LA", "LV", "RA", "RV"]
    if cd is None:
        cls_idx_arr = np.unique(vol)[3:]
        cd          = dict(zip(ALL_CLS, cls_idx_arr))
    spacing     = header.get_zooms()[:3]
    run_list = targets if targets is not None else ALL_CLS
    args_list = [(cls, vol, cd, axis, spacing, affine, output_path) for cls in run_list]
    with ThreadPoolExecutor(max_workers=len(args_list)) as exe:
        results = list(exe.map(_process_chamber, args_list))
    for cls_name, ok, err in results:
        if not ok:
            print(f"E [{cls_name}]: {input_path}  {err}")
    return results
