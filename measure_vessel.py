"""
Pulmonary artery (PT/RPA/LPA) and ascending aorta (AA) slice selection + diameter
measurement.

The fixed vessel-specific stable-segment parameters below were selected by GridSearch
with 5-fold cross-validation on the development cohort and then applied without tuning
to the independent evaluation cohort:
  PT  - rel: max_rel_slope=0.01, max_rel_std=0.01, min_stable_len=7
  RPA - rel: max_rel_slope=0.01, max_rel_std=0.01, min_stable_len=3
  LPA - rel: max_rel_slope=0.01, max_rel_std=0.02, min_stable_len=7
"""

import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import uniform_filter1d
from utils import *


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixed per-region find_stable_segment parameters
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARTERY_SEG_PARAMS = {
    'PT' : dict(mode='rel', max_rel_slope=0.01, max_rel_std=0.01, min_stable_len=7),
    'RPA': dict(mode='rel', max_rel_slope=0.01, max_rel_std=0.01, min_stable_len=3),
    'LPA': dict(mode='rel', max_rel_slope=0.01, max_rel_std=0.02, min_stable_len=7),
}


def deepest_points_by_three_arcs(mask, center_xy=None,
                                 smooth_sigma=5, min_sep_ratio=0.18,
                                 plot=True, n=0, save_path=None):
    m = (mask > 0).astype(np.uint8)
    m = mopology_noise_remover(m)
    if center_xy is None:
        M = cv2.moments(m, binaryImage=True)
        if M['m00'] == 0:
            raise ValueError("mask is empty or too small.")
        cx = M['m10'] / M['m00']; cy = M['m01'] / M['m00']
        center_rc = (cy, cx)
    else:
        center_rc = (float(center_xy[1]), float(center_xy[0]))

    contour_rc = extract_main_contour(m)
    cuts, mins, d_s, arcs = split_by_ridge_maxima(
        contour_rc, center_rc, smooth_sigma=smooth_sigma, min_sep_ratio=min_sep_ratio)

    N = len(contour_rc)
    min_indices = []; min_points_rc = []
    for arc in arcs:
        i_min, delta = gradient_descent_on_arc(d_s, list(arc))
        i_f = (i_min + delta) % N
        i0 = int(math.floor(i_f)); alpha = float(i_f - i0)
        p0 = contour_rc[i0 % N]; p1 = contour_rc[(i0+1) % N]
        p = (1-alpha)*p0 + alpha*p1
        min_indices.append(int(i_min))
        min_points_rc.append((float(p[0]), float(p[1])))

    ridges = [[int(contour_rc[i,0]), int(contour_rc[i,1])] for i in cuts]
    results = dict(
        center_rc=(float(center_rc[0]), float(center_rc[1])),
        contour_rc=contour_rc, cut_indices=cuts.tolist(),
        min_indices=min_indices, min_points_rc=min_points_rc,
        arcs=[np.array(a, dtype=int) for a in arcs], ridges=ridges, d_s=d_s)
    center_rc = np.mean(np.float32(min_points_rc), axis=0)
    return results


def choose_best_path(global_path, local_path, deg_th=10.0):
    """Use the local path when PCA directions agree within deg_th; otherwise use global."""
    vg = pca_direction_on_path(global_path)
    vl = pca_direction_on_path(local_path)
    ang = angle_between_dirs(vg, vl)
    if ang < np.deg2rad(deg_th):
        return local_path
    else:
        return global_path


def extract_best_path_via_centerline_and_astar(skel, centerline, center, ridge):
    centerline = np.asarray(centerline, dtype=float)
    c0 = centerline[0]; c1 = centerline[-1]
    if np.linalg.norm(np.asarray(c0)-np.asarray(ridge[::-1])) < \
       np.linalg.norm(np.asarray(c1)-np.asarray(ridge[::-1])):
        c = c0
    else:
        c = c1
    s0 = project_point_to_skel(center[::-1], skel)
    s1 = project_point_to_skel(c, skel)
    return astar_on_skel(skel, s0, s1), s0, s1


def project_point_to_skel(pt, skel):
    ys, xs = np.where(skel)
    skel_pts = np.stack([xs, ys], axis=1)
    d = np.linalg.norm(skel_pts - np.asarray(pt), axis=1)
    return tuple(skel_pts[np.argmin(d)])


def find_path(mask, center, ridge):
    cx, cy = rc_to_xy(center)
    centerline = extract_centerline_from_mask(
        mask, center=(cx,cy), num_samples=1000,
        slice_half_width=40, drop_tail_ratio=0.2)
    skel = skeletonize(mask > 0)
    best_path, s0, s1 = extract_best_path_via_centerline_and_astar(
        skel=skel, centerline=centerline, center=center, ridge=ridge)
    return best_path


def extend_path_by_direction(path, mask, step=1.0, max_steps=300, k_dir=30):
    path = np.asarray(path, dtype=float)
    mask = mask.astype(bool); H, W = mask.shape; parts = []
    v_head = estimate_direction(path, at_head=True, k=k_dir)
    if v_head is not None:
        ext = []; p = path[0].copy()
        for _ in range(max_steps):
            p = p+v_head*step; x,y = int(round(p[0])),int(round(p[1]))
            if not (0<=x<W and 0<=y<H and mask[y,x]): break
            ext.append(p.copy())
        if ext: parts.append(np.asarray(ext[::-1]))
    parts.append(path)
    v_tail = estimate_direction(path, at_head=False, k=k_dir)
    if v_tail is not None:
        ext = []; p = path[-1].copy()
        for _ in range(max_steps):
            p = p+v_tail*step; x,y = int(round(p[0])),int(round(p[1]))
            if not (0<=x<W and 0<=y<H and mask[y,x]): break
            ext.append(p.copy())
        if ext: parts.append(np.asarray(ext))
    return np.vstack(parts)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# find_stable_segment: two variants, abs / rel
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _find_stable_segment_abs(lens, smooth_window=5, win_size=5,
                              max_abs_slope=0.3, max_std=3.0, min_stable_len=5):
    """Absolute variant (longest segment)"""
    lens = np.asarray(lens, dtype=float)
    n = len(lens)
    if n < win_size: return None, None, (None, None)
    smooth = uniform_filter1d(lens, size=smooth_window, mode="nearest")
    windows = sliding_window_view(smooth, win_size)
    xs = np.arange(win_size, dtype=float); xm = xs.mean()
    xv = ((xs-xm)**2).sum()
    ym = windows.mean(axis=1)
    slopes = ((windows-ym[:,None])*(xs-xm)).sum(axis=1) / xv
    stds = windows.std(axis=1)
    stable = (np.abs(slopes) <= max_abs_slope) & (stds <= max_std)
    is_stable = np.zeros(n, dtype=bool)
    for i in np.where(stable)[0]: is_stable[i:i+win_size] = True
    best_start = best_end = None
    i = 0
    while i < n:
        if not is_stable[i]: i += 1; continue
        j = i
        while j < n and is_stable[j]: j += 1
        if j-i >= min_stable_len:
            if best_start is None or (j-i) > (best_end-best_start):
                best_start, best_end = i, j
        i = j
    if best_start is None: return None, None, (None, None)
    seg = lens[best_start:best_end]
    local_idx = int(np.argmax(seg))
    # Do not accept a zero-width (or negligible) segment as stable:
    #   an all-zero lens has slope=std=0 and would be mistaken for 'flat', which
    #   blocks the ret=True + zero-width bug.
    if float(seg[local_idx]) <= 0.0:
        return None, None, (None, None)
    return best_start+local_idx, float(seg[local_idx]), (best_start, best_end)


def _find_stable_segment_rel(lens, smooth_window=5, win_size=5,
                              max_rel_slope=0.02, max_rel_std=0.05, min_stable_len=5):
    """Relative variant (segment containing the max diameter)"""
    lens = np.asarray(lens, dtype=float)
    n = len(lens)
    if n < win_size: return None, None, (None, None)
    smooth = uniform_filter1d(lens, size=smooth_window, mode="nearest")
    scale = float(np.mean(smooth)) if np.mean(smooth) > 1e-6 else 1.0
    windows = sliding_window_view(smooth, win_size)
    xs = np.arange(win_size, dtype=float); xm = xs.mean()
    xv = ((xs-xm)**2).sum()
    ym = windows.mean(axis=1)
    slopes = ((windows-ym[:,None])*(xs-xm)).sum(axis=1) / xv
    stds = windows.std(axis=1)
    stable = (np.abs(slopes)/scale <= max_rel_slope) & (stds/scale <= max_rel_std)
    is_stable = np.zeros(n, dtype=bool)
    for i in np.where(stable)[0]: is_stable[i:i+win_size] = True
    best_start = best_end = None; best_max = -np.inf
    i = 0
    while i < n:
        if not is_stable[i]: i += 1; continue
        j = i
        while j < n and is_stable[j]: j += 1
        if j-i >= min_stable_len:
            seg_max = float(lens[i:j].max())
            if seg_max > best_max:
                best_max = seg_max; best_start, best_end = i, j
        i = j
    if best_start is None: return None, None, (None, None)
    seg = lens[best_start:best_end]
    local_idx = int(np.argmax(seg))
    if float(seg[local_idx]) <= 0.0:
        return None, None, (None, None)
    return best_start+local_idx, float(seg[local_idx]), (best_start, best_end)


def find_stable_segment(lens, artery='LPA', **kwargs):
    """Pick the variant for this region"""
    p = ARTERY_SEG_PARAMS.get(artery, ARTERY_SEG_PARAMS['LPA'])
    mode = p['mode']
    params = {k:v for k,v in p.items() if k != 'mode'}
    params.update(kwargs)
    if mode == 'rel':
        return _find_stable_segment_rel(lens, **params)
    else:
        return _find_stable_segment_abs(lens, **params)


def draw_lines(m, parab, r0, c0, mask, center, mode, case_n, idx,
               skel_label, ratio_dict, save_path, spacing,
               global_center, dist, ridge, neck, artery='LPA'):
    us, lens, details, ratios = draw_clipped_normals_and_measure(
        None, parab, start_r=r0, start_c=c0, mask=m,
        du=1.0, count_each_side=400, length=100.0,
        color='tab:red', linewidth=2.0, center_rc=center,
        only_inside=True, step_along_normal=0.25, return_details=True,
        skel_label=skel_label, ratio_dict=ratio_dict, slice_idx=idx,
        spacing=spacing, global_center=global_center,
        dist=dist, ridge=ridge, neck=neck)

    max_idx, max_val, (s, e) = find_stable_segment(lens, artery=artery)
    ret = False
    if max_idx is None:
        max_idx = int(lens.argmax()); max_val = float(lens.max())
    else:
        ret = True
    max_len_px = float(max_val) if len(lens) else 0.0
    coords = [(0,0),(0,0)]
    if details and details[max_idx][2]:
        r1, c1, r2, c2, _ = details[max_idx][2][0]
        coords = [(r1,c1),(r2,c2)]
    return max_len_px, ret, coords


def _process_single_slice(k, vol, axis, n, global_center, global_best_path,
                           partial_m, r0, c0, center, dist, spacing,
                           ridge, neck, artery='LPA', debug=False):
    """Measure along both the local and global paths and return both."""
    empty = (0.0, False, [(0, 0), (0, 0)])
    dbg = {'mask_px': 0, 'local_path_len': 0, 'err': None}
    try:
        im = slice_by_axis(vol, axis, k).T
        mask = im == n
        m = (mask > 0).astype(np.uint8)
        clean_m = mopology_noise_remover(m)
        m = cv2.bitwise_and(m, clean_m)
        m = cv2.bitwise_and(m, partial_m)
        dbg['mask_px'] = int(m.sum())

        local_path = find_path(m, global_center, ridge)
        dbg['local_path_len'] = len(local_path) if local_path is not None else 0

        out = {}
        for label, path in [('local', local_path), ('global', global_best_path)]:
            try:
                full_path = extend_path_by_direction(path=path, mask=m)
                parab = fit_centerline_polynomial_rc(full_path, max_degree=3)
                val, ret, coord = draw_lines(
                    m, parab, r0, c0, mask, center,
                    None, None, k, [], {}, None,
                    spacing, global_center, dist, ridge, neck, artery=artery)
                out[label] = (val, ret, coord)
            except Exception as e:
                out[label] = empty
                dbg['err'] = f"{label}:{type(e).__name__}:{e}"
        if debug: out['_dbg'] = dbg
        return k, out
    except Exception as e:
        dbg['err'] = f"outer:{type(e).__name__}:{e}"
        res = {'local': empty, 'global': empty}
        if debug: res['_dbg'] = dbg
        return k, res


def proj_mask(vol, j, n, window_size=10):
    D = vol.shape[2]
    j0 = max(0, j-window_size); j1 = min(D-1, j+window_size)
    win_vol = vol[:,:,j0:j1+1].transpose(2,0,1)
    proj = np.sum(win_vol == n, axis=0)
    return (proj > 0).astype(np.uint8).T


def measure_artery(input_path, save_path, max_workers=8, debug=False):
    dist = (1.05, 18.46)
    save_name = ['PT', 'RPA', 'LPA']
    axis = 'axial'
    data, affine, header = load_nii(input_path)
    vol = pick_volume3d(data)
    center_xy = None; X, Y, Z = vol.shape; n = 1
    spacing = header.get_zooms()[:3]
    window_size = int(round(25 / spacing[2]))
    if debug: print(f"\n[DEBUG] input={os.path.basename(input_path)}  Z={Z} spacing={tuple(round(s,3) for s in spacing)} window={window_size}")

    sum_masks = []
    for s in range(Z):
        im = slice_by_axis(vol, axis, s).T; mask = im == n
        m = (mask > 0).astype(np.uint8)
        try:
            contour_rc = extract_main_contour(m)
            center_rc = np.mean(contour_rc, axis=0) if center_xy is None \
                        else (float(center_xy[1]), float(center_xy[0]))
            r, c = contour_rc[:,0], contour_rc[:,1]
            d = np.hypot(r-center_rc[0], c-center_rc[1])
            d_s = gaussian_filter1d(d, sigma=5, mode='wrap')
            mins = find_mins(d_s)
            sum_masks.append([len(mins), len(contour_rc)])
        except Exception:
            sum_masks.append([0,0])

    sum_masks = np.array(sum_masks)
    idx = np.where(sum_masks[:,0] == np.max(sum_masks[:,0]))
    j = idx[0][np.argmax(sum_masks[idx][:,1])]
    if debug: print(f"[DEBUG] reference slice j={j} (max mins={sum_masks[:,0].max()})")

    proj_m = proj_mask(vol, j, n, window_size=window_size)
    if debug: print(f"[DEBUG] proj_m pixels={int(proj_m.sum())}")
    result = deepest_points_by_three_arcs(proj_m, plot=False)
    ridge_df = pd.DataFrame(result['ridges'], columns=['y','x'])
    head_index, left_index, right_index = pick_indices(ridge_df)
    if debug: print(f"[DEBUG] ridges={len(ridge_df)}, min_points={len(result['min_points_rc'])}, "
                    f"idx(H/L/R)={head_index}/{left_index}/{right_index}")

    modes = ["Head", "Left", "Right"]
    modes_index = [head_index, left_index, right_index]
    head_idx = j

    for t in range(3):
        artery = save_name[t]
        try:
            mode = modes[t]; mode_index = modes_index[t]
            ridge = ridge_df.iloc[mode_index].to_numpy()
            neck_df = pd.DataFrame(result['min_points_rc'], columns=['y','x'])
            nearest_neck_index = two_nearest_indices(neck_df, ridge)
            neck = neck_df.iloc[nearest_neck_index].to_numpy()
            r0, c0 = np.mean(neck, axis=0)
            center = np.mean(neck, axis=0)
            global_center = (center[0], center[1])
            center = np.mean(np.float32(result['min_points_rc']), axis=0)
            partial_m = partial_mask(proj_m, result, nearest_neck_index, neck, center)
            global_best_path = find_path(partial_m, global_center, ridge)
            if debug:
                gp_len = len(global_best_path) if global_best_path is not None else 0
                print(f"[DEBUG][{artery}] partial_m pixels={int(np.sum(partial_m))} "
                      f"global_path length={gp_len} ridge={ridge.tolist()}")

            slice_range = [k for k in range(j-window_size, j+window_size+1) if 0<=k<Z]

            raw_results = {}
            slice_dbgs = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_k = {
                    executor.submit(
                        _process_single_slice,
                        k, vol, axis, n, global_center, global_best_path,
                        partial_m, r0, c0, center, dist, spacing, ridge, neck,
                        artery, debug
                    ): k for k in slice_range
                }
                for future in as_completed(future_to_k):
                    k_val, res_dict = future.result()
                    if '_dbg' in res_dict:
                        slice_dbgs[k_val] = res_dict.pop('_dbg')
                    raw_results[k_val] = res_dict

            if debug:
                px = [d['mask_px'] for d in slice_dbgs.values()]
                lp = [d['local_path_len'] for d in slice_dbgs.values()]
                errs = [f"k{k}:{d['err']}" for k,d in slice_dbgs.items() if d['err']]
                print(f"[DEBUG][{artery}] slice mask_px range=[{min(px) if px else 0},{max(px) if px else 0}] "
                      f"local_path_len range=[{min(lp) if lp else 0},{max(lp) if lp else 0}]")
                if errs:
                    print(f"[DEBUG][{artery}] {len(errs)} slice exception(s): {errs[:5]}")

            # Compare the max over all slices -> pick the smaller of the two paths
            max_local  = max((raw_results[k]['local'][0]  for k in raw_results), default=0)
            max_global = max((raw_results[k]['global'][0] for k in raw_results), default=0)
            chosen = 'local' if max_local <= max_global else 'global'

            lengths_idx = sorted(raw_results.keys())
            lengths = np.array([raw_results[k][chosen][0] for k in lengths_idx])
            rets    = np.array([raw_results[k][chosen][1] for k in lengths_idx], dtype=bool)
            total_coords = [raw_results[k][chosen][2] for k in lengths_idx]
            if debug:
                nz = int((lengths > 0).sum())
                print(f"[DEBUG][{artery}] slices={len(lengths_idx)} max_local={max_local:.2f} "
                      f"max_global={max_global:.2f} chosen={chosen} | "
                      f"slices with lengths>0={nz}/{len(lengths)} ret=True={int(rets.sum())} "
                      f"lengths_max={lengths.max() if len(lengths) else 0:.2f}")

            # Only slices with ret=True (stable segment found) and width>0 are candidates
            # (guards against zero width)
            finite_positive = np.isfinite(lengths) & (lengths > 0)
            valid = rets & finite_positive
            if np.any(valid):
                cand = np.where(valid)[0]
                measure_idx = int(cand[np.argmax(lengths[cand])])
                src = 'stable(ret=True)'
            elif np.any(finite_positive):
                cand = np.where(finite_positive)[0]
                measure_idx = int(cand[np.argmax(lengths[cand])])
                src = 'fallback(argmax over all)'
            else:
                raise ValueError(f"{artery}: no positive finite measurement")

            measure_coords = total_coords[measure_idx]
            measure_value = lengths[measure_idx] * spacing[0]
            if debug:
                print(f"[DEBUG][{artery}] measured slice={lengths_idx[measure_idx]} "
                      f"({src}) len_px={lengths[measure_idx]:.2f} → value={measure_value:.3f}mm "
                      f"coords={measure_coords}")
                if measure_value == 0:
                    print(f"[DEBUG][{artery}] value=0 cause: "
                          f"{'all slices measured 0 (path/mask problem)' if max(max_local,max_global)==0 else 'argmax picked a slice measuring 0'}")

            phys1 = voxel_yxz_to_LPS(
                np.float64(list(measure_coords[0]) + [lengths_idx[measure_idx]]), affine)
            phys2 = voxel_yxz_to_LPS(
                np.float64(list(measure_coords[1]) + [lengths_idx[measure_idx]]), affine)
            save_line_markup_json(save_path + f"/{save_name[t]}.mrk.json",
                                   phys1, phys2, measure_value)
            if mode == "Head":
                head_idx = lengths_idx[measure_idx]
        except Exception as e:
            if debug:
                import traceback
                print(f"[DEBUG][{artery}] exception: {e}")
                traceback.print_exc()
            continue

    return head_idx


def measure_aorta(input_path, save_path, head_idx, axis='axial'):
    try:
        data, affine, header = load_nii(input_path)
        spacing = header.get_zooms()[:3]
        vol = pick_volume3d(data)
        im = slice_by_axis(vol, axis, head_idx).T
        mask = im == 2; m = (mask > 0).astype(np.uint8)
        info = analyze_two_masks_from_single_mask(
            m, spacing=spacing[0], connectivity=1, do_plot=False)
        if info['centroid1'][0] < info['centroid2'][0]:
            AA_value = info['width1_phys']; AA_coords = info['segment1_endpoints']
        else:
            AA_value = info['width2_phys']; AA_coords = info['segment2_endpoints']
        phys1 = voxel_yxz_to_LPS(np.float64(list(AA_coords[0])+[head_idx]), affine)
        phys2 = voxel_yxz_to_LPS(np.float64(list(AA_coords[1])+[head_idx]), affine)
        save_line_markup_json(save_path+"/AA.mrk.json", phys1, phys2, AA_value)
    except Exception:
        pass
