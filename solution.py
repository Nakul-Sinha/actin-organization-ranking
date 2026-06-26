"""Microscopy Actin Pairwise Organization Ranking — official solution.

TRANSFER-ROBUST confound-orthogonal Bradley-Terry ranker (self-contained,
CPU-only, deterministic, no network). Reads ./dataset/public/ (falls back to
./dataset/), writes ./working/submission.csv.

The test tiles are distribution-shifted from train, and the pairs are matched on
intensity/texture/gradient/coverage. A model that uses the confounds OR the
shift-sensitive raw feature scales fails on the matched, shifted test set. So:
  1. extract 155 per-tile morphology/topology features;
  2. rank-normalize each feature within its own set (train among train, test among
     test) -> removes the marginal train/test distribution shift;
  3. drop the 50% most train/test-shifted features;
  4. residualize against a confound basis + orthogonalize the test pair logits
     against the pair's confound differences (=> ~0 confound correlation);
  5. calibrate the temperature on a SIMULATED train->test shift (a train-vs-test
     direction splits train tiles into train-like/test-like halves; train on one,
     score the other), mildly conservative.
Metric: gap-weighted pair log loss. Every model emits a per-tile score so
predictions stay consistent across the reused-tile graph.
"""
import warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import filters, feature, morphology, measure
from skimage.filters import sato, frangi, meijering, gabor_kernel
from skimage.morphology import disk
from sklearn.linear_model import LogisticRegression

_GABOR = None
CONF_FEATS = ["int_mean", "int_std", "int_max", "dark_frac", "nz_frac", "grad_mean",
              "grad_std", "glcm_contrast_mean", "glcm_contrast_std",
              "glcm_dissimilarity_mean", "shannon_entropy", "fg_frac", "blob_n",
              "cc_area_sum"]
KEEP_PCT, C_LIN, T_MULT, CLIP = 0.5, 0.01, 1.25, 0.30



# ===== features.py =====

def _gabor_bank():
    global _GABOR
    if _GABOR is None:
        ks = []
        for theta in np.arange(0, np.pi, np.pi / 6):       # 6 orientations
            for freq in [0.1, 0.2, 0.35]:                   # 3 frequencies
                ks.append((np.real(gabor_kernel(freq, theta=theta)), theta, freq))
        _GABOR = ks
    return _GABOR

def _safe(x):
    x = float(x)
    if not np.isfinite(x):
        return 0.0
    return x

def _stats(vals, prefix):
    vals = np.asarray(vals, dtype=np.float64).ravel()
    if vals.size == 0:
        return {f"{prefix}_{s}": 0.0 for s in ["mean", "std", "p50", "p90", "p99", "max", "sum"]}
    return {
        f"{prefix}_mean": _safe(vals.mean()),
        f"{prefix}_std": _safe(vals.std()),
        f"{prefix}_p50": _safe(np.percentile(vals, 50)),
        f"{prefix}_p90": _safe(np.percentile(vals, 90)),
        f"{prefix}_p99": _safe(np.percentile(vals, 99)),
        f"{prefix}_max": _safe(vals.max()),
        f"{prefix}_sum": _safe(vals.sum()),
    }

def extract(img_u8):
    """img_u8: HxW uint8 grayscale. Returns dict of scalar features."""
    f = {}
    img = img_u8.astype(np.float64)
    H, W = img.shape
    npix = H * W

    # --- intensity baseline (confounds; included so model can residualize) ---
    f["int_mean"] = _safe(img.mean())
    f["int_std"] = _safe(img.std())
    mx = img.max()
    imn = img / (mx + 1e-9)  # normalized to [0,1] by own max -> intensity-robust
    f["int_max"] = _safe(mx)
    f["dark_frac"] = _safe((img_u8 == 0).mean())
    f["nz_frac"] = _safe((img_u8 > 0).mean())

    # gradient magnitude (confound)
    gx = ndi.sobel(imn, axis=1); gy = ndi.sobel(imn, axis=0)
    gmag = np.hypot(gx, gy)
    f["grad_mean"] = _safe(gmag.mean())
    f["grad_std"] = _safe(gmag.std())

    # --- foreground mask (Otsu on normalized) for topology ---
    fg_frac = 0.0
    try:
        t = filters.threshold_otsu(imn) if imn.max() > imn.min() else 1.0
    except Exception:
        t = 0.5
    mask = imn > t
    fg_frac = mask.mean()
    f["otsu_t"] = _safe(t)
    f["fg_frac"] = _safe(fg_frac)

    # connected components on mask -> fragmentation / compactness topology
    lab = measure.label(mask, connectivity=2)
    nobj = int(lab.max())
    f["cc_n"] = float(nobj)
    f["cc_density"] = _safe(nobj / (mask.sum() + 1e-6))  # objects per fg pixel = fragmentation
    if nobj > 0:
        props = measure.regionprops(lab, intensity_image=imn)
        areas = np.array([p.area for p in props], dtype=np.float64)
        ecc = np.array([p.eccentricity for p in props], dtype=np.float64)
        solidity = np.array([p.solidity for p in props], dtype=np.float64)
        extent = np.array([p.extent for p in props], dtype=np.float64)
        perim = np.array([p.perimeter for p in props], dtype=np.float64)
        # shape complexity: perimeter^2 / area (circularity inverse)
        compl = (perim ** 2) / (areas + 1e-6)
        f.update(_stats(areas, "cc_area"))
        f.update(_stats(ecc, "cc_ecc"))
        f.update(_stats(solidity, "cc_solidity"))
        f.update(_stats(extent, "cc_extent"))
        f.update(_stats(compl, "cc_compl"))
        f["cc_area_gini"] = _safe(_gini(areas))
        f["cc_big_frac"] = _safe(areas.max() / (areas.sum() + 1e-6))  # dominance of largest blob
    else:
        for pre in ["cc_area", "cc_ecc", "cc_solidity", "cc_extent", "cc_compl"]:
            f.update(_stats([], pre))
        f["cc_area_gini"] = 0.0
        f["cc_big_frac"] = 0.0

    # --- skeleton topology (branchiness / length) ---
    try:
        skel = morphology.skeletonize(mask)
        sk_len = skel.sum()
        f["skel_len"] = _safe(sk_len)
        f["skel_frac"] = _safe(sk_len / (mask.sum() + 1e-6))  # how filamentous vs blobby
        # branch points = skeleton pixels with >=3 neighbors
        if sk_len > 0:
            nb = ndi.convolve(skel.astype(np.uint8), np.ones((3, 3)), mode="constant") - 1
            nb = nb * skel
            f["skel_branch"] = _safe((nb >= 3).sum())
            f["skel_end"] = _safe((nb == 1).sum())
            f["skel_branch_rate"] = _safe((nb >= 3).sum() / (sk_len + 1e-6))
            f["skel_end_rate"] = _safe((nb == 1).sum() / (sk_len + 1e-6))
        else:
            f["skel_branch"] = f["skel_end"] = f["skel_branch_rate"] = f["skel_end_rate"] = 0.0
    except Exception:
        f["skel_len"] = f["skel_frac"] = f["skel_branch"] = f["skel_end"] = 0.0
        f["skel_branch_rate"] = f["skel_end_rate"] = 0.0

    # --- ridge / tubeness filters (filament detection at multiple scales) ---
    sig = [1, 2, 3]
    for name, fn in [("sato", sato), ("frangi", frangi), ("meij", meijering)]:
        try:
            if name == "frangi":
                r = fn(imn, sigmas=sig, black_ridges=False)
            else:
                r = fn(imn, sigmas=sig, black_ridges=False)
        except Exception:
            r = np.zeros_like(imn)
        r = np.nan_to_num(r)
        f.update(_stats(r, f"ridge_{name}"))
        # fraction of strong ridge response
        rt = r > (r.mean() + r.std() + 1e-9)
        f[f"ridge_{name}_strongfrac"] = _safe(rt.mean())

    # --- puncta / blob detection (punctate organization) ---
    try:
        blobs = feature.blob_log(imn, min_sigma=1, max_sigma=4, num_sigma=4, threshold=0.05)
        f["blob_n"] = float(len(blobs))
        f["blob_density"] = _safe(len(blobs) / (fg_frac * npix + 1e-6))
        if len(blobs):
            f.update(_stats(blobs[:, 2], "blob_sigma"))
        else:
            f.update(_stats([], "blob_sigma"))
    except Exception:
        f["blob_n"] = 0.0; f["blob_density"] = 0.0; f.update(_stats([], "blob_sigma"))

    # --- GLCM Haralick texture (organization vs random texture) ---
    try:
        q = np.clip((imn * 31).astype(np.uint8), 0, 31)  # 32 levels
        glcm = feature.graycomatrix(q, distances=[1, 2, 4], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                                    levels=32, symmetric=True, normed=True)
        for prop in ["contrast", "homogeneity", "energy", "correlation", "dissimilarity", "ASM"]:
            v = feature.graycoprops(glcm, prop)
            f[f"glcm_{prop}_mean"] = _safe(v.mean())
            f[f"glcm_{prop}_std"] = _safe(v.std())
    except Exception:
        for prop in ["contrast", "homogeneity", "energy", "correlation", "dissimilarity", "ASM"]:
            f[f"glcm_{prop}_mean"] = 0.0; f[f"glcm_{prop}_std"] = 0.0

    # --- Local Binary Pattern histogram (micro-texture) ---
    try:
        lbp = feature.local_binary_pattern((imn * 255).astype(np.uint8), P=8, R=1, method="uniform")
        hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
        for i, h in enumerate(hist):
            f[f"lbp_{i}"] = _safe(h)
    except Exception:
        for i in range(10):
            f[f"lbp_{i}"] = 0.0

    # --- entropy / complexity ---
    f["shannon_entropy"] = _safe(measure.shannon_entropy(img_u8))
    # gradient orientation histogram entropy (directionality)
    ang = (np.arctan2(gy, gx) + np.pi)  # [0,2pi)
    wsum = gmag.sum() + 1e-9
    hgt = mask  # weight by foreground
    oh, _ = np.histogram(ang[hgt], bins=18, range=(0, 2*np.pi), weights=gmag[hgt])
    oh = oh / (oh.sum() + 1e-9)
    f["orient_entropy"] = _safe(-(oh * np.log(oh + 1e-12)).sum())

    # fractal dimension (box counting) on mask -> structural complexity
    f["fractal_dim"] = _safe(_fractal_dimension(mask))

    # multi-threshold structure: how foreground count changes with threshold (organization profile)
    for i, qt in enumerate([0.5, 0.7, 0.9]):
        m = imn > qt
        l = measure.label(m, connectivity=2)
        f[f"thr_{int(qt*100)}_n"] = float(l.max())
        f[f"thr_{int(qt*100)}_frac"] = _safe(m.mean())

    # ============ TOPOLOGY / SHAPE / SPATIAL ARRANGEMENT (intensity-robust) ============
    # These survive the dataset's brightness/texture/gradient matching.

    # Euler number + holes (topology of foreground)
    try:
        eul = measure.euler_number(mask, connectivity=2)
        f["euler_number"] = _safe(eul)
        f["n_holes"] = _safe(max(0, nobj - eul))  # components - euler = holes
    except Exception:
        f["euler_number"] = 0.0; f["n_holes"] = 0.0

    # Structure tensor coherence (filament alignment vs random) at 2 scales
    for sg in [1.5, 3.0]:
        try:
            Axx, Axy, Ayy = feature.structure_tensor(imn, sigma=sg, order="rc")
            tmp = np.sqrt((Axx - Ayy) ** 2 + 4 * Axy ** 2)
            l1 = (Axx + Ayy + tmp) / 2; l2 = (Axx + Ayy - tmp) / 2
            coh = (l1 - l2) / (l1 + l2 + 1e-9)
            wmask = mask
            f[f"coh_{sg}_mean"] = _safe(coh[wmask].mean() if wmask.sum() else 0)
            f[f"coh_{sg}_std"] = _safe(coh[wmask].std() if wmask.sum() else 0)
        except Exception:
            f[f"coh_{sg}_mean"] = 0.0; f[f"coh_{sg}_std"] = 0.0

    # Gabor bank: orientation selectivity / energy spread (directional organization)
    try:
        energies = []
        for kern, th, fr in _gabor_bank():
            r = ndi.convolve(imn, kern, mode="reflect")
            energies.append(np.abs(r).mean())
        energies = np.array(energies).reshape(6, 3)  # [orient, freq]
        per_orient = energies.sum(1)
        f["gabor_total"] = _safe(energies.sum())
        f["gabor_orient_std"] = _safe(per_orient.std() / (per_orient.mean() + 1e-9))  # anisotropy
        f["gabor_orient_max"] = _safe(per_orient.max() / (per_orient.sum() + 1e-9))
        for fi in range(3):
            f[f"gabor_freq{fi}"] = _safe(energies[:, fi].mean())
    except Exception:
        f["gabor_total"] = f["gabor_orient_std"] = f["gabor_orient_max"] = 0.0
        for fi in range(3):
            f[f"gabor_freq{fi}"] = 0.0

    # Granulometry (pattern spectrum): structure size distribution via openings
    try:
        base = mask.sum() + 1e-6
        prev = mask.sum()
        spec = []
        for r in [1, 2, 3, 5, 8]:
            op = morphology.binary_opening(mask, disk(r)).sum()
            spec.append((prev - op) / base)  # fraction removed at this scale
            prev = op
        for i, v in enumerate(spec):
            f[f"granulo_{i}"] = _safe(v)
        f["granulo_residual"] = _safe(prev / base)  # large-structure fraction
    except Exception:
        for i in range(5):
            f[f"granulo_{i}"] = 0.0
        f["granulo_residual"] = 0.0

    # Distance transform: structure thickness distribution
    try:
        dt = ndi.distance_transform_edt(mask)
        dv = dt[mask]
        f.update(_stats(dv, "thick"))
    except Exception:
        f.update(_stats([], "thick"))

    # Spatial point pattern of component centroids (clustering vs dispersion)
    try:
        if nobj >= 3:
            cents = np.array([p.centroid for p in measure.regionprops(lab)])
            tree = cKDTree(cents)
            dd, _ = tree.query(cents, k=2)
            nnd = dd[:, 1]
            f["nnd_mean"] = _safe(nnd.mean())
            f["nnd_std"] = _safe(nnd.std())
            f["nnd_cv"] = _safe(nnd.std() / (nnd.mean() + 1e-9))
            # Clark-Evans clustering index R = observed mean NND / expected (random)
            area = H * W; dens = nobj / area
            expected = 1.0 / (2 * np.sqrt(dens + 1e-12))
            f["clark_evans"] = _safe(nnd.mean() / (expected + 1e-9))
            f["centroid_spread"] = _safe(cents.std(0).mean())
        else:
            for k in ["nnd_mean", "nnd_std", "nnd_cv", "clark_evans", "centroid_spread"]:
                f[k] = 0.0
    except Exception:
        for k in ["nnd_mean", "nnd_std", "nnd_cv", "clark_evans", "centroid_spread"]:
            f[k] = 0.0

    # Convex-hull fill: how spread vs compact the whole structure is
    try:
        if mask.sum() > 5:
            ch = morphology.convex_hull_image(mask)
            f["hull_fill"] = _safe(mask.sum() / (ch.sum() + 1e-6))
            f["hull_frac"] = _safe(ch.sum() / (H * W))
        else:
            f["hull_fill"] = 0.0; f["hull_frac"] = 0.0
    except Exception:
        f["hull_fill"] = 0.0; f["hull_frac"] = 0.0

    # Lacunarity (gliding-box) at 2 box sizes -> gappiness/heterogeneity
    for bs in [4, 8]:
        try:
            f[f"lacun_{bs}"] = _safe(_lacunarity(mask.astype(np.float64), bs))
        except Exception:
            f[f"lacun_{bs}"] = 0.0

    # Multi-scale LoG blob counts (puncta size profile)
    try:
        for j, (ms, Ms) in enumerate([(1, 2), (2, 4), (4, 7)]):
            b = feature.blob_log(imn, min_sigma=ms, max_sigma=Ms, num_sigma=3, threshold=0.04)
            f[f"blogscale_{j}"] = float(len(b))
    except Exception:
        for j in range(3):
            f[f"blogscale_{j}"] = 0.0

    return f

def _lacunarity(Z, box):
    H, W = Z.shape
    sums = []
    for i in range(0, H - box + 1, max(1, box // 2)):
        for j in range(0, W - box + 1, max(1, box // 2)):
            sums.append(Z[i:i + box, j:j + box].sum())
    sums = np.array(sums)
    m = sums.mean()
    if m <= 1e-9:
        return 0.0
    return float((sums.var() / (m ** 2)) + 1.0)

def _gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)

def _fractal_dimension(Z):
    Z = Z > 0
    if Z.sum() == 0:
        return 0.0
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
            np.arange(0, Z.shape[1], k), axis=1)
        return len(np.where((S > 0) & (S < k * k))[0])
    p = min(Z.shape)
    n = 2 ** np.floor(np.log2(p))
    n = int(np.log2(n))
    sizes = 2 ** np.arange(n, 1, -1)
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))
    counts = np.array(counts, dtype=np.float64)
    sizes = np.array(sizes, dtype=np.float64)
    valid = counts > 0
    if valid.sum() < 2:
        return 0.0
    coeffs = np.polyfit(np.log(sizes[valid]), np.log(counts[valid] + 1e-9), 1)
    return -coeffs[0]


# ===== build_robust.py =====

def metric(y, p, w):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return min(100.0, 100.0 * (w * (-(y*np.log(p)+(1-y)*np.log(1-p)))).sum()/w.sum())

def img_confounds(rel, root):
    a = np.array(Image.open(root / rel), dtype=np.float64)
    gx, gy = np.gradient(a)
    lap = np.abs(np.gradient(np.gradient(a, axis=0), axis=0)).mean()
    return np.array([a.mean(), a.std(), np.percentile(a, 99), a.max(), a.sum()/1e4,
                     (a > 0).mean(), (a > 50).mean(),
                     np.abs(gx).mean()+np.abs(gy).mean(), lap])

def residualize(F, conf_cols, fit_mask):
    C = F[:, conf_cols]
    Cz = (C - C[fit_mask].mean(0)) / (C[fit_mask].std(0) + 1e-8)
    Cz = np.c_[Cz, np.ones(len(Cz))]
    out = F.copy()
    for j in range(F.shape[1]):
        if j in conf_cols:
            out[:, j] = 0.0; continue
        coef, *_ = np.linalg.lstsq(Cz[fit_mask], F[fit_mask, j], rcond=None)
        out[:, j] = F[:, j] - Cz @ coef
    return out

def orthogonalize(logit, Dconf):
    """Remove linear component of logit explained by confound differences."""
    D = np.c_[Dconf, np.ones(len(Dconf))]
    beta, *_ = np.linalg.lstsq(D, logit, rcond=None)
    return logit - D @ beta

def pair_confdiff(pairs_left, pairs_right, confmap):
    return np.array([confmap[l] - confmap[r] for l, r in zip(pairs_left, pairs_right)])


# ===== build_transfer.py =====

def rank_norm(F, ref):
    out = np.empty_like(F)
    for j in range(F.shape[1]):
        sv = np.sort(ref[:, j])
        out[:, j] = np.searchsorted(sv, F[:, j], side="right") / (len(sv) + 1.0)
    return out

def directions(Ftr, Fte, n=6):
    X = np.vstack([Ftr, Fte]); lab = np.r_[np.zeros(len(Ftr)), np.ones(len(Fte))]
    mu = X.mean(0); sd = X.std(0) + 1e-8; Xz = (X - mu) / sd
    out = []; rng = np.random.RandomState(0)
    for _ in range(n):
        s = rng.choice(len(Xz), int(0.8 * len(Xz)), replace=False)
        out.append((LogisticRegression(C=0.05, max_iter=2000).fit(Xz[s], lab[s]).coef_.ravel(), mu, sd))
    return out

def fit_predict(Frank_tr, keep, conf_idx, fit_mask, L, R, y, w, trm, Dvam_idx=None):
    F = Frank_tr.copy(); F[:, ~keep] = 0.0
    Fr = residualize(F, conf_idx, fit_mask)
    Fz = (Fr - Fr[fit_mask].mean(0)) / (Fr[fit_mask].std(0) + 1e-8)
    clf = LogisticRegression(C=C_LIN, max_iter=4000)
    clf.fit((Fz[L] - Fz[R])[trm], y[trm], sample_weight=w[trm])
    s = Fz @ clf.coef_.ravel()
    return s, (s[L[trm]] - s[R[trm]]).std() + 1e-8

def core(tr, te, tiles, Ftr, te_tiles, Fte, keys, root):
    conf_idx = sorted(set(keys.index(k) for k in CONF_FEATS if k in keys))
    confmap = {t: img_confounds(t, root) for t in set(tiles) | set(te_tiles)}
    idx = {t: i for i, t in enumerate(tiles)}; teidx = {t: i for i, t in enumerate(te_tiles)}
    L = np.array([idx[t] for t in tr.left_image_path]); R = np.array([idx[t] for t in tr.right_image_path])
    y = tr.left_higher_organization.values.astype(int); w = tr.pair_weight.values.astype(float)
    Dall = pair_confdiff(tr.left_image_path.values, tr.right_image_path.values, confmap)
    smd = np.abs((Ftr.mean(0) - Fte.mean(0)) / (np.sqrt(0.5*(Ftr.var(0)+Fte.var(0))) + 1e-8))
    keep = smd <= np.percentile(smd, KEEP_PCT * 100)
    nT = len(tiles)

    # ---- calibrate temperature on the simulated-shift proxy ----
    POOL_lg, POOL_y, POOL_w = [], [], []
    for cf, mu, sd in directions(Ftr, Fte):
        proj = ((Ftr - mu) / sd) @ cf; order = np.argsort(proj)
        for frac in (0.38, 0.45):
            k = int(nT * frac); tl = np.zeros(nT, bool); vl = np.zeros(nT, bool)
            tl[order[:k]] = True; vl[order[-k:]] = True
            trm = np.array([(tl[l] and tl[r]) for l, r in zip(L, R)])
            vam = np.array([(vl[l] and vl[r]) for l, r in zip(L, R)])
            if trm.sum() < 30 or vam.sum() < 20:
                continue
            Frank = Ftr.copy(); Frank[tl] = rank_norm(Ftr, Ftr[tl])[tl]; Frank[vl] = rank_norm(Ftr, Ftr[vl])[vl]
            s, sdt = fit_predict(Frank, keep, conf_idx, tl, L, R, y, w, trm)
            lg = orthogonalize((s[L[vam]] - s[R[vam]]) / sdt, Dall[vam]); lg /= (np.std(lg) + 1e-8)
            POOL_lg.append(lg); POOL_y.append(y[vam]); POOL_w.append(w[vam])
    plg = np.concatenate(POOL_lg); py = np.concatenate(POOL_y); pw = np.concatenate(POOL_w)
    T_opt = min(np.linspace(0.5, 12, 120), key=lambda T: metric(py, 1/(1+np.exp(-plg/T)), pw))
    proxy_loss = metric(py, 1/(1+np.exp(-plg/T_opt)), pw)
    proxy_acc = ((plg > 0).astype(int) == py).mean()
    T_safe = T_opt * T_MULT
    print(f"[transfer] proxy shift: acc={proxy_acc:.3f} loss={proxy_loss:.2f} T_opt={T_opt:.2f} -> T_safe={T_safe:.2f}")

    # ---- final fit on ALL train tiles ----
    Frank_all = Ftr.copy(); Frank_all[:] = rank_norm(Ftr, Ftr)
    Fte_rank = rank_norm(Fte, Fte)
    allmask = np.ones(nT, bool)
    F = Frank_all.copy(); F[:, ~keep] = 0.0
    Fr = residualize(F, conf_idx, allmask)
    # residualize test (rank-normed) with train-fit confound model
    Fte_use = Fte_rank.copy(); Fte_use[:, ~keep] = 0.0
    Cc = Fr[:, conf_idx]  # zeros (residualized); use original rank confs for test removal
    # simpler: residualize test against train rank-confounds directly
    Cz = Frank_all[:, conf_idx]; mC = Cz.mean(0); sCc = Cz.std(0) + 1e-8
    Ctr = np.c_[(Cz - mC) / sCc, np.ones(nT)]; Cte = np.c_[(Fte_rank[:, conf_idx] - mC) / sCc, np.ones(len(Fte))]
    Fte_r = Fte_use.copy()
    for j in range(Fte.shape[1]):
        if (j in conf_idx) or (not keep[j]):
            Fte_r[:, j] = 0.0; continue
        coef, *_ = np.linalg.lstsq(Ctr, Frank_all[:, j], rcond=None); Fte_r[:, j] = Fte_rank[:, j] - Cte @ coef
    mu = Fr.mean(0); sd = Fr.std(0) + 1e-8; Fz = (Fr - mu) / sd; Fz_te = (Fte_r - mu) / sd
    clf = LogisticRegression(C=C_LIN, max_iter=4000); clf.fit(Fz[L] - Fz[R], y, sample_weight=w)
    coef = clf.coef_.ravel(); s_tr = Fz @ coef; s_te = Fz_te @ coef
    sdt = (s_tr[L] - s_tr[R]).std() + 1e-8
    Lt = np.array([teidx[t] for t in te.left_image_path]); Rt = np.array([teidx[t] for t in te.right_image_path])
    lg_te = (s_te[Lt] - s_te[Rt]) / sdt
    Dte = pair_confdiff(te.left_image_path.values, te.right_image_path.values, confmap)
    lg_te = orthogonalize(lg_te, Dte); lg_te /= (np.std(lg_te) + 1e-8)
    prob = np.clip(1/(1+np.exp(-lg_te / T_safe)), 0.5 - CLIP, 0.5 + CLIP)
    return tr, te, te_tiles, prob, Dte, dict(proxy_loss=proxy_loss, proxy_acc=proxy_acc, T_safe=T_safe)

def _extract_one(arg):
    rel, root = arg
    im = np.array(Image.open(Path(root) / rel), dtype=np.uint8)
    d = extract(im); keys = sorted(d.keys())
    return rel, np.array([d[k] for k in keys], dtype=np.float32), keys


def extract_features(tiles, root, workers=6):
    rows = {}; keys = None
    args = [(t, str(root)) for t in tiles]
    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for rel, vec, k in ex.map(_extract_one, args, chunksize=8):
                rows[rel] = vec; keys = k
    except Exception:
        for a in args:
            rel, vec, k = _extract_one(a); rows[rel] = vec; keys = k
    return np.stack([rows[t] for t in tiles]), keys


def main():
    t0 = time.time()
    ROOT = Path("dataset/public") if (Path("dataset/public") / "train.csv").exists() else Path("dataset")
    WORK = Path("working"); WORK.mkdir(exist_ok=True, parents=True)
    print(f"[solution] ROOT={ROOT}")
    tr = pd.read_csv(ROOT / "train.csv"); te = pd.read_csv(ROOT / "test.csv")
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    tiles = sorted(set(tr.left_image_path) | set(tr.right_image_path))
    te_tiles = sorted(set(te.left_image_path) | set(te.right_image_path))
    print("[solution] extracting features...")
    Ftr, keys = extract_features(tiles, ROOT)
    Fte, _ = extract_features(te_tiles, ROOT)
    print(f"[solution] features tr{Ftr.shape} te{Fte.shape} ({time.time()-t0:.0f}s)")

    tr, te, te_tiles, prob, Dte, info = core(tr, te, tiles, Ftr.astype(float),
                                             te_tiles, Fte.astype(float), keys, ROOT)
    cc = max(abs(np.corrcoef(Dte[:, i], prob)[0, 1]) for i in range(Dte.shape[1]))
    print(f"[solution] proxy-shift acc={info['proxy_acc']:.3f} loss={info['proxy_loss']:.2f} "
          f"T_safe={info['T_safe']:.2f} max|confound-corr|={cc:.3f}")

    sub = pd.DataFrame({"id": te["id"].values, "prob_left_higher_organization": prob})
    sub = sub.set_index("id").loc[sample["id"].values].reset_index()
    sub.to_csv(WORK / "submission.csv", index=False)
    assert list(sub.columns) == ["id", "prob_left_higher_organization"]
    assert len(sub) == len(sample) and sub["id"].is_unique and set(sub["id"]) == set(sample["id"])
    pv = sub["prob_left_higher_organization"].to_numpy()
    assert np.isfinite(pv).all() and (pv >= 0).all() and (pv <= 1).all()
    print(f"[solution] wrote {WORK/'submission.csv'} {sub.shape} "
          f"prob[min={pv.min():.3f} mean={pv.mean():.3f} max={pv.max():.3f} std={pv.std():.3f}] "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
