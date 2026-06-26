"""Microscopy Actin Pairwise Organization Ranking — official solution.

Pipeline (self-contained, reads ./dataset/public/ or ./dataset/, writes
./working/submission.csv):
  1. Per-tile hand-crafted morphology/topology features (intensity-robust).
  2. Bradley-Terry latent organization score z per tile from weighted pairs.
  3. Two per-tile scorers: linear BT (LogReg on feature diff) and a pretrained
     resnet18 regressing z (heavy dropout/weight-decay + dihedral augmentation).
  4. Tile-disjoint OOF -> per-model temperature calibration + blend weight.
  5. Fit both on all data, predict test, calibrate, blend, write submission.

Each scorer outputs a per-tile score, so pairwise predictions are globally
consistent across the reused-tile graph. Metric: gap-weighted pair log loss.
"""
import warnings, math, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy import sparse
from skimage import filters, feature, morphology, measure
from skimage.filters import sato, frangi, meijering, gabor_kernel
from skimage.morphology import disk
from sklearn.linear_model import LogisticRegression
import torch, torch.nn as nn, torch.nn.functional as F
import timm

SEED = 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"
_GABOR = None



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


# ===== bt_scores.py =====

def fit_bt(tr, tiles, C=1.0, weight=True):
    idx = {t: i for i, t in enumerate(tiles)}
    n = len(tiles)
    rows, cols, vals = [], [], []
    for r, (_, row) in enumerate(tr.iterrows()):
        li = idx[row.left_image_path]; ri = idx[row.right_image_path]
        rows += [r, r]; cols += [li, ri]; vals += [1.0, -1.0]
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(len(tr), n))
    y = tr.left_higher_organization.values.astype(int)
    w = tr.pair_weight.values.astype(float) if weight else None
    clf = LogisticRegression(C=C, fit_intercept=False, max_iter=5000, solver="lbfgs")
    clf.fit(X, y, sample_weight=w)
    z = clf.coef_.ravel()
    return z, idx


# ===== cnn_regress.py (deep) =====

def preload(tiles):
    arr = np.zeros((len(tiles), 1, 128, 128), dtype=np.float32)
    for i, rel in enumerate(tiles):
        arr[i, 0] = np.array(Image.open(ROOT / rel), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).to(DEV)

def augment(x, train=True, rot_deg=25.0, trans=0.10, noise=0.03):
    if train:
        if torch.rand(1).item() < 0.5: x = torch.flip(x, dims=[3])
        if torch.rand(1).item() < 0.5: x = torch.flip(x, dims=[2])
        k = int(torch.randint(0, 4, (1,)).item())
        if k: x = torch.rot90(x, k, dims=[2, 3])
        B = x.shape[0]
        ang = (torch.rand(B, device=x.device) * 2 - 1) * (rot_deg * math.pi / 180)
        cos, sin = torch.cos(ang), torch.sin(ang)
        sc = 1.0 + (torch.rand(B, device=x.device) * 2 - 1) * 0.12
        tx = (torch.rand(B, device=x.device) * 2 - 1) * trans
        ty = (torch.rand(B, device=x.device) * 2 - 1) * trans
        theta = torch.zeros(B, 2, 3, device=x.device)
        theta[:, 0, 0] = cos*sc; theta[:, 0, 1] = -sin*sc; theta[:, 0, 2] = tx
        theta[:, 1, 0] = sin*sc; theta[:, 1, 1] = cos*sc; theta[:, 1, 2] = ty
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")
        if noise > 0: x = x + torch.randn_like(x) * noise
    mu = x.mean(dim=[2, 3], keepdim=True); sd = x.std(dim=[2, 3], keepdim=True) + 1e-5
    return (x - mu) / sd

def tta_views(x):
    out = []
    for fh in [False, True]:
        for k in range(4):
            v = torch.flip(x, dims=[3]) if fh else x
            if k: v = torch.rot90(v, k, dims=[2, 3])
            mu = v.mean(dim=[2, 3], keepdim=True); sd = v.std(dim=[2, 3], keepdim=True) + 1e-5
            out.append((v - mu) / sd)
    return out

class TimmReg(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=True, drop=0.4):
        super().__init__()
        self.bk = timm.create_model(backbone, pretrained=pretrained, num_classes=0,
                                    in_chans=1, global_pool="avg")
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(self.bk.num_features, 1))

    def forward(self, x):
        return self.head(self.bk(x)).squeeze(-1)

def make_model(kind, drop):
    if kind == "small":
        return SmallNet(drop=drop).to(DEV)
    return TimmReg(kind, pretrained=True, drop=drop).to(DEV)

@torch.no_grad()
def predict_z(model, imgs, tta=True):
    model.eval()
    if not tta:
        return model(augment(imgs, train=False)).cpu().numpy()
    acc = 0; vs = tta_views(imgs)
    for v in vs: acc = acc + model(v)
    return (acc/len(vs)).cpu().numpy()

# ----------------------------- metric / calibration -----------------------------
def metric(y, p, w):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return min(100.0, 100.0 * (w * loss).sum() / w.sum())


def fit_temp(y, logit, w):
    best_t, best_l = 1.0, 1e9
    for t in np.linspace(0.2, 8.0, 120):
        l = metric(y, 1 / (1 + np.exp(-logit / t)), w)
        if l < best_l:
            best_l, best_t = l, t
    return best_t, best_l


def _extract_one(arg):
    rel, root = arg
    im = np.array(Image.open(Path(root) / rel), dtype=np.uint8)
    d = extract(im)
    keys = sorted(d.keys())
    return rel, np.array([d[k] for k in keys], dtype=np.float32), keys


def extract_features(tiles, root, workers=6):
    rows = {}
    keys = None
    args = [(t, str(root)) for t in tiles]
    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for rel, vec, k in ex.map(_extract_one, args, chunksize=8):
                rows[rel] = vec; keys = k
    except Exception:
        for a in args:
            rel, vec, k = _extract_one(a); rows[rel] = vec; keys = k
    return np.stack([rows[t] for t in tiles]), keys


# ----------------------------- deep training -----------------------------
def train_deep(imgs, train_ids, z_tiles, epochs, n_seed, val_pairs=None,
               drop=0.5, lr=2e-3, wd=3e-2, bs=48, backbone="resnet18"):
    """Seed-ensembled resnet18 regression of z. Early-stop on val_pairs if given,
    else fixed epochs. Returns TTA-averaged per-tile scores over ALL imgs."""
    nT = imgs.shape[0]
    zhat = np.zeros(nT)
    zt = torch.tensor(z_tiles, dtype=torch.float32, device=DEV)
    ids = torch.tensor(train_ids, device=DEV)
    n = len(train_ids); steps = max(1, n // bs)
    for sd in range(n_seed):
        torch.manual_seed(SEED + sd); np.random.seed(SEED + sd)
        model = make_model(backbone, drop)
        lr_use = lr * 0.15
        opt = torch.optim.AdamW(model.parameters(), lr=lr_use, weight_decay=wd)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr_use, epochs=epochs,
                                                    steps_per_epoch=steps, pct_start=0.15)
        best = (1e9, None)
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(n, device=DEV)
            for s in range(steps):
                b = perm[s * bs:(s + 1) * bs]; gi = ids[b]
                x = augment(imgs[gi], train=True)
                loss = F.smooth_l1_loss(model(x), zt[b], beta=0.5)
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if val_pairs is not None and (ep + 1) % 5 == 0:
                zh = predict_z(model, imgs, tta=False)
                Lv, Rv, yv, wv = val_pairs
                p = 1 / (1 + np.exp(-(zh[Lv] - zh[Rv])))
                vl = metric(yv, p, wv)
                if vl < best[0]:
                    best = (vl, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if best[1] is not None:
            model.load_state_dict(best[1])
        zhat += predict_z(model, imgs, tta=True)
    return zhat / n_seed


# ----------------------------- orchestration -----------------------------
def main():
    t0 = time.time()
    ROOT = Path("dataset/public") if (Path("dataset/public") / "train.csv").exists() else Path("dataset")
    WORK = Path("working"); WORK.mkdir(exist_ok=True, parents=True)
    print(f"[solution] ROOT={ROOT} DEV={DEV}")
    tr = pd.read_csv(ROOT / "train.csv")
    te = pd.read_csv(ROOT / "test.csv")
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    tiles = sorted(set(tr.left_image_path) | set(tr.right_image_path))
    te_tiles = sorted(set(te.left_image_path) | set(te.right_image_path))
    idx = {t: i for i, t in enumerate(tiles)}; teidx = {t: i for i, t in enumerate(te_tiles)}

    print("[solution] extracting features...")
    Ftr, keys = extract_features(tiles, ROOT)
    Fte, _ = extract_features(te_tiles, ROOT)
    mu = Ftr.mean(0); sd = Ftr.std(0) + 1e-8
    Fz = (Ftr - mu) / sd; Fz_te = (Fte - mu) / sd
    print(f"[solution] features {Ftr.shape} ({time.time()-t0:.0f}s)")

    L = np.array([idx[t] for t in tr.left_image_path]); R = np.array([idx[t] for t in tr.right_image_path])
    y = tr.left_higher_organization.values.astype(np.float32); w = tr.pair_weight.values.astype(np.float32)
    Lt = np.array([teidx[t] for t in te.left_image_path]); Rt = np.array([teidx[t] for t in te.right_image_path])
    nT = len(tiles); nP = len(tr)
    btC = 0.5; C_lin = 0.01

    imgs = torch.from_numpy(np.stack([
        np.array(Image.open(ROOT / t), dtype=np.float32)[None] / 255.0 for t in tiles])).to(DEV)
    imgs_te = torch.from_numpy(np.stack([
        np.array(Image.open(ROOT / t), dtype=np.float32)[None] / 255.0 for t in te_tiles])).to(DEV)
    bank = torch.cat([imgs, imgs_te], 0)

    # ---- OOF for calibration + blend weight (tile-disjoint) ----
    N_SPLITS = 12; VAL_FRAC = 0.3
    oof = {"lin": np.zeros(nP), "deep": np.zeros(nP)}; cnt = np.zeros(nP)
    print("[solution] OOF calibration...")
    for spi in range(N_SPLITS):
        rng = np.random.RandomState(spi)
        perm = rng.permutation(nT); val = set(perm[:int(nT * VAL_FRAC)].tolist())
        ttrain = np.array([t not in val for t in range(nT)])
        trm = np.array([(l not in val and r not in val) for l, r in zip(L, R)])
        vam = np.array([(l in val and r in val) for l, r in zip(L, R)])
        if vam.sum() == 0:
            continue
        mu2 = Ftr[ttrain].mean(0); sd2 = Ftr[ttrain].std(0) + 1e-8
        Fz2 = (Ftr - mu2) / sd2
        sub = tr[trm].reset_index(drop=True)
        sub_tiles = sorted(set(sub.left_image_path) | set(sub.right_image_path))
        z_sub, sidx = fit_bt(sub, sub_tiles, C=btC)
        z_tiles = np.array([z_sub[sidx[t]] for t in sub_tiles], dtype=np.float32)
        z_tiles = (z_tiles - z_tiles.mean()) / (z_tiles.std() + 1e-8)
        train_ids = [idx[t] for t in sub_tiles]
        # linear BT
        clf = LogisticRegression(C=C_lin, max_iter=4000)
        clf.fit((Fz2[L] - Fz2[R])[trm], y[trm], sample_weight=w[trm])
        s = Fz2 @ clf.coef_.ravel()
        sdt = (s[L[trm]] - s[R[trm]]).std() + 1e-8
        oof["lin"][vam] += (s[L[vam]] - s[R[vam]]) / sdt
        # deep (2 seeds for a stable OOF temperature/blend, early-stop on val)
        vp = (L[vam], R[vam], y[vam], w[vam])
        zh = train_deep(imgs, train_ids, z_tiles, epochs=60, n_seed=2, val_pairs=vp)
        sdt = (zh[L[trm]] - zh[R[trm]]).std() + 1e-8
        oof["deep"][vam] += (zh[L[vam]] - zh[R[vam]]) / sdt
        cnt[vam] += 1
        print(f"  oof split {spi+1}/{N_SPLITS} ({time.time()-t0:.0f}s)")
    seen = cnt > 0
    for m in oof:
        oof[m][seen] /= cnt[seen]
    temps = {}; cal = {}
    for m in oof:
        t, l = fit_temp(y[seen], oof[m][seen], w[seen]); temps[m] = t
        cal[m] = 1 / (1 + np.exp(-oof[m] / t))
        print(f"  {m}: OOF={l:.2f} T={t:.2f}")
    # blend weight grid
    best = (1e9, 0.5)
    for wl in np.linspace(0, 1, 41):
        p = wl * cal["lin"][seen] + (1 - wl) * cal["deep"][seen]
        l = metric(y[seen], p, w[seen])
        if l < best[0]:
            best = (l, wl)
    wlin = best[1]
    print(f"[solution] blend w_lin={wlin:.2f} OOF={best[0]:.2f}")

    # ---- final fit on ALL data ----
    z_all, sidx = fit_bt(tr, tiles, C=btC)
    z_tiles = np.array([z_all[sidx[t]] for t in tiles], dtype=np.float32)
    z_tiles = (z_tiles - z_tiles.mean()) / (z_tiles.std() + 1e-8)
    clf = LogisticRegression(C=C_lin, max_iter=4000)
    clf.fit(Fz[L] - Fz[R], y, sample_weight=w); coef = clf.coef_.ravel()
    s_tr = Fz @ coef; s_te = Fz_te @ coef
    sdt = (s_tr[L] - s_tr[R]).std() + 1e-8
    p_lin = 1 / (1 + np.exp(-((s_te[Lt] - s_te[Rt]) / sdt) / temps["lin"]))

    print("[solution] final deep training...")
    zhat = train_deep(bank, list(range(nT)), z_tiles, epochs=50, n_seed=6)
    s_trd = zhat[:nT]; s_ted = zhat[nT:]
    sdt = (s_trd[L] - s_trd[R]).std() + 1e-8
    p_deep = 1 / (1 + np.exp(-((s_ted[Lt] - s_ted[Rt]) / sdt) / temps["deep"]))

    prob = np.clip(wlin * p_lin + (1 - wlin) * p_deep, 1e-6, 1 - 1e-6)

    sub = pd.DataFrame({"id": te["id"].values, "prob_left_higher_organization": prob})
    sub = sub.set_index("id").loc[sample["id"].values].reset_index()
    sub.to_csv(WORK / "submission.csv", index=False)
    # validate
    assert list(sub.columns) == ["id", "prob_left_higher_organization"]
    assert len(sub) == len(sample) and sub["id"].is_unique
    assert set(sub["id"]) == set(sample["id"])
    pv = sub["prob_left_higher_organization"].to_numpy()
    assert np.isfinite(pv).all() and (pv >= 0).all() and (pv <= 1).all()
    print(f"[solution] wrote {WORK/'submission.csv'} {sub.shape} "
          f"prob[min={pv.min():.3f} mean={pv.mean():.3f} max={pv.max():.3f}] ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
