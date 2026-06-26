"""Per-tile morphology feature extraction for actin organization ranking.

Goal: capture filament topology / organization (boundaries, puncta, compact
assemblies, protrusions, texture complexity) that survive the dataset's matched
confounds (brightness, gradient strength, dark-pixel fraction).
"""
import numpy as np
from skimage import filters, feature, morphology, measure
from skimage.filters import sato, frangi, meijering, gabor_kernel
from skimage.morphology import disk
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings("ignore")

_GABOR = None
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


FEATURE_KEYS = None  # populated on first extract via caller


if __name__ == "__main__":
    # quick self-test on one image
    import sys
    from PIL import Image
    p = sys.argv[1] if len(sys.argv) > 1 else None
    if p:
        im = np.array(Image.open(p))
        d = extract(im)
        print(f"{len(d)} features")
        for k, v in d.items():
            print(f"  {k:24s} {v:.4f}")
