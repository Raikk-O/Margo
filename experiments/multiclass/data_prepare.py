
import os
import glob
from collections import defaultdict

import numpy as np

from traj_classify import extract_features, load_tracks_full, load_labels_local

CLASS_NAMES = ["normal", "deviate", "heading", "sog_noise"]

LOG_FEATURES = ["max_dstep", "max_step", "top5_dstep", "top5_step", "tortuosity"]


def _normalize_features(X_train, X_valid, X_test, names, log_features):
    
    X_train = np.asarray(X_train, dtype=np.float32).copy()
    X_valid = np.asarray(X_valid, dtype=np.float32).copy()
    X_test = np.asarray(X_test, dtype=np.float32).copy()
    log_idx = [names.index(f) for f in log_features if f in names]
    for X in (X_train, X_valid, X_test):
        for i in log_idx:
            X[:, i] = np.log1p(X[:, i])
    feat_min = X_train.min(0)
    feat_max = X_train.max(0)
    rng = feat_max - feat_min
    rng[rng == 0] = 1.0
    X_train = (X_train - feat_min) / rng
    X_valid = (X_valid - feat_min) / rng
    X_test = (X_test - feat_min) / rng
    return X_train, X_valid, X_test

_REGION_META = {
    "west":   (34.5, -126.0),
    "mexico": (25.2, -92.5),
}


def get_regions(data_base):
    return {name: (os.path.join(data_base, f"{name}_coast"), lat_min, lon_min)
            for name, (lat_min, lon_min) in _REGION_META.items()}


def load_multiclass(region_names, REGIONS):
    normal = []       
    classes = {1: [], 2: [], 3: []}
    for name in region_names:
        root, lat_min, lon_min = REGIONS[name]
        for f in ("ct_train.pkl", "ct_valid.pkl"):
            for tr in load_tracks_full(os.path.join(root, f)):
                normal.append((tr, lat_min, lon_min))
        for t in ("1", "2", "3"):
            tdir = os.path.join(root, f"type_{t}")
            for pd_ in sorted([p for p in glob.glob(os.path.join(tdir, "*")) if os.path.isdir(p)]):
                tp = os.path.join(pd_, "ct_test.pkl")
                lp = os.path.join(pd_, "test_labels.pkl")
                if not (os.path.exists(tp) and os.path.exists(lp)):
                    continue
                tracks = load_tracks_full(tp)
                labs = load_labels_local(lp)
                for tr, lb in zip(tracks, labs):
                    if int(lb[0]) == 0 and int(lb[1]) == 0:
                        continue
                    classes[int(t)].append((tr, lat_min, lon_min))
    abn_mmsi = set()
    for t in (1, 2, 3):
        for tr, _, _ in classes[t]:
            abn_mmsi.add(int(tr[0, 5]))
    normal = [x for x in normal if int(x[0][0, 5]) not in abn_mmsi]
    return normal, classes


def split_grouped(tracks, bounds, labels, test_r=0.2, valid_r=0.15, seed=42):
    
    rng = np.random.default_rng(seed)
    grp = defaultdict(list)
    for i, (tr, b) in enumerate(zip(tracks, bounds)):
        grp[(round(b[0]), int(tr[0, 5]))].append(i)
    grp_label = {m: int(np.bincount([labels[i] for i in idx]).argmax())
                 for m, idx in grp.items()}
    by_class = defaultdict(list)
    for m in grp:
        by_class[grp_label[m]].append(m)
    tr_idx, va_idx, te_idx = [], [], []
    for c in sorted(by_class):
        lst = by_class[c]
        rng.shuffle(lst)
        n = len(lst)
        n_te = int(n * test_r)
        n_va = int(n * valid_r)
        te = lst[:n_te]
        va = lst[n_te:n_te + n_va]
        tr = lst[n_te + n_va:]

        def idx(ms):
            return [i for m in ms for i in grp[m]]
        tr_idx += idx(tr)
        va_idx += idx(va)
        te_idx += idx(te)
    return tr_idx, va_idx, te_idx


def prepare_multiclass_data(data_base, regions="west,mexico", seed=42,
                            refresh=False, verbose=True):
    
    cache_path = os.path.join(data_base, f"traj_prepared_seed{seed}.npz")
    if os.path.exists(cache_path) and not refresh:
        d = np.load(cache_path, allow_pickle=True)
        names = list(d["feature_cols"])
        out = (names, (d["X_train"], d["y_train"]),
               (d["X_valid"], d["y_valid"]), (d["X_test"], d["y_test"]))
        if verbose:
            print(f"[Data] cache hit {cache_path}")
        return out

    REGIONS = get_regions(data_base)
    region_names = regions.split(",")
    normal, classes = load_multiclass(region_names, REGIONS)
    if verbose:
        print(f"[Data] normal={len(normal)} deviate={len(classes[1])} "
              f"heading={len(classes[2])} sog_noise={len(classes[3])}")

    tracks, bounds, labels = [], [], []
    for tr, lo, la in normal:
        tracks.append(tr); bounds.append((lo, la)); labels.append(0)
    for t in (1, 2, 3):
        for tr, lo, la in classes[t]:
            tracks.append(tr); bounds.append((lo, la)); labels.append(t)

    n_norm = len(normal)
    n_abn_max = max(len(classes[1]), len(classes[2]), len(classes[3]))
    if n_norm > n_abn_max * 3:
        rng = np.random.default_rng(seed)
        keep = rng.choice(n_norm, n_abn_max * 3, replace=False)
        mask = np.zeros(n_norm, dtype=bool)
        mask[keep] = True
        keep_set = set(i for i in range(n_norm) if mask[i])
        tracks2, bounds2, labels2 = [], [], []
        for i, (tr, b, lb) in enumerate(zip(tracks, bounds, labels)):
            if lb == 0 and i not in keep_set:
                continue
            tracks2.append(tr); bounds2.append(b); labels2.append(lb)
        tracks, bounds, labels = tracks2, bounds2, labels2
        if verbose:
            print(f"[Data] normal downsampled to {len(keep_set)} samples")

    y = np.asarray(labels)
    names = sorted(extract_features(tracks[0], *bounds[0]).keys())
    X = np.array([[extract_features(tracks[i], *bounds[i])[k] for k in names]
                  for i in range(len(tracks))], dtype=np.float32)
    if verbose:
        print(f"[Data] feature dim={X.shape[1]}, total samples={X.shape[0]}")
        print("[Data] per-class counts: " + " ".join(
            f"{CLASS_NAMES[c]}={int((y == c).sum())}" for c in range(4)))

    tr_idx, va_idx, te_idx = split_grouped(tracks, bounds, y, seed=seed)
    X_train, y_train = X[tr_idx], y[tr_idx]
    X_valid, y_valid = X[va_idx], y[va_idx]
    X_test, y_test = X[te_idx], y[te_idx]

    X_train, X_valid, X_test = _normalize_features(
        X_train, X_valid, X_test, names, LOG_FEATURES)

    np.savez(cache_path,
             X_train=X_train, y_train=y_train,
             X_valid=X_valid, y_valid=y_valid,
             X_test=X_test, y_test=y_test,
             feature_cols=np.array(names))
    if verbose:
        print(f"[Data] cache saved {cache_path} (normalized log1p+min-max)")
    return names, (X_train, y_train), (X_valid, y_valid), (X_test, y_test)
