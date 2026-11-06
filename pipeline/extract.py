"""Extract features from every recording. Usage:
    python -I pipeline/extract.py <data_dir> <out.npz>
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import features, load, parse_name, rms  # noqa: E402


def one(path: str):
    d = load(path)
    meta = parse_name(path)
    meta["label"] = 1 if "/bad/" in path else 0
    meta["path"] = path
    dt = d - d.mean(axis=0)
    meta["rms_vec"] = rms(np.linalg.norm(dt, axis=1))
    return meta, features(d)


def main(data_dir: str, out: str):
    files = sorted(str(p) for p in Path(data_dir).rglob("*.h5"))
    with Pool(2) as pool:
        res = pool.map(one, files, chunksize=16)
    metas = [m for m, _ in res]
    X = np.vstack([f for _, f in res])
    np.savez_compressed(out, X=X,
                        machine=np.array([m["machine"] for m in metas]),
                        op=np.array([m["op"] for m in metas]),
                        tf=np.array([m["tf_index"] for m in metas]),
                        label=np.array([m["label"] for m in metas]),
                        rms=np.array([m["rms_vec"] for m in metas]),
                        path=np.array([m["path"] for m in metas]))
    print(f"{len(files)} recordings, {X.shape[1]} features -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
