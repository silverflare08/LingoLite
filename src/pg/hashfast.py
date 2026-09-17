"""Level 2: HashFast — a compact hashed-ngram embedding-bag classifier.

Architecture (fastText-style, deliberately shallow):

    text -> normalize -> {char 3-5 grams, word 1-2 grams}
         -> CRC32 hash into B buckets      (no vocabulary file, no OOV)
         -> embedding bag, mean pooled     (E: B x d)
         -> linear softmax                 (W: d x C, b: C)

Why this and not a transformer:
  * Parameter budget is 500M; this lands at ~6M, leaving headroom to grow.
  * Inference is one hash pass + one gather + one mean + one 3xd matmul.
    There is no attention, so cost is linear in message length with a tiny
    constant -> single-digit-millisecond latency on CPU is reachable, which
    a 500M-parameter transformer cannot do per-message on commodity CPU.
  * Hashing removes the OOV problem structurally: an unseen spelling still
    lands in buckets shared with its neighbours via overlapping char n-grams.

Trained with sparse SGD (only touched embedding rows are updated), which is
what keeps this trainable in pure NumPy on one core.
"""
import json
import zlib

import numpy as np

from .normalize import featurize


def _macro_f1(y_true, y_pred, n_classes):
    f = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(sum(f) / len(f))


def _hash(s: str, buckets: int) -> int:
    return zlib.crc32(s.encode("utf-8")) % buckets


class HashFast:
    def __init__(self, n_classes=3, buckets=200_000, dim=32, seed=0,
                 keep_emoji=True, keep_punct=True, cmin=2, cmax=4, wmax=2,
                 pooling="mean+max"):
        rng = np.random.default_rng(seed)
        self.buckets, self.dim, self.n_classes = buckets, dim, n_classes
        self.E = (rng.standard_normal((buckets, dim), dtype=np.float32)
                  * (1.0 / np.sqrt(dim))).astype(np.float32)
        self.pooling = pooling
        out_dim = dim * (2 if pooling == "mean+max" else 1)
        self.out_dim = out_dim
        self.W = np.zeros((out_dim, n_classes), dtype=np.float32)
        self.b = np.zeros(n_classes, dtype=np.float32)
        self.keep_emoji, self.keep_punct = keep_emoji, keep_punct
        self.cmin, self.cmax, self.wmax = cmin, cmax, wmax

    # ---------- encoding ----------
    def encode(self, text, pre_normalized=False):
        from .normalize import normalize
        t = text if pre_normalized else normalize(text, self.keep_emoji, self.keep_punct)
        grams = featurize(t, self.cmin, self.cmax, self.wmax)
        if not grams:
            return np.zeros(1, dtype=np.int64)
        return np.fromiter((_hash(g, self.buckets) for g in grams),
                           dtype=np.int64, count=len(grams))

    def encode_all(self, texts):
        return [self.encode(t) for t in texts]

    # ---------- forward ----------
    def _pool(self, batch_idx):
        lens = np.fromiter((len(a) for a in batch_idx), dtype=np.int64,
                           count=len(batch_idx))
        flat = np.concatenate(batch_idx)
        offs = np.zeros(len(lens), dtype=np.int64)
        np.cumsum(lens[:-1], out=offs[1:])
        rows = self.E[flat]
        summed = np.add.reduceat(rows, offs, axis=0)
        mean = summed / lens[:, None].astype(np.float32)
        if self.pooling != "mean+max":
            return mean, flat, offs, lens, None
        # Max pooling in parallel with mean. Rationale: an emoji contributes
        # 1-3 n-grams out of ~100, so mean pooling dilutes exactly the signal
        # the problem statement says inverts meaning. Max pooling lets a single
        # strong n-gram survive averaging.
        amax = np.empty((len(lens), self.dim), dtype=np.int64)
        mx = np.empty((len(lens), self.dim), dtype=np.float32)
        for i, (o, L) in enumerate(zip(offs, lens)):
            blk = rows[o:o + L]
            j = blk.argmax(axis=0)
            amax[i] = j + o
            mx[i] = blk[j, np.arange(self.dim)]
        return np.concatenate([mean, mx], axis=1), flat, offs, lens, amax

    def logits(self, batch_idx):
        h = self._pool(batch_idx)[0]
        return h @ self.W + self.b

    def predict(self, texts):
        enc = self.encode_all(texts)
        return np.argmax(self.logits(enc), axis=1)

    def predict_encoded(self, enc):
        return np.argmax(self.logits(enc), axis=1)

    # ---------- training ----------
    def fit(self, enc, y, epochs=12, lr=0.5, batch_size=64, seed=0,
            val=None, verbose=True, class_weight=None, restore_best=True):
        rng = np.random.default_rng(seed)
        n = len(enc)
        y = np.asarray(y)
        cw = np.ones(self.n_classes, dtype=np.float32)
        if class_weight == "balanced":
            counts = np.bincount(y, minlength=self.n_classes).astype(np.float32)
            cw = (n / (self.n_classes * np.maximum(counts, 1))).astype(np.float32)
        hist = []
        best = {"score": -1.0, "epoch": -1, "state": None}
        for ep in range(epochs):
            order = rng.permutation(n)
            cur_lr = lr * (1.0 - ep / max(epochs, 1)) + 1e-3
            tot_loss, seen = 0.0, 0
            for s in range(0, n, batch_size):
                sel = order[s:s + batch_size]
                bidx = [enc[i] for i in sel]
                yb = y[sel]
                h, flat, offs, lens, amax = self._pool(bidx)
                z = h @ self.W + self.b
                z -= z.max(axis=1, keepdims=True)
                ez = np.exp(z)
                p = ez / ez.sum(axis=1, keepdims=True)
                w = cw[yb]
                loss = -np.log(np.maximum(p[np.arange(len(yb)), yb], 1e-9))
                tot_loss += float((loss * w).sum()); seen += len(yb)

                dz = p.copy()
                dz[np.arange(len(yb)), yb] -= 1.0
                dz *= (w / len(yb))[:, None]

                gW = h.T @ dz
                gb = dz.sum(axis=0)
                dh = dz @ self.W.T                      # (B, out_dim)
                dmean = dh[:, :self.dim]
                per_tok = np.repeat(dmean / lens[:, None].astype(np.float32),
                                    lens, axis=0)
                self.W -= cur_lr * gW
                self.b -= cur_lr * gb
                np.add.at(self.E, flat, -cur_lr * per_tok)
                if amax is not None:
                    # max-pool gradient routes only to the argmax n-gram per dim
                    dmax = dh[:, self.dim:]
                    cols = np.tile(np.arange(self.dim), len(lens))
                    np.add.at(self.E, (flat[amax.ravel()], cols),
                              -cur_lr * dmax.ravel())
            rec = {"epoch": ep + 1, "train_loss": tot_loss / seen}
            if val is not None:
                ve, vy = val
                vp = self.predict_encoded(ve)
                vy = np.asarray(vy)
                rec["val_acc"] = float((vp == vy).mean())
                rec["val_macro_f1"] = _macro_f1(vy, vp, self.n_classes)
                if restore_best and rec["val_macro_f1"] > best["score"]:
                    best = {"score": rec["val_macro_f1"], "epoch": ep + 1,
                            "state": (self.E.copy(), self.W.copy(), self.b.copy())}
            hist.append(rec)
            if verbose:
                print("  " + json.dumps(rec))
        if best["state"] is not None:
            # Early stopping on a SHIFTED dev split. Training past this point
            # keeps improving train loss while memorising phrasings that do not
            # transfer, which is visible as a falling dev macro-F1.
            self.E, self.W, self.b = best["state"]
            self.best_epoch = best["epoch"]
            if verbose:
                print(f"  restored epoch {best['epoch']} "
                      f"(dev macro-F1 {best['score']:.4f})")
        return hist

    # ---------- bookkeeping ----------
    def n_params(self):
        return int(self.E.size + self.W.size + self.b.size)

    def size_mb(self, dtype_bytes=4):
        return self.n_params() * dtype_bytes / 1e6

    def save(self, path):
        np.savez_compressed(
            path, E=self.E, W=self.W, b=self.b,
            meta=json.dumps({
                "buckets": self.buckets, "dim": self.dim,
                "n_classes": self.n_classes, "keep_emoji": self.keep_emoji,
                "keep_punct": self.keep_punct, "cmin": self.cmin,
                "cmax": self.cmax, "wmax": self.wmax,
                "pooling": self.pooling}))

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        m = json.loads(str(z["meta"]))
        obj = cls(n_classes=m["n_classes"], buckets=m["buckets"], dim=m["dim"],
                  keep_emoji=m["keep_emoji"], keep_punct=m["keep_punct"],
                  cmin=m["cmin"], cmax=m["cmax"], wmax=m["wmax"],
                  pooling=m.get("pooling", "mean"))
        obj.E, obj.W, obj.b = z["E"], z["W"], z["b"]
        return obj
