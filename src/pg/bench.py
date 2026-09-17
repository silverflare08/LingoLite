"""Latency / throughput measurement.

The requirement is 'single-digit millisecond latency', which is meaningless
without stating the measurement boundary. We report two boundaries:

  end_to_end : raw string in -> label out. Includes normalization, n-gram
               extraction, hashing, pooling, matmul, argmax. This is the
               number that matters for a routing service.
  model_only : pre-encoded ids in -> label out. Isolates the arithmetic from
               the Python-level feature extraction.

Batch size 1 is the real-time routing case; larger batches are the
queue-drain case and give the throughput number.
"""
import platform
import time

import numpy as np


def _percentiles(times_ms):
    a = np.asarray(times_ms)
    return {
        "mean_ms": float(a.mean()),
        "p50_ms": float(np.percentile(a, 50)),
        "p90_ms": float(np.percentile(a, 90)),
        "p99_ms": float(np.percentile(a, 99)),
        "max_ms": float(a.max()),
    }


def hardware():
    import os
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "logical_cores": os.cpu_count(),
        "numpy": np.__version__,
        "device": "CPU (no accelerator)",
        "threads_note": "single process, default BLAS threading",
    }


def latency_single(model, texts, n=1000, warmup=50):
    """batch_size = 1, end-to-end."""
    pool = texts * (n // max(len(texts), 1) + 1)
    for t in pool[:warmup]:
        model.predict([t])
    out = []
    for t in pool[:n]:
        s = time.perf_counter()
        model.predict([t])
        out.append((time.perf_counter() - s) * 1e3)
    return _percentiles(out)


def latency_model_only(model, texts, n=1000, warmup=50):
    enc = model.encode_all(texts)
    is_list = isinstance(enc, list)
    m = len(enc) if is_list else enc.shape[0]
    idx = [i % m for i in range(n + warmup)]

    def one(i):
        model.predict_encoded([enc[i]] if is_list else enc[i:i + 1])

    for i in idx[:warmup]:
        one(i)
    out = []
    for i in idx[warmup:warmup + n]:
        s = time.perf_counter()
        one(i)
        out.append((time.perf_counter() - s) * 1e3)
    return _percentiles(out)


def throughput(model, texts, batch_size=256, rounds=20):
    pool = (texts * (batch_size // max(len(texts), 1) + 1))[:batch_size]
    model.predict(pool)                     # warm
    s = time.perf_counter()
    for _ in range(rounds):
        model.predict(pool)
    el = time.perf_counter() - s
    return {
        "batch_size": batch_size,
        "rounds": rounds,
        "msgs_per_sec": float(batch_size * rounds / el),
        "ms_per_msg": float(el / (batch_size * rounds) * 1e3),
    }


def input_stats(texts):
    lens = np.array([len(t) for t in texts])
    toks = np.array([len(t.split()) for t in texts])
    return {
        "n_messages": int(len(texts)),
        "chars_mean": float(lens.mean()), "chars_p95": float(np.percentile(lens, 95)),
        "tokens_mean": float(toks.mean()), "tokens_p95": float(np.percentile(toks, 95)),
    }
