"""OPTIONAL scale-up path: the same architecture in PyTorch, plus a char-CNN.

NOT executed in the reference results — those were produced by the NumPy
implementation in hashfast.py, which has no dependency beyond NumPy and
reproduces on any machine. Use this file when you want GPU training, a larger
embedding table, or the convolutional variant.

Parameter accounting for the 500M budget:
    HashFastTorch(buckets=B, dim=d)  ->  B*d + 2d*C + C
    CharCNN(vocab=V, emb=e, ch=k)    ->  V*e + 3*(e*k*w) + k*3*C + ...
Both stay in the single-digit millions at the defaults below. The budget is a
ceiling, not a target: latency, not accuracy, is the binding constraint here.
"""
try:
    import torch
    import torch.nn as nn
except ImportError:  # keeps the package importable without torch installed
    torch = None
    nn = object


class HashFastTorch(nn.Module if torch else object):
    """EmbeddingBag with mean+max pooling, mirroring pg.hashfast.HashFast."""

    def __init__(self, buckets=100_000, dim=64, n_classes=3):
        super().__init__()
        self.emb = nn.Embedding(buckets, dim, sparse=True)
        nn.init.normal_(self.emb.weight, std=dim ** -0.5)
        self.head = nn.Linear(dim * 2, n_classes)

    def forward(self, idx, offsets):
        # idx: flat LongTensor of hashed n-grams; offsets: doc start positions
        e = self.emb(idx)
        outs = []
        for s, t in zip(offsets[:-1], offsets[1:]):
            blk = e[s:t]
            outs.append(torch.cat([blk.mean(0), blk.max(0).values]))
        return self.head(torch.stack(outs))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class CharCNN(nn.Module if torch else object):
    """Character-CNN alternative. Use if you need position-sensitive patterns
    (negation scope, "nahi ... hai" constructions) that a bag of n-grams
    cannot represent. Costs ~3-5x the latency of HashFast for a few points of
    macro-F1 on longer messages; measure before adopting."""

    def __init__(self, vocab=256, emb=48, ch=128, n_classes=3, widths=(3, 4, 5)):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb, ch, w, padding=w // 2) for w in widths])
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(ch * len(widths), n_classes)

    def forward(self, x):
        h = self.emb(x).transpose(1, 2)
        feats = [torch.relu(c(h)).max(dim=2).values for c in self.convs]
        return self.head(self.drop(torch.cat(feats, dim=1)))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


BUDGET = 500_000_000


def check_budget(model):
    n = model.n_params()
    return {"n_params": n, "budget": BUDGET, "ok": n <= BUDGET,
            "pct": round(n / BUDGET * 100, 4)}
