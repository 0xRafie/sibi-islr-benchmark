"""SHuBERT + rank-1 LoRA: fine-tuning per leave-one-signer-out fold, combined setting (114 classes).

The authors' LoRA code is not in their repository or Space, so the recipe follows the paper, §4.3 (Gueuwou et al., ACL
2025):
- rank-1 LoRA on every nn.Linear of SHuBERT (channel projections, post_extract_proj, q/k/v/out, fc1, fc2), all other
  weights frozen, scale 1 (alpha is not given). It is registered as a parametrization of `weight`: fairseq's
  MultiheadAttention reads q_proj.weight etc. directly (F.multi_head_attention_forward), so a forward wrapper would be
  bypassed. 169,870 parameters = "0.17M" in Table 2 of the paper. The pretraining heads are removed first
  (remove_pretraining_modules).
- head: time average of the final encoder output over the valid frames + BatchNorm1d + Linear(114).
- Adam, lr 1e-4 for the head and 1e-5 for LoRA, weight decay 1e-4, label smoothing 0.1, batch 128 (drop_last, for
  BatchNorm), at most 125 epochs, early stopping on validation accuracy with patience 10 (the paper gives no patience).
- validation: the signer of fold (k + 1) mod 22, removed from the training data (20 training signers); the test signer is
  never used for selection.
- test accuracy and macro-F1 are logged every epoch for reference only.
Input = the per-frame features of shubert_features.py stage a (frames_<category>.npz), as for the frozen-encoder run.
Encoder configuration = the default SHubertConfig(), as in the authors' load_model (dropout 0.1, attention dropout 0.1,
layerdrop 0). Environment: Python 3.10 + SHuBERT's bundled fairseq (environment b of shubert_features.py). Fold tables
are read from qc/ if it exists, else from the working directory; the encoder checkpoint from $SHUBERT_CKPT, else the
pinned Hugging Face download.

    python scripts/shubert_lora.py --demo                     # numpy part; in environment b also checks LoRA and padding
    python scripts/shubert_lora.py run <frames dir> <out dir> <fold ...> [epochs=E] [cpu]
"""
import json, math, os, sys, time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np

from train_heads import table
from train_main import metrics, split

QC = "qc" if os.path.isdir("qc") else "."   # fold CSVs: qc/, or the working directory when the scripts sit in one flat folder
CATS = ("affix", "number", "alphabet", "kata")
R = dict(rank=1, scale=1.0, lr_head=1e-4, lr_lora=1e-5, wd=1e-4, smooth=0.1, batch=128, epochs=125, patience=10)
N_LORA = 3 * (384 + 256) + (14 + 256) + (1024 + 768) + 12 * (4 * 2 * 768 + 2 * (768 + 3072))


def load(frames_dir, setting="combined"):
    """Clips in the row order of folds_<setting>.csv: a (face, lhand, rhand, pose) tuple per clip (slices, no copies)."""
    rel, meta, names = table(setting, QC)
    clips = {}
    for c in CATS:
        d = np.load(f"{frames_dir}/frames_{c}.npz")
        arr, off = [d[k] for k in ("face", "lhand", "rhand", "pose")], d["off"]
        for i, r in enumerate(d["relpath"]):
            clips[str(r)] = tuple(a[off[i]:off[i + 1]] for a in arr)
    return [clips[r] for r in rel], meta, names                  # a missing clip raises KeyError, not silence


def folds3(meta, fold):
    """Train, validation, and test indices of leave-one-signer-out fold `fold`: validation = the signer of fold
    (fold + 1) mod 22."""
    tr, te = split(meta, "cl", fold, 0, QC)
    v = meta["cl"][tr] == (fold + 1) % 22
    return tr[~v], tr[v], te


def pad(batch):
    """[(face, lhand, rhand, pose)] -> four float32 arrays (B, T, d), zero-padded at the end, + padding mask (True =
    pad)."""
    T = max(len(b[0]) for b in batch)
    out = [np.zeros((len(batch), T, b.shape[1]), np.float32) for b in batch[0]]
    for i, b in enumerate(batch):
        for o, a in zip(out, b):
            o[i, :len(a)] = a
    return out, np.arange(T)[None] >= np.array([len(b[0]) for b in batch])[:, None]


def add_lora(model):
    """LoRA of rank R['rank'] on every nn.Linear; all other weights frozen. -> LoRA parameters."""
    import torch
    from torch import nn
    from torch.nn.utils import parametrize

    class LoRA(nn.Module):
        def __init__(self, out_f, in_f):
            super().__init__()
            self.A = nn.Parameter(torch.empty(R["rank"], in_f))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))      # as in PEFT; B = 0, so training starts from the original encoder
            self.B = nn.Parameter(torch.zeros(out_f, R["rank"]))

        def forward(self, W):
            return W + R["scale"] * (self.B @ self.A)

    model.requires_grad_(False)
    lins = [m for m in model.modules() if isinstance(m, nn.Linear)]
    for m in lins:
        parametrize.register_parametrization(m, "weight", LoRA(*m.weight.shape).to(m.weight.device))
    return [p for m in lins for p in m.parametrizations.weight[0].parameters()]


def logits(model, head, batch, device):
    """Final encoder output (B, T, 768), averaged over the valid frames, then the head."""
    import torch
    (f, lh, rh, po), pm = pad(batch)
    t = lambda a: torch.from_numpy(a).to(device)
    f, lh, rh, po, pm = t(f), t(lh), t(rh), t(po), t(pm)
    z = torch.zeros((pm.shape[1], 1), device=device)
    src = [dict(face=f[i], left_hand=lh[i], right_hand=rh[i], body_posture=po[i], label_face=z, label_left_hand=z,
                label_right_hand=z, label_body_posture=z) for i in range(len(batch))]
    x = model.extract_features(src, padding_mask=pm, kmeans_labels=None, mask=False)["x"]
    keep = (~pm).unsqueeze(-1).float()
    return head((x * keep).sum(1) / keep.sum(1))


def train_fold(X, meta, names, fold, device, epochs=R["epochs"], seed=0):
    import random
    import torch
    from sklearn.metrics import f1_score
    import shubert_features as S
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    torch.use_deterministic_algorithms(True, warn_only=True)
    tr, va, te = folds3(meta, fold)
    y, t0 = meta["label"], time.time()
    model, rep = S.encoder(os.environ.get("SHUBERT_CKPT"), device=device)   # $SHUBERT_CKPT, else the Hugging Face download
    model.remove_pretraining_modules()
    lora = add_lora(model)
    assert sum(p.numel() for p in lora) == N_LORA, sum(p.numel() for p in lora)
    head = torch.nn.Sequential(torch.nn.BatchNorm1d(768), torch.nn.Linear(768, len(names))).to(device)
    opt = torch.optim.Adam([dict(params=head.parameters(), lr=R["lr_head"]), dict(params=lora, lr=R["lr_lora"])],
                           weight_decay=R["wd"])
    lossf = torch.nn.CrossEntropyLoss(label_smoothing=R["smooth"])
    g = torch.Generator().manual_seed(seed)

    def predict(idx):
        model.eval(), head.eval()
        with torch.no_grad():
            return np.concatenate([logits(model, head, [X[j] for j in idx[i:i + R["batch"]]], device).float().cpu().numpy()
                                   for i in range(0, len(idx), R["batch"])])

    best, best_ep, bad, keep, curve = -1.0, 0, 0, None, []
    for e in range(epochs):
        t = time.time()
        model.train(), head.train()
        perm, tot = tr[torch.randperm(len(tr), generator=g).numpy()], []
        for i in range(0, len(perm) - R["batch"] + 1, R["batch"]):          # drop_last: BatchNorm needs > 1 clip
            b = perm[i:i + R["batch"]]
            loss = lossf(logits(model, head, [X[j] for j in b], device), torch.from_numpy(y[b]).to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot.append(loss.detach())
        pv, pt = predict(va).argmax(1), predict(te).argmax(1)
        acc_va = float((pv == y[va]).mean())
        curve.append(dict(epoch=e + 1, loss=round(float(torch.stack(tot).mean()), 5), val_acc=acc_va,
                          test_acc=float((pt == y[te]).mean()), test_f1=float(f1_score(y[te], pt, average="macro")),
                          sec=round(time.time() - t, 1)))
        print(f"shubert_lora fold {fold} epoch {e + 1}: " + " ".join(f"{k} {v}" for k, v in curve[-1].items() if k != "epoch"),
              flush=True)
        if acc_va > best:                                                  # the earliest epoch on ties
            best, best_ep, bad = acc_va, e + 1, 0
            keep = [p.detach().clone() for p in lora], {k: v.clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= R["patience"]:
                break
    with torch.no_grad():                                                  # weights of the best validation epoch
        for p, q in zip(lora, keep[0]):
            p.copy_(q)
    head.load_state_dict(keep[1])
    scores = predict(te)
    m, pred = metrics(scores, y[te], {k: v[te] for k, v in meta.items()}, names)
    from train_main import versions
    res = dict(setting="combined", n_classes=len(names), protocol="cl", fold=fold, seed=seed, n_train=len(tr), n_val=len(va),
               n_test=len(te), val_subjects=sorted(map(int, np.unique(meta["subject"][va]))),
               test_subjects=sorted(map(int, np.unique(meta["subject"][te]))), label="shubert_lora",
               code_commit=open("kode_commit.txt").read().strip() if os.path.exists("kode_commit.txt") else None,
               models={"shubert_lora": dict(metrics=m, pred=pred.tolist(), train_acc=float((predict(tr).argmax(1) == y[tr]).mean()))},
               test_index=te.tolist(), best_epoch=best_ep, epochs_run=len(curve), curve=curve, recipe=R, lora_params=N_LORA,
               encoder={k: v for k, v in rep.items() if k != "ckpt"}, device=device, torch=torch.__version__,
               gpu=torch.cuda.get_device_name(0) if device == "cuda" else "", sec=round(time.time() - t0, 1), **versions())
    return res, scores


def main(frames_dir, out, folds, epochs=R["epochs"], device="cuda"):
    X, meta, names = load(frames_dir)
    os.makedirs(out, exist_ok=True)
    for k in folds:
        path = f"{out}/combined_cl_f{k:02d}_s0.json"
        if os.path.exists(path):
            continue
        res, scores = train_fold(X, meta, names, k, device, epochs)
        np.savez_compressed(path[:-5] + "_scores.npz", shubert_lora=scores.astype(np.float32))   # in test_index order
        json.dump(res, open(path + ".tmp", "w"))
        os.replace(path + ".tmp", path)
        m = res["models"]["shubert_lora"]["metrics"]
        print(f"fold {k}: acc {100 * m['acc']:.2f} macro-F1 {100 * m['macro_f1']:.2f} best epoch {res['best_epoch']}"
              f"/{res['epochs_run']} ({res['sec']:.0f} s)", flush=True)
        # one write, so the result can be recovered from the log even if the job is killed
        print(f"RESULT_JSON {os.path.basename(path)} {json.dumps(res)}\n", end="", flush=True)


def demo():
    import tempfile
    rng = np.random.default_rng(0)
    batch = [tuple(rng.random((n, d), np.float32) for d in (384, 384, 384, 14)) for n in (3, 5)]
    (f, *_), pm = pad(batch)
    assert f.shape == (2, 5, 384) and pm.tolist() == [[0, 0, 0, 1, 1], [0, 0, 0, 0, 0]]
    assert np.array_equal(f[0, :3], batch[0][0]) and not f[0, 3:].any()
    assert N_LORA == 169870                                                 # "0.17M" in Table 2 of the paper
    meta = {"label": np.arange(66) % 3, "subject": np.repeat(np.arange(1, 23), 3), "cl": np.repeat(np.arange(22), 3)}
    for k in (0, 21):
        tr, va, te = folds3(meta, k)
        assert set(meta["cl"][te]) == {k} and set(meta["cl"][va]) == {(k + 1) % 22} and len(tr) == 60
        assert not set(tr) & set(va) and not set(tr) & set(te)
    with tempfile.TemporaryDirectory() as d:                               # CSV order, not npz order
        rows = ["relpath,category,class,label,subject,rep,cs,cg,cl,cp"] + [
            f"{c}/k/s{i},{c},k,0,{i + 1},1,0,0,{i},0" for c in CATS for i in range(2)]
        os.makedirs(f"{d}/qc")
        open(f"{d}/qc/folds_combined.csv", "w").write("\n".join(rows) + "\n")
        for c in CATS:
            np.savez(f"{d}/frames_{c}.npz", face=np.arange(5 * 384).reshape(5, 384).astype(np.float16), lhand=np.zeros((5, 384)),
                     rhand=np.zeros((5, 384)), pose=np.zeros((5, 14)), off=[0, 2, 5], relpath=[f"{c}/k/s1", f"{c}/k/s0"])
        global QC
        QC, old = f"{d}/qc", QC
        try:
            X, m, _ = load(d)
        finally:
            QC = old
        assert len(X) == 8 and len(X[0][0]) == 3 and len(X[1][0]) == 2 and X[1][0][0, 0] == 0   # s0 = frames 2-4, s1 = 0-1
    print("demo ok: padding + mask, 169,870 LoRA parameters, validation = signer of the next fold, clip order follows the CSV")
    try:
        import torch
        import shubert_features as S
        sys.path.insert(0, str(S.GITHUB / "fairseq"))
        import fairseq  # noqa: F401
    except Exception:                                                       # noqa: BLE001 (no torch/fairseq)
        return
    model, _ = S.encoder(os.environ.get("SHUBERT_CKPT"))
    model.remove_pretraining_modules()
    head = torch.nn.Sequential(torch.nn.BatchNorm1d(768), torch.nn.Linear(768, 7)).eval()
    with torch.no_grad():
        ref = logits(model, head, batch, "cpu")
        one = torch.cat([logits(model, head, [b], "cpu") for b in batch])
    lora = add_lora(model)
    assert sum(p.numel() for p in lora) == N_LORA and all(not p.requires_grad for n, p in model.named_parameters()
                                                         if "parametrizations" not in n or n.endswith("original"))
    with torch.no_grad():
        same = logits(model, head, batch, "cpu")
    d0 = float((same - ref).abs().max())                                   # the parametrized weight is a new tensor:
    assert d0 < 1e-5, f"LoRA with B = 0 must equal the original encoder ({d0})"   # BLAS may differ in the last bit
    assert torch.allclose(ref, one, atol=1e-4), (ref - one).abs().max()     # padding + mask = one clip at a time
    model.train(), head.train()
    logits(model, head, batch, "cpu").logsumexp(1).sum().backward()
    gb = [m.parametrizations.weight[0].B.grad for m in model.modules() if hasattr(m, "parametrizations")]
    assert len(gb) == 77 and all(g is not None and g.abs().sum() > 0 for g in gb), "the gradient does not reach every LoRA"
    print(f"demo LoRA ok: 77 linear layers (including q/k/v/out through F.multi_head_attention_forward), B = 0 equal "
          f"(max {d0:.1e}), padded batch = one clip at a time (max {float((ref - one).abs().max()):.1e})")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] == "--demo":
        demo()
    elif a[0] == "run":
        opts = [x for x in a[4:] if not x.isdigit()]
        ep = int(next((o[7:] for o in opts if o.startswith("epochs=")), R["epochs"]))
        main(a[1], a[2], [int(x) for x in [a[3], *a[4:]] if x.isdigit()], ep, "cpu" if "cpu" in opts else "cuda")
    else:
        sys.exit(__doc__)
