"""Uni-Sign fine-tuning for isolated sign classification (one GPU; the reported runs used a Tesla T4).

The encoder is unisign_features.UniSignEncoder (pose encoder + mT5 encoder, weights wlasl_pose_only_islr.pth), now
trained. The mT5 token embedding (250k x 768) stays frozen, because only the 8 prefix tokens use it. New head: linear
768 -> classes on the mean of the mT5 encoder output over the pose-token positions, i.e. the `mt5` feature of the frozen
version.

The recipe follows the official ISLR fine-tuning (script/train_stage3.sh, utils.py):
- AdamW lr 3e-4, eps 1e-9, weight decay 1e-4 (no decay for biases and norms, as in timm), cosine schedule without warmup;
- 20 epochs, batch 32 (8 x 4 GPUs), label smoothing 0.2;
- 64 random frames in order in training (datasets.py:472-473), evenly spaced at test (unisign_features.frame_index).
Differences: a linear classification head (not the generative decoder, because the metrics need a score per class),
and fp32 (the T4 has no bf16, and mT5 is unstable in fp16). Metrics come from the last epoch.

--protocol all trains on all 22 signers without test data (the released model; its expected accuracy on new signers is
the C-L result). --drop face|hands|thumbs trains and tests an ablation: no face (51 of 69 points), hands and wrists only
(44), or thumbs masked (61), through explain_unisign.occlude. --state-dir keeps a checkpoint per epoch, so an interrupted
run resumes; --save-dir saves the final weights of each fold. Fold tables are read from qc/ if it exists, else from the
working directory. Weights and the Uni-Sign clone: see unisign_features.py ($UNISIGN_CKPT, $UNISIGN_MT5, $UNISIGN_DIR).

    python scripts/unisign_finetune.py --demo [ckpt]      # needs torch, the checkpoint, and $RTMW_AFFIX
    python scripts/unisign_finetune.py <setting> <rtmw.npz[,rtmw2.npz ...]> <out_dir> --protocol cl --folds 0-10 --seeds 0 [--epochs 20]
"""
import argparse, json, os, random, sys, time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS; set before torch is imported
import numpy as np
import torch

from train_heads import table
from train_main import device_info, metrics, rng, split
from unisign_features import MAX_LEN, MT5, UniSignEncoder, _Embeds, _ds, frame_index
from explain_unisign import GROUPS, occlude

RECIPE = dict(lr=3e-4, eps=1e-9, wd=1e-4, epochs=20, batch=32, smoothing=0.2)
# ablation variants, in training and test: no face (51 points), hands + wrists only (44), thumbs masked (61)
DROP = {"face": GROUPS["face"], "hands": GROUPS["body"] + GROUPS["face"], "thumbs": GROUPS["thumbs"]}


class FineTune(torch.nn.Module):
    def __init__(self, n_cls, ckpt=None, mt5=MT5, device="cpu"):
        super().__init__()
        enc = UniSignEncoder(ckpt, mt5, device)
        self.report, self.n_prefix, self.device = enc.report, enc.n_prefix, enc.device
        self.uni = enc.uni.requires_grad_(True)
        self.uni.mt5_model.encoder.embed_tokens.requires_grad_(False)
        self.head = torch.nn.Linear(768, n_cls).to(device)

    def forward(self, src, tgt):
        src = {k: (v.to(self.device, torch.float32) if k in ("body", "left", "right", "face_all") else v.to(self.device))
               if torch.is_tensor(v) else v for k, v in src.items()}
        try:
            self.uni(src, tgt)                               # Uni_Sign.forward up to just before mT5 (pre-hook)
            raise RuntimeError("mT5 pre-hook was not called")
        except _Embeds as e:
            x, mask = e.args
        h = self.uni.mt5_model.encoder(inputs_embeds=x, attention_mask=mask).last_hidden_state
        w = src["attention_mask"][..., None].float()
        return self.head((h[:, self.n_prefix:] * w).sum(1) / w.sum(1))


def batch(kp, sc, off, idx, train, rnd, post=None):
    """Clips idx -> src, tgt of Uni-Sign's collate_fn. Training: 64 random frames in order; test: evenly spaced. post:
    applied to the load_part_kp output (occlusion, explain_unisign.occlude)."""
    items = []
    for i in idx:
        n = int(off[i + 1] - off[i])
        f = np.array(sorted(rnd.sample(range(n), MAX_LEN))) if train and n > MAX_LEN else frame_index(n)
        p = _ds().load_part_kp(np.asarray(kp[off[i]:off[i + 1]], np.float64)[f, None],
                               np.asarray(sc[off[i]:off[i + 1]], np.float32)[f, None], force_ok=True)
        items.append((str(i), post(p) if post else p, "", "", {}))
    return _ds().collate_fn(argparse.Namespace(rgb_support=False), items)


def optimizer(net):
    decay = [p for p in net.parameters() if p.requires_grad and p.ndim >= 2]
    other = [p for p in net.parameters() if p.requires_grad and p.ndim < 2]
    return torch.optim.AdamW([dict(params=decay, weight_decay=RECIPE["wd"]), dict(params=other, weight_decay=0.0)],
                             lr=RECIPE["lr"], eps=RECIPE["eps"])


def scores(net, data, idx, post=None):
    net.eval()
    out = []
    with torch.no_grad():
        for j in range(0, len(idx), 64):
            out.append(net(*batch(*data, idx[j:j + 64], False, None, post)).float().cpu().numpy())
    return np.concatenate(out)


def train_fold(data, tr, ytr, te, n_cls, seed, device, epochs, ckpt=None, mt5=MT5, state=None, post=None, save=None):
    """tr, te = clip positions in the npz; ytr = labels of tr, aligned. state: checkpoint written every epoch; if it
    exists, training resumes from that epoch (a cloud session can end before one combined fold of about 4.5 h finishes).
    save: final weights (trained parameters + buffers) saved once after the last epoch, without the optimizer
    (about 360 MB)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    net = FineTune(n_cls, ckpt, mt5, device)
    opt = optimizer(net)
    steps = epochs * -(-len(tr) // RECIPE["batch"])
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + np.cos(np.pi * min(s, steps) / steps)))
    lossf = torch.nn.CrossEntropyLoss(label_smoothing=RECIPE["smoothing"])
    g, rnd, yt, losses, t0 = torch.Generator().manual_seed(seed), random.Random(seed), torch.tensor(ytr), [], time.time()
    # the frozen token embedding is not saved; buffers (ST-GCN BatchNorm statistics, updated in training) must be
    keep = {k for k, p in net.named_parameters() if p.requires_grad} | {k for k, _ in net.named_buffers()}
    e0 = 0
    if state and os.path.exists(state):
        s = torch.load(state, map_location="cpu", weights_only=False)
        net.load_state_dict(s["net"], strict=False)
        opt.load_state_dict(s["opt"])
        sch.load_state_dict(s["sch"])
        g.set_state(s["g"]), rnd.setstate(s["rnd"]), random.setstate(s["py"]), np.random.set_state(s["np"])
        torch.set_rng_state(s["cpu"]), torch.cuda.set_rng_state_all(s["cuda"])
        e0, losses = s["epoch"], s["losses"]
        print(f"  resuming from checkpoint epoch {e0}", flush=True)
    for e in range(e0, epochs):
        net.train()
        total = []
        for b in torch.randperm(len(tr), generator=g).split(RECIPE["batch"]):
            b = b.numpy()
            loss = lossf(net(*batch(*data, tr[b], True, rnd, post)), yt[b].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
            total.append(loss.item())
        losses.append(float(np.mean(total)))
        print(f"  epoch {e + 1}/{epochs} loss {losses[-1]:.4f} ({time.time() - t0:.0f} s)", flush=True)
        if state:
            sd = net.state_dict()
            torch.save(dict(net={k: v for k, v in sd.items() if k in keep}, opt=opt.state_dict(), sch=sch.state_dict(), epoch=e + 1,
                            losses=losses, g=g.get_state(), rnd=rnd.getstate(), py=random.getstate(),
                            np=np.random.get_state(), cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()),
                       state + ".tmp")
            os.replace(state + ".tmp", state)
    if save:
        sd = net.state_dict()
        torch.save(dict(net={k: v for k, v in sd.items() if k in keep}, epoch=epochs, losses=losses), save + ".tmp")
        os.replace(save + ".tmp", save)
    train_acc = float((scores(net, data, tr, post).argmax(1) == ytr).mean())
    sc_te = scores(net, data, te, post) if len(te) else np.zeros((0, n_cls), np.float32)   # protocol all: no test data
    return sc_te, train_acc, losses, net.report


def load_rtmw(paths):
    """One or more rtmw_<category>.npz -> concatenated kp, sc, off, relpath (each file read once)."""
    parts = [dict(np.load(p)) for p in paths]
    base = np.cumsum([0] + [len(p["kp"]) for p in parts[:-1]])
    off = np.concatenate([[0]] + [p["off"][1:] + b for p, b in zip(parts, base)])
    return (*(np.concatenate([p[k] for p in parts]) for k in ("kp", "sc")), off,
            np.concatenate([p["relpath"] for p in parts]))


def main(setting, src, out, protocol, folds, seeds, epochs, device, shard, ckpt=None, mt5=MT5, state_dir=None, drop=None,
         save_dir=None):
    name, post = (f"unisign_ft_{drop}", lambda p: occlude(p, DROP[drop])) if drop else ("unisign_ft", None)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    rel, meta, names = table(setting)
    kp, sc, off, rp = load_rtmw(src.split(","))                   # combined: rtmw_<category>.npz, comma-separated
    pos = {r: i for i, r in enumerate(rp)}
    order = np.array([pos[r] for r in rel])                       # fold-CSV row -> clip in the npz (missing -> KeyError)
    data = (kp, sc, off)
    y = meta["label"]
    os.makedirs(out, exist_ok=True)
    for d_ in (state_dir, save_dir):
        if d_:
            os.makedirs(d_, exist_ok=True)                  # create the output folders if needed
    i_sh, n_sh = map(int, shard.split("/"))
    jobs = [(f, s) for f in folds for s in seeds][i_sh::n_sh]
    for fold, seed in jobs:
        path = f"{out}/{setting}_{protocol}_f{fold:02d}_s{seed}_{name}.json"
        if os.path.exists(path):
            continue
        if protocol == "all":           # released model: train on all 22 signers, no test data; expected accuracy = C-L result
            tr, te = np.arange(len(y)), np.arange(0)
        else:
            tr, te = split(meta, protocol, fold, seed, "qc" if os.path.isdir("qc") else ".")
        t0 = time.time()
        tag = f"{setting}_{protocol}_f{fold:02d}_s{seed}{'_' + drop if drop else ''}.pt"
        state, save = (f"{state_dir}/{tag}" if state_dir else None), (f"{save_dir}/{tag}" if save_dir else None)
        sc_te, train_acc, losses, report = train_fold(data, order[tr], y[tr], order[te], len(names), seed, device, epochs,
                                                     ckpt, mt5, state, post, save)   # tr, te = fold-CSV rows; order -> npz positions
        m, pred = (metrics(sc_te, y[te], {k: v[te] for k, v in meta.items()}, names) if len(te)
                   else (dict(acc=float("nan"), macro_f1=float("nan")), np.zeros(0, int)))
        # key `ft_recipe`, not `recipe`: analyze.load drops JSONs whose `recipe` epochs do not match (smoke tests)
        res = dict(setting=setting, protocol=protocol, fold=fold, seed=seed, n_train=int(len(tr)), n_test=int(len(te)),
                   code_commit=open("kode_commit.txt").read().strip() if os.path.exists("kode_commit.txt") else None,   # optional code commit file
                   test_subjects=sorted(map(int, np.unique(meta["subject"][te]))), train_acc=train_acc, epochs=epochs,
                   ft_recipe=dict(RECIPE, epochs=epochs), losses=losses, sec=round(time.time() - t0, 1), **device_info(),
                   ckpt_strict=report["strict"],
                   label=f"unisign_finetune, ablation: {drop}" if drop else "unisign_finetune",
                   models={name: dict(metrics=m, pred=pred.tolist(), top5=np.argsort(-sc_te, 1)[:, :5].tolist())},
                   test_index=te.tolist())
        np.savez_compressed(path[:-5] + "_scores.npz", **{name: sc_te.astype(np.float32)})   # in test_index order
        json.dump(res, open(path, "w"))
        # one write, so the result can be recovered from the log even if the job is killed
        print(f"RESULT_JSON {os.path.basename(path)} {json.dumps(res)}\n", end="", flush=True)
        print(f"{setting} {protocol} fold {fold} seed {seed}: acc {100 * m['acc']:.2f} macro-F1 {100 * m['macro_f1']:.2f} "
              f"train {100 * train_acc:.2f} ({res['sec']:.0f} s)", flush=True)


def demo(ckpt=None):
    """Needs torch, the checkpoint, and $RTMW_AFFIX: two training steps on 8 real affix clips, then gradient and shape
    checks."""
    import tempfile
    assert {k: 69 - len(v) for k, v in DROP.items()} == {"face": 51, "hands": 44, "thumbs": 61}
    with tempfile.TemporaryDirectory() as t:                 # combined: several rtmw_<category>.npz concatenated
        np.savez(f"{t}/a.npz", kp=np.zeros((5, 133, 2)), sc=np.zeros((5, 133)), off=[0, 2, 5], relpath=["a/x/1", "a/x/2"])
        np.savez(f"{t}/b.npz", kp=np.ones((3, 133, 2)), sc=np.ones((3, 133)), off=[0, 3], relpath=["k/y/1"])
        kp, sc, off, rp = load_rtmw([f"{t}/a.npz", f"{t}/b.npz"])
        assert list(off) == [0, 2, 5, 8] and list(rp) == ["a/x/1", "a/x/2", "k/y/1"] and kp[5:].min() == 1
    d = np.load(os.environ.get("RTMW_AFFIX", "data/pilot/rtmw_affix.npz"))
    data = (d["kp"], d["sc"], d["off"])
    seen = []                                                # hands variant: post is called per clip, body 0 and 3-8 zeroed, wrists kept
    batch(*data, np.arange(2), False, None, lambda p: seen.append(occlude(p, DROP["hands"])) or p)
    assert len(seen) == 2 and not seen[0]["body"][:, :7].any() and seen[0]["body"][:, 7:].any()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = FineTune(18, ckpt, MT5, dev)
    assert net.report["strict"], net.report
    frozen = net.uni.mt5_model.encoder.embed_tokens.weight
    opt, lossf = optimizer(net), torch.nn.CrossEntropyLoss(label_smoothing=RECIPE["smoothing"])
    idx, rnd = np.arange(8), random.Random(0)
    yb = torch.arange(8, device=dev) % 18
    before = net.head.weight.detach().clone()
    for _ in range(2):
        loss = lossf(net(*batch(*data, idx, True, rnd)), yb)
        opt.zero_grad()
        loss.backward()
        assert torch.isfinite(loss) and frozen.grad is None
        assert net.uni.pose_proj.weight.grad.abs().sum() > 0 and net.head.weight.grad.abs().sum() > 0
        opt.step()
    assert not torch.equal(before, net.head.weight.detach())
    s = scores(net, data, idx)
    assert s.shape == (8, 18) and np.isfinite(s).all()
    with tempfile.TemporaryDirectory() as t:                 # checkpoint: the second call resumes from epoch 1 without training
        y8 = np.arange(8) % 18
        a = train_fold(data, idx, y8, idx, 18, 0, dev, 1, ckpt, MT5, f"{t}/s.pt")
        b = train_fold(data, idx, y8, idx, 18, 0, dev, 1, ckpt, MT5, f"{t}/s.pt")
        assert np.array_equal(a[0], b[0]) and a[2] == b[2], "weights from the checkpoint differ from the trained weights"
        c = train_fold(data, idx, y8, idx, 18, 0, dev, 1, ckpt, MT5, None, None, f"{t}/w.pt")   # final weights only
        net2 = FineTune(18, ckpt, MT5, dev)
        net2.load_state_dict(torch.load(f"{t}/w.pt", map_location="cpu", weights_only=False)["net"], strict=False)
        assert np.array_equal(scores(net2, data, idx), c[0]), "saved final weights differ from the trained weights"
    n_tr = sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6
    print(f"demo ok: Uni-Sign fine-tuning, {n_tr:.1f} M trained parameters, frozen token embedding, loss {loss.item():.3f}, "
          f"test scores (8, 18); peak GPU memory {torch.cuda.max_memory_allocated() / 1e9 if dev == 'cuda' else 0:.2f} GB")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] == "--demo":
        demo(a[1] if len(a) > 1 else None)
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("setting")
        ap.add_argument("src")
        ap.add_argument("out")
        ap.add_argument("--protocol", required=True)
        ap.add_argument("--folds", nargs="+", default=["0-99"])
        ap.add_argument("--seeds", nargs="+", default=["0"])
        ap.add_argument("--epochs", type=int, default=RECIPE["epochs"])
        ap.add_argument("--device", default="cuda")
        ap.add_argument("--shard", default="0/1")
        ap.add_argument("--ckpt", default=os.environ.get("UNISIGN_CKPT"))
        ap.add_argument("--mt5", default=MT5)
        ap.add_argument("--state-dir")                       # per-epoch checkpoints, for sessions that can end early
        ap.add_argument("--drop", choices=sorted(DROP))      # ablation variant
        ap.add_argument("--save-dir")                        # final weights of each fold
        o = ap.parse_args(a)
        from train_main import PROTOCOLS
        main(o.setting, o.src, o.out, o.protocol, rng(o.folds, PROTOCOLS.get(o.protocol, 1)), rng(o.seeds, 100), o.epochs,
             o.device, o.shard, o.ckpt, o.mt5, o.state_dir, o.drop, o.save_dir)
