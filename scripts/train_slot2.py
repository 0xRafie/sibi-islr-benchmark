"""Training harness for the skeleton models in scripts/models/ (signbart, dsta_slr), exploratory.

The reported SignBart and DSTA-SLR results come from slot2.py, which runs the official code of both models at the
pinned commits. This harness uses the same folds, metrics, and JSON format as train_main.py, so all results compare
directly.

Each model is a module `scripts/models/<name>.py` with the same interface:

    NAME, SOURCE (repo + commit), RECIPE (optimizer, lr, schedule, epochs, batch; from the official repo)
    prepare(seq150, how) -> array per clip   model points from pose xy 33 + hand xy 2 x 21 (MediaPipe);
                                             how = "f0" (models.f0, main) or "native" (the repo's own normalization)
    make_batch(xs, train, rng) -> tensor or tuple of tensors   time handling (+ augmentation in training) as in the repo
    build(n_cls) -> nn.Module                forward(*batch) -> logits

    python train_slot2.py --demo
    python train_slot2.py --model signbart --how f0 --setting affix --protocol cl --folds 0-21 --seeds 0-4 \
        --data <dir with mp_*.npz> --folds-dir <dir with folds_*.csv> --out <dir> [--max-hours 10.5] [--shard 0/2]
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse, csv, importlib, json, sys, time

import numpy as np

from train_main import PROTOCOLS, SETTINGS, device_info, metrics, rng, split


def load_raw(setting, data_dir, folds_dir):
    """seq150 sequences cut to their trimming `span`, in the row order of folds_<setting>.csv."""
    rows = list(csv.DictReader(open(f"{folds_dir}/folds_{setting}.csv")))
    packs = {c: dict(np.load(f"{data_dir}/mp_{c}.npz")) for c in SETTINGS[setting]}
    index = {c: {r: i for i, r in enumerate(p["relpath"])} for c, p in packs.items()}
    seqs = []
    for r in rows:
        p = packs[r["category"]]
        i = index[r["category"]][r["relpath"]]
        a, b = p["span"][i]
        seqs.append(np.array(p["seq"][p["off"][i]:p["off"][i + 1]][a:b]))
    meta = {k: np.array([int(r[k]) for r in rows]) for k in ("label", "subject", "cs", "cg", "cl", "cp")}
    meta["category"] = np.array([r["category"] for r in rows])
    names = {int(r["label"]): f"{r['category']}/{r['class']}" for r in rows}
    return seqs, meta, [names[i] for i in range(len(names))]


def optimizer(net, recipe):
    import torch
    o = recipe["optimizer"]
    if o == "sgd":
        opt = torch.optim.SGD(net.parameters(), lr=recipe["lr"], momentum=recipe.get("momentum", 0.9),
                              weight_decay=recipe.get("weight_decay", 0), nesterov=recipe.get("nesterov", False))
    elif o in ("adam", "adamw"):
        cls = torch.optim.Adam if o == "adam" else torch.optim.AdamW
        opt = cls(net.parameters(), lr=recipe["lr"], weight_decay=recipe.get("weight_decay", 0))
    else:
        raise ValueError(o)
    s = recipe.get("schedule", ("none",))
    if recipe.get("reduce_on_plateau"):                        # SignBart main.py:107, other arguments torch defaults
        rp = recipe["reduce_on_plateau"]
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=rp["factor"], patience=rp["patience"])
    elif s[0] == "multistep":
        sch = torch.optim.lr_scheduler.MultiStepLR(opt, s[1], s[2])
    elif s[0] == "step":
        sch = torch.optim.lr_scheduler.StepLR(opt, s[1], s[2])
    elif s[0] == "cosine":
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, recipe["epochs"])
    else:
        sch = None
    return opt, sch


def to(batch, device):
    """make_batch may return one tensor or a tuple of tensors (e.g. keypoints + attention mask)."""
    return tuple(t.to(device) for t in batch) if isinstance(batch, tuple) else (batch.to(device),)


class Paused(Exception):
    """The session time limit was reached; the checkpoint is written and the run resumes in the next session."""


def train(mod, xs, y, tr, te, n_cls, seed, device, epochs=None, ckpt=None, pause=None, every=1800):
    """ckpt (optional): checkpoint file. If it exists, training resumes from it with all random states, so the result is
    bit-identical to an uninterrupted run. pause(e) -> True: write the checkpoint after epoch e, then raise Paused."""
    import torch
    recipe = mod.RECIPE
    epochs = epochs or recipe["epochs"]
    torch.manual_seed(seed)
    r = np.random.default_rng(seed)
    net = mod.build(n_cls).to(device)
    opt, sch = optimizer(net, recipe)
    lossf = torch.nn.CrossEntropyLoss(label_smoothing=recipe.get("label_smoothing", 0.0))
    g = torch.Generator().manual_seed(seed)
    ytr = torch.tensor(y[tr], device=device)
    warm, kp = recipe.get("warmup", 0), recipe.get("keep_prob")
    rop = isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau)
    log, snap, start, t_save = dict(lr=[], loss=[]), None, 0, time.time()   # log is a record only, not used in training
    if ckpt and os.path.exists(ckpt):
        st = torch.load(ckpt, map_location=device, weights_only=False)
        net.load_state_dict(st["net"])
        opt.load_state_dict(st["opt"])
        if sch:
            sch.load_state_dict(st["sch"])
        torch.set_rng_state(st["rng_cpu"])
        if st["rng_cuda"] is not None:
            torch.cuda.set_rng_state_all(st["rng_cuda"])
        r.bit_generator.state, log, snap, start = st["rng_np"], st["log"], st["snap"], st["epoch"]
        g.set_state(st["rng_g"])
        print(f"resuming from epoch {start} ({ckpt})", flush=True)
    for e in range(start, epochs):
        net.train()
        if e < warm:                                           # lr = lr * (e + 1) / warmup, per epoch (DSTA-SLR main.py)
            for grp in opt.param_groups:                       # only correct for multistep/step schedules with
                grp["lr"] = recipe["lr"] * (e + 1) / warm      # milestones after the warmup; cosine needs LambdaLR
        extra = (kp[0] + (kp[1] - kp[0]) * min(e, kp[2]) / kp[2],) if kp else ()   # keep_prob (start, end, epochs)
        log["lr"].append(opt.param_groups[0]["lr"])
        tot, nb = torch.zeros((), device=device), 0
        for b in torch.randperm(len(tr), generator=g).split(recipe["batch"]):
            if recipe.get("drop_last") and len(b) < recipe["batch"]:
                continue                                       # DataLoader(drop_last=True) in the original repo
            batch = to(mod.make_batch([xs[tr[i]] for i in b.tolist()], True, r), device)
            opt.zero_grad()
            loss = lossf(net(*batch, *extra), ytr[b.to(device)])
            loss.backward()
            if recipe.get("clip_grad"):
                torch.nn.utils.clip_grad_norm_(net.parameters(), recipe["clip_grad"])
            opt.step()
            tot, nb = tot + loss.detach(), nb + 1
        log["loss"].append(round(tot.item() / max(nb, 1), 5))
        if rop:
            sch.step(loss.item())                              # loss of the last batch of the epoch (SignBart utils.py:88-89)
            if snap is None and opt.param_groups[0]["lr"] <= sch.eps / (1 - sch.factor):   # floor: lr cannot drop further
                snap, log["floor_epoch"] = {k: v.detach().clone() for k, v in net.state_dict().items()}, e + 1
        elif sch:
            sch.step()
        stop = bool(pause and pause(e))
        if ckpt and e + 1 < epochs and (stop or time.time() - t_save > every):
            torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sch=sch.state_dict() if sch else None,
                            rng_cpu=torch.get_rng_state(), rng_np=r.bit_generator.state, rng_g=g.get_state(),
                            rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                            log=log, snap=snap, epoch=e + 1), ckpt + ".tmp")
            os.replace(ckpt + ".tmp", ckpt)
            t_save = time.time()
            if stop:
                raise Paused(f"paused after epoch {e + 1}/{epochs}")
    net.eval()

    def predict(idx_all):
        with torch.no_grad():
            return np.concatenate([net(*to(mod.make_batch([xs[i] for i in idx], False, r), device)).float().cpu().numpy()
                                   for idx in np.array_split(idx_all, max(1, len(idx_all) // 256))])
    out, pred_tr = predict(te), predict(tr).argmax(1)
    if snap is not None:                                       # lr floor: does the model still change after lr bottoms out?
        final = {k: v.detach().clone() for k, v in net.state_dict().items()}
        log["floor_max_dw"] = max(float((final[k].float() - snap[k].float()).abs().max()) for k in final)
        net.load_state_dict(snap)
        log["floor_train_agree"] = float((predict(tr).argmax(1) == pred_tr).mean())   # without reading test data
        net.load_state_dict(final)
    return out, float((pred_tr == y[tr]).mean()), log


def prepare(mod, seqs, how):
    return [mod.prepare(s, how) for s in seqs]


def run(mod, how, xs, meta, names, protocol, fold, seed, folds_dir, epochs=None, ckpt=None, pause=None, scores_out=None):
    info = device_info()
    tr, te = split(meta, protocol, fold, seed, folds_dir)
    y = meta["label"]
    t0 = time.time()
    scores, train_acc, log = train(mod, xs, y, tr, te, len(names), seed, info["device"], epochs, ckpt, pause)
    meta_te = {k: v[te] for k, v in meta.items()}
    m, pred = metrics(scores, y[te], meta_te, names)
    if scores_out:                                             # full scores per test clip, in test_index order
        np.savez_compressed(scores_out, **{f"{mod.NAME}_{how}": scores.astype(np.float32)})
    return dict(model=mod.NAME, how=how, source=mod.SOURCE, recipe={k: str(v) for k, v in mod.RECIPE.items()},
                protocol=protocol, fold=fold, seed=seed, n_train=int(len(tr)), n_test=int(len(te)),
                test_subjects=sorted(map(int, np.unique(meta["subject"][te]))), train_acc=train_acc,
                sec=round(time.time() - t0, 1), epochs=epochs or mod.RECIPE["epochs"], **info,
                label="train_slot2", test_index=te.tolist(), train_log=log,
                models={f"{mod.NAME}_{how}": dict(metrics=m, pred=pred.tolist(), top5=np.argsort(-scores, 1)[:, :5].tolist())})


def main(argv):
    import torch
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--how", choices=("f0", "native"), default="f0")
    p.add_argument("--setting", choices=SETTINGS, required=True)
    p.add_argument("--protocol", choices=PROTOCOLS, required=True)
    p.add_argument("--folds", nargs="+", default=["0-99"])
    p.add_argument("--seeds", nargs="+", default=["0-4"])
    p.add_argument("--data", required=True)
    p.add_argument("--folds-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-hours", type=float, default=10.5)
    p.add_argument("--shard", default="0/1")
    p.add_argument("--epochs", type=int, help="for smoke tests only; real runs use the recipe epochs")
    p.add_argument("--resume", action="store_true",
                   help="checkpoint <json>.ckpt every 30 minutes and at --max-hours; the run continues in the next session")
    a = p.parse_args(argv)
    mod = importlib.import_module(f"models.{a.model}")
    i, n = map(int, a.shard.split("/"))
    jobs = [(f, s) for s in rng(a.seeds, 5) for f in rng(a.folds, PROTOCOLS[a.protocol])][i::n]
    os.makedirs(a.out, exist_ok=True)
    t_start, xs = time.time(), None
    for fold, seed in jobs:
        path = f"{a.out}/{a.model}_{a.how}_{a.setting}_{a.protocol}_f{fold:02d}_s{seed}.json"
        if os.path.exists(path):
            continue
        if (time.time() - t_start) / 3600 > a.max_hours:
            print(f"stopping: exceeded --max-hours {a.max_hours}", flush=True)
            break
        if xs is None:
            seqs, meta, names = load_raw(a.setting, a.data, a.folds_dir)
            xs = prepare(mod, seqs, a.how)
        ck = path[:-5] + ".ckpt" if a.resume else None
        stop = (lambda e: (time.time() - t_start) / 3600 > a.max_hours) if a.resume else None
        try:
            res = dict(setting=a.setting, n_classes=len(names),
                       **run(mod, a.how, xs, meta, names, a.protocol, fold, seed, a.folds_dir, a.epochs, ck, stop,
                           path[:-5] + "_scores.npz"))
        except Paused as err:
            print(f"{a.model} {a.how} {a.setting} {a.protocol} fold {fold} seed {seed}: {err}", flush=True)
            break
        json.dump(res, open(path + ".tmp", "w"))
        os.replace(path + ".tmp", path)
        if ck and os.path.exists(ck):
            os.remove(ck)
        m = res["models"][f"{a.model}_{a.how}"]["metrics"]
        print(f"{a.model} {a.how} {a.setting} {a.protocol} fold {fold} seed {seed}: acc {100 * m['acc']:.2f}  "
              f"F1 {100 * m['macro_f1']:.2f}  train {100 * res['train_acc']:.1f}  ({res['sec']:.0f} s, {res['gpu'] or 'cpu'})", flush=True)


def demo():
    """Toy model with the same interface, to test the training loop, split, and metrics."""
    import types, torch
    mod = types.SimpleNamespace(
        NAME="toy", SOURCE="demo", RECIPE=dict(optimizer="adam", lr=1e-2, epochs=40, batch=8, schedule=("step", 20, 0.5)),
        prepare=lambda s, how: s[:, 66:].reshape(len(s), 42, 2),
        make_batch=lambda xs, train, r: torch.tensor(np.stack([x.mean(0).ravel() for x in xs]), dtype=torch.float32),
        build=lambda n: torch.nn.Linear(84, n))
    rg = np.random.default_rng(0)
    seqs, label, subj = [], [], []
    for k in range(3):
        for s in (1, 2, 3, 4):
            for _ in range(3):
                x = np.zeros((12, 150), np.float32)
                x[:, 66:] = rg.normal(0.5, 0.01, (12, 84))
                x[:, 66 + 20 * k:86 + 20 * k] += 1.0           # each class shifts its own block of features
                seqs.append(x); label.append(k); subj.append(s)
    meta = dict(label=np.array(label), subject=np.array(subj), cl=np.array(subj) - 1, cs=np.arange(36) % 5,
                cg=np.arange(36) % 5, cp=np.array(subj) - 1, category=np.array(["affix"] * 36))
    res = run(mod, "f0", prepare(mod, seqs, "f0"), meta, ["affix/a", "affix/b", "affix/c"], "cl", 0, 0, ".")
    m = res["models"]["toy_f0"]["metrics"]
    assert res["n_test"] == 9 and res["test_subjects"] == [1] and m["acc"] == 1.0, m
    # pause + resume must be bit-identical to an uninterrupted run (random augmentation and dropout included)
    import tempfile
    mod.make_batch = lambda xs, train, r: torch.tensor(np.stack([x.mean(0).ravel() + (r.normal(0, .1, 84) if train else 0)
                                                                 for x in xs]), dtype=torch.float32)
    mod.build = lambda n: torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(84, n))
    mod.RECIPE = dict(mod.RECIPE, warmup=3, epochs=12)
    xs_ = prepare(mod, seqs, "f0")
    tr_, te_ = np.arange(12, 36), np.arange(12)
    full = train(mod, xs_, np.array(label), tr_, te_, 3, 0, "cpu")
    with tempfile.TemporaryDirectory() as d:
        ck = f"{d}/x.ckpt"
        try:
            train(mod, xs_, np.array(label), tr_, te_, 3, 0, "cpu", ckpt=ck, pause=lambda e: e == 4)
            raise AssertionError("not paused")
        except Paused:
            pass
        again = train(mod, xs_, np.array(label), tr_, te_, 3, 0, "cpu", ckpt=ck, pause=lambda e: False)
    assert np.array_equal(full[0], again[0]) and full[2]["lr"] == again[2]["lr"] and full[2]["loss"] == again[2]["loss"]
    print("demo ok: model interface, recipe training loop, C-L split, metrics and JSON as in train_main, "
          "pause + resume bit-identical")


if __name__ == "__main__":
    demo() if "--demo" in sys.argv else main(sys.argv[1:])
