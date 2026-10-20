"""Word-boundary candidate D: forced alignment of each sentence video to its known sign sequence.

Boundary detectors such as B (sign-language-processing/segmentation) and FAST only guess where a new sign starts, from
what they learned on German Sign Language. D uses what is known for every sentence, its signs and their order (the
composition table of the dataset), and learns from the SIBI videos themselves. It follows the classic hybrid NN-HMM
recipe (flat start, then Viterbi realignment) used for the frame labels of RWTH-PHOENIX:
1. Each sign has 3 left-to-right states; one silence state may open and close the video.
2. Initial labels: the active span (trim.span) is split evenly into N signs x 3 states; frames outside it are silence.
3. A temporal CNN is trained per frame on the current labels (cross-entropy). Its context is local (29 frames) on
   purpose: a BiLSTM over the whole sentence learned the typical boundary positions of each sentence instead of what
   the signs look like (its boundaries varied half as much as those of B and FAST across videos of the same sentence,
   and agreed with B and FAST only at chance level).
4. Realignment: Viterbi over log p(state | x) - log prior along the chain silence? -> sign 1 ... sign N -> silence?, each
   state lasting at least DMIN frames (a sign at least 15 frames, as in the dynamic-programming candidates A, B, C).
   Steps 3-4 are repeated ITERS times.
Training data: the neutral, fear, and surprise sentence videos, plus the isolated clips of the same signers for sentence
tokens that have their own class (ISO; affixes except `di`, which is a preposition in the sentences). Output: per
sentence video, the points [start of sign 1, boundary 1|2, ..., end of sign N], in the format of the manual annotation.

Paths: under SIBI_DATA (default `data`): kata/keypoints/sentence/neutral/ and keypoints/sentence/{fear,surprise}/
(sentence keypoints, extract.py), pilot/mp_<category>.npz (pack_mp.py), segsample/ (B and FAST signals), kata/ (outputs).
extract.py --darmawan writes the neutral keypoints to keypoints/sentence/neutral/, so link kata/keypoints once:
`ln -s ../keypoints data/kata/keypoints`. The benchmark uses the F+ aligner (seg_fplus.py), which trains on the
align_data.npz written by `pack`.

    python scripts/align_words.py --demo                       # numpy only: Viterbi, initial labels, features
    python scripts/align_words.py pack                         # -> data/kata/align_data.npz (numpy)
    python scripts/align_words.py train align_data.npz <out>   # GPU (torch): -> <out>/align_D.csv, <out>/align_D.pt
    python scripts/align_words.py usulan                       # -> data/kata/usulan.csv (median of B, FAST, D; proposed
                                                               #    points for manual correction)
"""
import csv
import glob
import os
import re
import sys

import numpy as np

DATA = os.environ.get("SIBI_DATA", "data")
KATA = f"{DATA}/kata"
DMIN, ITERS = 5, 6
ISO = {"affix": {"an", "ber", "kah", "kan", "ke", "me", "nya", "ter"}, "number": {"1", "2", "3", "12", "ribu"},
       "alphabet": {"a", "g", "i", "j", "n"}}


def tok(t):
    """Composition-table token -> model name: angka:2, huruf:J, or a lowercase word or affix."""
    t = t.strip()
    if t[0].isdigit():
        return f"angka:{int(float(t))}"
    return f"huruf:{t}" if len(t) == 1 and t.isupper() else t.lower()


def feats(s):
    """(T, 150) pose xy 66 + hands xy 84 -> (T, 194): upper body (points 11-22) and hands normalized by the shoulder
    centre and shoulder width (clip medians), hand-present flags, and hand velocity."""
    T = len(s)
    pose, hands = s[:, :66].reshape(T, 33, 2), s[:, 66:].reshape(T, 2, 21, 2)
    sh = pose[:, [11, 12]]
    ok = np.all(sh != 0, axis=(1, 2))
    c = np.median(sh[ok].mean(1), 0) if ok.any() else np.full(2, 0.5)
    w = np.median(np.linalg.norm(sh[ok, 0] - sh[ok, 1], axis=1)) if ok.any() else 0.3
    present = np.any(hands != 0, axis=(2, 3))                                  # (T, 2)
    hn = np.where(present[:, :, None, None], (hands - c) / w, 0).reshape(T, 84)
    vel = np.diff(hn, axis=0, prepend=hn[:1]) * np.repeat(present & np.vstack([present[:1], present[:-1]]), 42, axis=1)
    upper = (pose[:, 11:23] - c) / w
    return np.concatenate([upper.reshape(T, 24), hn, present.astype(np.float32), vel], 1).astype(np.float32)


def chain(ids):
    """Sign ids -> chain of states: 0 = silence, sign g -> 1 + 3g + {0, 1, 2}."""
    return [0] + [1 + 3 * g + s for g in ids for s in range(3)] + [0]


def flat(T, n, a, b):
    """Initial labels: chain position per frame, active span [a, b) split evenly into n signs x 3 states."""
    pos = np.zeros(T, int)
    edges = np.linspace(a, b, 3 * n + 1).round().astype(int)
    for j in range(3 * n):
        pos[edges[j]:edges[j + 1]] = 1 + j
    pos[b:] = 3 * n + 1
    return pos


def viterbi(score, ch, dmin=DMIN):
    """Best path through chain `ch` (state ids) for scores (T, S); every sign state lasts at least dmin frames, and the
    opening and closing silence may be skipped. Returns the chain position per frame (0 opening silence, 1..3n signs,
    3n+1 closing silence)."""
    L = len(ch)
    exp = [(j, k) for j in range(L) for k in range(1 if j in (0, L - 1) else dmin)]   # (chain position, k-th copy)
    E, T = len(exp), len(score)
    em = score[:, [ch[j] for j, _ in exp]]
    loop = np.array([k == (1 if j in (0, L - 1) else dmin) - 1 for j, k in exp])    # only the last copy may stay
    dp, bp = np.full((T, E), -np.inf), np.zeros((T, E), int)
    dp[0, 0], dp[0, 1] = em[0, 0], em[0, 1]                                        # start in silence or in sign 1
    for t in range(1, T):
        stay = np.where(loop, dp[t - 1], -np.inf)
        adv = np.concatenate([[-np.inf], dp[t - 1, :-1]])
        bp[t] = np.where(stay >= adv, np.arange(E), np.arange(E) - 1)
        dp[t] = np.maximum(stay, adv) + em[t]
    end = E - 1 if dp[-1, E - 1] >= dp[-1, E - 2] else E - 2                         # end in silence or in sign N
    path = np.zeros(T, int)
    path[-1] = end
    for t in range(T - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]
    return np.array([exp[e][0] for e in path])


def points(pos, n):
    """Chain position per frame -> [start of sign 1, ..., start of sign n, end of sign n]."""
    first = [int(np.argmax(pos >= 1 + 3 * g)) for g in range(n)]
    end = int(np.argmax(pos == 3 * n + 1)) if (pos == 3 * n + 1).any() else len(pos)
    return first + [end]


def sentences():
    """(relpath, path of the 1662-value keypoints) of the neutral, fear, and surprise sentence videos whose keypoints
    exist."""
    out = [(f"sentence/neutral/{os.path.basename(p)[:-4]}.mp4", p)
           for p in sorted(glob.glob(f"{KATA}/keypoints/sentence/neutral/*.npy"))]
    assert os.path.isdir(f"{DATA}/keypoints/sentence"), f"{DATA}/keypoints/sentence not found: extract the sentence keypoints first (extract.py --darmawan)"
    out += [(os.path.relpath(p, f"{DATA}/keypoints")[:-4] + ".mp4", p)
            for e in ("fear", "surprise") for p in sorted(glob.glob(f"{DATA}/keypoints/sentence/{e}/*.npy"))]
    return out


def pack(out=f"{KATA}/align_data.npz"):
    sys.path.insert(0, "scripts")
    from pack_mp import frame150
    from segment_words import gloss_of
    from trim import span
    X, rows = [], []
    for rel, p in sentences():
        raw = np.load(p).reshape(-1, 1662)
        a, b = span(raw)
        X.append(feats(frame150(raw)))
        rows.append((rel, "|".join(tok(t) for t in gloss_of(rel)), a, b, "kalimat"))
    for cat, keep in ISO.items():
        d = dict(np.load(f"{DATA}/pilot/mp_{cat}.npz"))
        for i, rel in enumerate(map(str, d["relpath"])):
            cls = rel.split("/")[1]
            if cls in keep:
                X.append(feats(d["seq"][d["off"][i]:d["off"][i + 1]]))
                name = {"number": f"angka:{cls}" if cls[0].isdigit() else cls, "alphabet": f"huruf:{cls.upper()}"}.get(cat, cls)
                rows.append((rel, name, *map(int, d["span"][i]), "isolasi"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rel, toks, a, b, kind = zip(*rows)
    np.savez_compressed(out, X=np.concatenate(X).astype(np.float16), off=np.cumsum([0] + [len(x) for x in X]),
                        rel=np.array(rel), toks=np.array(toks), a=np.array(a), b=np.array(b), kind=np.array(kind))
    print(f"{out}: {sum(k == 'kalimat' for k in kind)} sentences, {sum(k == 'isolasi' for k in kind)} isolated clips, "
          f"{sum(len(x) for x in X)} frames, {len({t for s in toks for t in s.split('|')})} tokens")


def train(npz, out, iters=ITERS, epochs=(12, 6), device="cuda"):
    import torch
    from torch import nn
    torch.manual_seed(0)
    d = dict(np.load(npz))                     # NpzFile decompresses again on every d[...]: load once
    X = [torch.tensor(d["X"][a:b].astype(np.float32)) for a, b in zip(d["off"][:-1], d["off"][1:])]
    seqs = [s.split("|") for s in d["toks"]]
    vocab = sorted({t for s in seqs for t in s})
    ids = [[vocab.index(t) for t in s] for s in seqs]
    chains = [chain(v) for v in ids]
    labels = [np.array(chains[i])[flat(len(x), len(ids[i]), int(d["a"][i]), int(d["b"][i]))] for i, x in enumerate(X)]
    S = 1 + 3 * len(vocab)

    class Net(nn.Module):
        def __init__(self, F=X[0].shape[1], H=128):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv1d(F, H, 5, padding=2), nn.ReLU(), nn.Conv1d(H, H, 5, padding=4, dilation=2),
                                      nn.ReLU(), nn.Conv1d(H, H, 5, padding=8, dilation=4), nn.ReLU(), nn.Conv1d(H, S, 1))

        def forward(self, x):                                  # (B, T, F) -> (B, T, S), receptive field 29 frames
            return self.conv(x.transpose(1, 2)).transpose(1, 2)

    net = Net().to(device)
    opt = torch.optim.AdamW(net.parameters(), 1e-3)
    order = np.argsort([len(x) for x in X])
    batches = [order[i:i + 32] for i in range(0, len(order), 32)]

    def fit(n_ep):
        net.train()
        for ep in range(n_ep):
            tot = 0.0
            for bi in np.random.default_rng(ep).permutation(len(batches)):
                b = batches[bi]
                xb = nn.utils.rnn.pad_sequence([X[i] for i in b], batch_first=True).to(device)
                yb = nn.utils.rnn.pad_sequence([torch.tensor(labels[i]) for i in b], batch_first=True, padding_value=-1).to(device)
                loss = nn.functional.cross_entropy(net(xb).reshape(-1, S), yb.reshape(-1), ignore_index=-1)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item()
            print(f"  epoch {ep + 1}/{n_ep} loss {tot / len(batches):.3f}", flush=True)

    @torch.no_grad()
    def realign():
        """Viterbi per sequence over log p(state | x) - log prior; log-probabilities are not kept (saves memory)."""
        net.eval()
        c = np.bincount(np.concatenate(labels), minlength=S) + 1.0
        prior = np.log(c / c.sum())
        return [viterbi(torch.log_softmax(net(x[None].to(device))[0], -1).cpu().numpy() - prior, chains[i])
                for i, x in enumerate(X)]

    for it in range(iters):
        fit(epochs[0] if it == 0 else epochs[1])
        pos = realign()
        new = [np.array(chains[i])[p] for i, p in enumerate(pos)]
        changed = np.mean(np.concatenate([n != o for n, o in zip(new, labels)]))
        labels = new
        print(f"iteration {it + 1}/{iters}: {changed:.1%} of frame labels changed", flush=True)
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/align_D.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["relpath", "n_isyarat", "titik"])
        for i in np.flatnonzero(d["kind"] == "kalimat"):
            w.writerow([d["rel"][i], len(ids[i]), " ".join(map(str, points(pos[i], len(ids[i]))))])
    torch.save(net.state_dict(), f"{out}/align_D.pt")
    print(f"-> {out}/align_D.csv ({int((d['kind'] == 'kalimat').sum())} sentences), vocabulary {len(vocab)}, states {S}")


def usulan(root=f"{DATA}/segsample", d_csv=f"{KATA}/align_D.csv", out=f"{KATA}/usulan.csv"):
    """Proposed points for manual correction of the 110 sample videos: the per-point median of B, FAST (dynamic
    programming of segment_words) and D, so no single candidate decides alone."""
    sys.path.insert(0, "scripts")
    from segment_words import boundaries, gloss_of, sample
    from trim import span
    dpts = {r["relpath"]: list(map(int, r["titik"].split())) for r in csv.DictReader(open(d_csv))}
    with open(out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["relpath", "titik"])
        for rel, _ in sample():
            n, (a, b) = len(gloss_of(rel)), span(np.load(f"{root}/keypoints/{rel[:-4]}.npy"))
            cand = [dpts[rel]] + [[a] + bd + [b] for d in ("segprobs", "fastprobs")
                                  if (bd := boundaries(-np.load(f"{root}/{d}/{rel[:-4]}.npy"), n, a, b))]
            w.writerow([rel, " ".join(map(str, np.median(np.array(cand), axis=0).round().astype(int)))])
    print("->", out)


def demo():
    rng = np.random.default_rng(0)
    # chain of signs [0, 2, 1]; truth: silence 0-9, signs 10-19, 20-34, 35-49, silence 50-59
    ch, T = chain([0, 2, 1]), 60
    truth = np.zeros(T, int)
    for j, (s0, s1) in enumerate([(10, 13), (13, 17), (17, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, 45), (45, 50)]):
        truth[s0:s1] = 1 + j
    truth[50:] = 10
    score = np.full((T, 10), -5.0) + rng.normal(0, 0.3, (T, 10))
    score[np.arange(T), np.array(ch)[truth]] = 0.0
    pos = viterbi(score, ch, dmin=2)
    assert points(pos, 3) == [10, 20, 35, 50], points(pos, 3)
    assert (np.diff(pos) >= 0).all()                                              # left to right
    f = flat(30, 2, 5, 25)
    assert f[4] == 0 and f[5] == 1 and f[24] == 6 and f[25] == 7 and points(f, 2) == [5, 15, 25]
    s = np.zeros((4, 150), np.float32)
    s[:, 22:24], s[:, 24:26] = (0.4, 0.5), (0.6, 0.5)                              # left and right shoulder (points 11, 12)
    s[1:, 66:108] = 0.5                                                           # left hand appears in frame 1
    x = feats(s)
    assert x.shape == (4, 194) and x[0, 108] == 0 and x[1, 108] == 1 and np.allclose(x[2, 110:194][:42], 0)
    assert tok("2.0") == "angka:2" and tok("J") == "huruf:J" and tok("me ") == "me"
    print("demo ok: left-to-right Viterbi with minimum duration, flat initial labels, body + hand features, token names")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] == "--demo":
        demo()
    elif a[0] == "pack":
        pack()
    elif a[0] == "train":
        train(a[1], a[2])
    elif a[0] == "usulan":
        usulan()
