"""Word aligner F+, which produced the word clips of the benchmark: semi-supervised forced alignment of every sentence
video to its known sign sequence.

Each sentence video must be cut into its signs, whose order is known from the composition table of the dataset. F+ is a
hybrid NN-HMM aligner trained on three kinds of labels: manual boundaries (strong), isolated clips of single signs, and
its own Viterbi alignments of the remaining sentences (weak), refined over several rounds. The chain of states (3
left-to-right states per sign, optional silence at both ends), the Viterbi decoder, the per-frame features, and the
temporal CNN (receptive field 29 frames) are those of candidate D in align_words.py, which starts from flat labels
instead. F+ works as follows:
1. Strong labels: sentence videos with manual boundaries. Each manual sign span is split evenly into 3 states; frames
   outside [first point, last point) are silence. Strong labels are never realigned, and their loss weight is WS.
2. Round 0: the CNN is trained on the strong labels and the isolated clips (one sign each; even labels over the active
   span are almost always right), not from a flat start. This model realigns the other sentence videos (weak labels).
   A sentence with a state that round 0 never saw (a sign absent from the annotated videos) keeps its flat labels until
   round 1.
3. Later rounds: train on strong + weak + isolated labels, then realign the weak labels by Viterbi in the known sign
   order: ROUNDS rounds for each cross-validation model, FINAL_ROUNDS for the final model.
4. Leave-one-signer-out cross-validation over the signers of the annotated videos: the test signer is removed from
   training entirely (strong, weak, and isolated labels), then its annotated videos are aligned and compared with the
   manual points (mean point error in frames) -> fplus_loso_points.csv, one prediction per annotated video.
5. The final model, trained on all data, cuts every sentence video -> fplus_points.csv, the input of
   `cut_words.py F+`.
The CNN sees only 29 frames, so it cannot memorize the typical boundary positions of each sentence.

Inputs of `train`:
- align_data.npz from `align_words.py pack`: per-frame features of the sentence videos (kind "kalimat") and of isolated
  clips (kind "isolasi"), with the sign sequence of each.
- the manual-boundary CSV, one row per annotation, with the columns
      relpath    sentence video as in align_data.npz, e.g. sentence/neutral/subject-3-sentence-1-neutral-rep-1.mp4
      n_isyarat  number of signs N
      titik      N + 1 space-separated frame indices: start of sign 1, the N - 1 boundaries, end of sign N (exclusive)
      flag       "1" marks an unusable annotation
  Other columns are ignored. The last row of a video wins; rows with flag 1, an unknown video, a wrong number of signs
  or points, a sign shorter than 3 frames, or points outside the video are dropped, and a missing file means no strong
  labels. The benchmark used the manual boundaries of the whole 110-video validation sample (segment_words.sample()).
Outputs (in the align_D.csv format: relpath, n_isyarat, titik): fplus_loso_points.csv, fplus_points.csv, and the
weights fplus.pt.

    python scripts/seg_fplus.py --demo                                       # numpy only
    python scripts/seg_fplus.py train align_data.npz <out> [manual.csv]     # GPU (torch); default manual CSV:
                                                                             #   <SIBI_DATA>/annotations/words_manual.csv
"""
import csv
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from align_words import chain, flat, points, viterbi

DATA = os.environ.get("SIBI_DATA", "data")
MANUAL = f"{DATA}/annotations/words_manual.csv"
WS, ROUNDS, FINAL_ROUNDS, EP0, EP = 3.0, 3, 5, 15, 4


def signer(rel):
    return int(re.search(r"subject-(\d+)-", rel).group(1))


def corrections(path, rel, toks, T):
    """Manual-boundary CSV -> {sequence index: points}. The last row of a video wins; flag 1, videos outside the data, a
    different number of signs, signs < 3 frames, or points outside the video are dropped."""
    idx, last, out = {r: i for i, r in enumerate(rel)}, {}, {}
    if path and os.path.exists(path):
        for r in csv.DictReader(open(path)):
            last[r["relpath"]] = r
    for r in last.values():
        i, p = idx.get(r["relpath"]), list(map(int, r["titik"].split()))
        if r["flag"] != "1" and i is not None and int(r["n_isyarat"]) == len(toks[i].split("|")) == len(p) - 1 \
                and (np.diff(p) >= 3).all() and 0 <= p[0] and p[-1] <= T[i]:
            out[i] = p
    print(f"manual boundaries: {len(out)} used of {len(last)} videos", flush=True)
    return out


def strong(T, p):
    """Manual points -> chain position per frame: each sign split evenly into 3 states, silence outside [start, end)."""
    n, pos = len(p) - 1, np.zeros(T, int)
    for g in range(n):
        e = np.linspace(p[g], p[g + 1], 4).round().astype(int)
        for j in range(3):
            pos[e[j]:e[j + 1]] = 1 + 3 * g + j
    pos[p[-1]:] = 3 * n + 1
    return pos


def write(path, rows):
    """rows: (relpath, n, points) -> CSV in the align_D.csv format."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["relpath", "n_isyarat", "titik"])
        for rel, n, p in rows:
            assert len(p) == n + 1 and all(x < y for x, y in zip(p, p[1:])), (rel, p)
            w.writerow([rel, n, " ".join(map(str, p))])


def train(npz, out, manual=None, device="cuda"):
    from multiprocessing import Pool
    pool = Pool(os.cpu_count())                # Viterbi in numpy, in parallel; created before CUDA starts
    import torch
    from torch import nn
    pad = nn.utils.rnn.pad_sequence
    d = dict(np.load(npz))
    X = [torch.tensor(d["X"][a:b].astype(np.float32)) for a, b in zip(d["off"][:-1], d["off"][1:])]
    T = [len(x) for x in X]
    seqs = [s.split("|") for s in d["toks"]]
    vocab = sorted({t for s in seqs for t in s})
    chains = [np.array(chain([vocab.index(t) for t in s])) for s in seqs]
    flat0 = [chains[i][flat(T[i], len(seqs[i]), int(d["a"][i]), int(d["b"][i]))] for i in range(len(X))]
    iso, sig, S = d["kind"] == "isolasi", np.array([signer(r) for r in d["rel"]]), 1 + 3 * len(vocab)
    cor = corrections(manual, d["rel"], d["toks"], T)
    kuat = {i: chains[i][strong(T[i], p)] for i, p in cor.items()}

    class Net(nn.Module):                      # the CNN of candidate D (align_words.py), receptive field 29 frames
        def __init__(self, F=X[0].shape[1], H=128):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv1d(F, H, 5, padding=2), nn.ReLU(), nn.Conv1d(H, H, 5, padding=4, dilation=2),
                                      nn.ReLU(), nn.Conv1d(H, H, 5, padding=8, dilation=4), nn.ReLU(), nn.Conv1d(H, S, 1))

        def forward(self, x):
            return self.conv(x.transpose(1, 2)).transpose(1, 2)

    def fit(net, opt, idx, lab, n_ep):
        net.train()
        idx = sorted(idx, key=lambda i: T[i])
        batches = [idx[k:k + 32] for k in range(0, len(idx), 32)]
        for ep in range(n_ep):
            tot = 0.0
            for bi in np.random.default_rng(ep).permutation(len(batches)):
                b = batches[bi]
                yb = pad([torch.tensor(lab[i]) for i in b], batch_first=True, padding_value=-1).to(device)
                wb = pad([torch.full((T[i],), WS if i in cor else 1.0) for i in b], batch_first=True).to(device)
                ce = nn.functional.cross_entropy(net(pad([X[i] for i in b], batch_first=True).to(device)).transpose(1, 2),
                                                 yb, ignore_index=-1, reduction="none")
                loss = (ce * wb).sum() / wb.sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item()
            print(f"    epoch {ep + 1}/{n_ep} loss {tot / len(batches):.3f}", flush=True)

    @torch.no_grad()
    def align(net, idx, prior):
        """Viterbi in the known sign order over log p(state | x) - log prior -> chain position per frame."""
        net.eval()
        sc = [torch.log_softmax(net(X[i][None].to(device))[0], -1).cpu().numpy() - prior for i in idx]
        return pool.starmap(viterbi, zip(sc, [chains[i] for i in idx]), chunksize=16)

    def run(tr, rounds):
        """Train F+ on the indices tr; strong labels stay fixed, weak labels are refreshed every round. Returns the model
        and the log prior."""
        torch.manual_seed(0)
        net = Net().to(device)
        opt = torch.optim.AdamW(net.parameters(), 1e-3)
        lab = {i: kuat.get(i, flat0[i]) for i in tr}
        r0 = [i for i in tr if i in cor or iso[i]]
        print(f"  round 0: {sum(i in cor for i in tr)} strong + {sum(iso[tr])} isolated", flush=True)
        fit(net, opt, r0, lab, EP0)
        seen = set(np.concatenate([lab[i] for i in r0]).tolist())
        weak = [i for i in tr if i not in cor and not iso[i]]
        def prior():
            c = np.bincount(np.concatenate([lab[i] for i in tr]), minlength=S) + 1.0
            return np.log(c / c.sum())
        for r in range(rounds):
            go = weak if r else [i for i in weak if seen.issuperset(chains[i].tolist())]
            for i, p in zip(go, align(net, go, prior())):
                lab[i] = chains[i][p]
            print(f"  round {r + 1}/{rounds}: realigned {len(go)} weak sentences", flush=True)
            fit(net, opt, tr, lab, EP)
        return net, prior()

    os.makedirs(out, exist_ok=True)
    rows, err = [], []
    for s in sorted({sig[i] for i in cor}):
        te = sorted(i for i in cor if sig[i] == s)
        print(f"Leave-One-Signer-Out: test signer {s} ({len(te)} videos)", flush=True)
        net, prior = run(np.flatnonzero(sig != s), ROUNDS)
        for i, p in zip(te, align(net, te, prior)):
            pts = points(p, len(seqs[i]))
            rows.append((d["rel"][i], len(seqs[i]), pts))
            err.append(np.abs(np.array(pts) - cor[i]))
        print(f"  point error, signer {s}: {np.concatenate(err[-len(te):]).mean():.1f} frames", flush=True)
    write(f"{out}/fplus_loso_points.csv", rows)
    if err:
        print(f"Leave-One-Signer-Out: mean point error {np.concatenate(err).mean():.1f} frames, "
              f"<= 5 frames {np.mean(np.concatenate(err) <= 5):.1%}", flush=True)
    print("final model", flush=True)
    net, prior = run(np.arange(len(X)), FINAL_ROUNDS)
    kal = np.flatnonzero(d["kind"] == "kalimat")
    write(f"{out}/fplus_points.csv", [(d["rel"][i], len(seqs[i]), points(p, len(seqs[i])))
                                      for i, p in zip(kal, align(net, kal, prior))])
    torch.save(net.state_dict(), f"{out}/fplus.pt")
    print(f"-> {out}/fplus_points.csv ({len(kal)} sentences), fplus_loso_points.csv ({len(rows)} videos)", flush=True)


def demo():
    import tempfile
    # manual-boundary reader: the last row wins; flag 1 / wrong number of signs / unknown video / points past the video
    # are dropped
    rel = ["sentence/neutral/subject-3-sentence-1-neutral-rep-1.mp4", "sentence/neutral/subject-4-sentence-2-neutral-rep-2.mp4",
           "sentence/neutral/subject-5-sentence-3-neutral-rep-3.mp4", "affix/an/subject-3-affix-an-rep-1"]
    toks, T = ["saya|me|makan", "dia|pergi", "a|b", "an"], [60, 50, 40, 30]
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write("relpath,n_isyarat,titik,flag,annotator,detik,waktu,mode\n"
                f"{rel[0]},3,1 2 3 4,0,x,1,t,koreksi\n"                     # overridden by the next row
                f"{rel[0]},3,10 20 35 50,0,x,1,t,koreksi\n"
                f"{rel[1]},2,5 20 40,1,x,1,t,koreksi\n"                     # flag 1
                f"{rel[2]},3,5 10 20 30,0,x,1,t,koreksi\n"                  # wrong number of signs
                f"{rel[2]},2,5 20 45,0,x,1,t,koreksi\n"                     # past the end of the video (T = 40)
                "sentence/neutral/subject-9-x.mp4,1,0 10,0,x,1,t,koreksi\n")
    cor = corrections(f.name, rel, toks, T)
    os.unlink(f.name)
    assert cor == {0: [10, 20, 35, 50]}, cor
    assert corrections(None, rel, toks, T) == {}
    assert [signer(r) for r in rel] == [3, 4, 5, 3]
    # strong labels: the points come back intact, and Viterbi in sign order over perfect scores from those labels finds
    # them again
    pos = strong(60, [10, 20, 35, 50])
    assert points(pos, 3) == [10, 20, 35, 50] and pos[9] == 0 and pos[10] == 1 and pos[49] == 9 and pos[50] == 10
    ch = np.array(chain([0, 2, 1]))
    score = np.full((60, 10), -5.0) + np.random.default_rng(0).normal(0, 0.3, (60, 10))
    score[np.arange(60), ch[pos]] = 0.0
    assert points(viterbi(score, list(ch), dmin=2), 3) == [10, 20, 35, 50]
    # output CSV: align_D.csv format
    with tempfile.TemporaryDirectory() as tmp:
        write(f"{tmp}/x.csv", [(rel[0], 3, [10, 20, 35, 50])])
        r = list(csv.DictReader(open(f"{tmp}/x.csv")))
        assert r == [{"relpath": rel[0], "n_isyarat": "3", "titik": "10 20 35 50"}]
    print("demo ok: manual-boundary reader, strong labels, Viterbi in sign order, CSV format")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] == "--demo":
        demo()
    elif a[0] == "train":
        train(a[1], a[2], a[3] if len(a) > 3 else MANUAL)
