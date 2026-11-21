"""Fig. 5 (Uni-Sign ablation variants), the occlusion/Shapley figure and Fig. 3 (Uni-Sign framework)
(moved from figures.py)."""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sibi_islr.figures.common import AXIS, COL, DATA, DICT, FULL, GRID, INK, INK2, MUTED, QC, SERIES, TEST, save
from sibi_islr.figures.keypoints import ABLATIONS, INPUT_SIGN, INPUT_T, KAMUS_SIGN, KAMUS_T, RT_HAND, UNISIGN_PTS, U_FACE, draw_unisign, frame_axis, parts


def fig_ablation_unisign():
    d = np.load(f"{DATA}/pilot/rtmw_{INPUT_SIGN.split('/')[0]}.npz")
    i = list(d["relpath"]).index(INPUT_SIGN)
    kp = d["kp"][d["off"][i] + INPUT_T].astype(float)
    sc = d["sc"][d["off"][i] + INPUT_T].astype(float)
    assert (sc[UNISIGN_PTS] > 0.3).all(), "all 69 keypoints must be detected in the example frame"
    body = kp[[5, 6]]
    cx, cy = body.mean(0)
    half = 1.15 * np.linalg.norm(body[0] - body[1])
    fig, axs = plt.subplots(1, len(ABLATIONS), figsize=(FULL, 1.95), gridspec_kw=dict(left=0.01, right=0.99, top=0.9,
                                                                                     bottom=0.1, wspace=0.04))
    for ax, (name, drop) in zip(axs, ABLATIONS):
        n = draw_unisign(ax, kp, drop)
        ax.set_title(f"{name} ({n} points)", fontsize=7, pad=3)
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy + 0.8 * half, cy - 0.75 * half)                          # without the hips and the top of the head
        frame_axis(ax)
    fig.legend(handles=[plt.Line2D([], [], color=SERIES[0], marker="o", ms=3, lw=0.8, label="right hand"),
                        plt.Line2D([], [], color=SERIES[1], marker="s", ms=3, lw=0.8, label="left hand"),
                        plt.Line2D([], [], color=INK, marker="o", ms=3, lw=0.8, label="body and face"),
                        plt.Line2D([], [], color=AXIS, marker="o", ms=3.6, mfc="#ffffff", mec=INK2, mew=0.5, lw=0.5,
                                   label="removed in this variant")],
               loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=1.6, fontsize=6.5)
    save(fig, "fig_ablation_unisign")


# ---------- Test-time occlusion of the full Uni-Sign on one signer (explain_unisign_run.py) ----------

E22 = f"{QC}/results/e22_report.json"
GROUP_LABEL = {"right hand, no thumb": "Right hand (no thumb)", "thumbs": "Thumbs", "body": "Body", "wrists": "Wrists",
               "face": "Face", "left hand, no thumb": "Left hand (no thumb)"}       # order = Shapley value, Overall macro-F1
E22_CATS = [("affix", "Affix", SERIES[0], "o"), ("number", "Number", SERIES[1], "s"), ("alphabet", "Alphabet", SERIES[2], "^"),
            ("kata", "Word", MUTED, "D"), ("ti+wati", "-ti/-wati", INK, "v")]


def fig_shapley(path=E22):
    """(a) Shapley values of the six groups on Overall 114-class macro-F1; (b) per-clip Shapley values on the
    log-probability of the true class, per category, with 95% bootstrap CI over clips. Values per keypoint are not
    drawn: the anchor keypoints (53, 91, 112) are already (0, 0) after loading, so occluding them only zeroes the score."""
    rep = json.load(open(path))
    f1, clip = rep["agregat"]["macro_f1"], rep["per_klip"]
    groups = list(GROUP_LABEL)
    y = np.arange(len(groups))[::-1]
    fig, (a, b) = plt.subplots(2, 1, figsize=(COL, 3.9), gridspec_kw=dict(height_ratios=[1, 1.9], hspace=0.6))
    v = [f1["shapley"][g] for g in groups]
    a.barh(y, v, color=TEST, height=0.62)
    for yi, vi in zip(y, v):
        a.text(vi + 0.6, yi, f"{vi:.1f}", va="center", fontsize=6.5, color=INK2)
    a.set_xlim(0, 35)
    a.set_xlabel("Contribution to 114-class macro-F1 (points)")
    a.set_title(f"(a) All {clip['overall']['n']:,} clips: {f1['full']:.1f} with all keypoints, {f1['none']:.1f} with none",
                loc="left")
    off = np.linspace(0.3, -0.3, len(E22_CATS))
    for k, (c, label, col, mk) in enumerate(E22_CATS):
        m = np.array([clip[c]["shapley"][g] for g in groups])
        b.errorbar(m[:, 0], y + off[k], xerr=[m[:, 0] - m[:, 1], m[:, 2] - m[:, 0]], fmt=mk, ms=3, color=col, lw=0.7,
                   capsize=0, label=f"{label} ({clip[c]['n']})")
    b.axvline(0, color=AXIS, lw=0.6)
    b.set_xlabel("Contribution to log-probability of the correct class, per clip")
    b.set_title("(b) By category (clips), 95% bootstrap CI over clips", loc="left")
    b.legend(loc="upper center", bbox_to_anchor=(0.42, -0.2), ncol=3, frameon=False, handletextpad=0.3, columnspacing=1.0)
    for ax in (a, b):
        ax.set_yticks(y, [GROUP_LABEL[g] for g in groups])
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    save(fig, "fig_shapley")


# ---------- Fig. 3: Uni-Sign as fine-tuned here ----------

def fig_unisign():
    """Uni-Sign as fine-tuned here. Style after Fig. 5 of the Uni-Sign paper (flat pastel blocks without borders, a
    skeleton icon per body part, wide bars for shared stages, tensors as small boxes in dashed boxes, italic process
    labels), with numbered panels per stage under sans-serif titles, layers as 3D slabs, arrow colors per stage, a dashed
    training box and a legend below. Content follows ext/Uni-Sign/models.py:72-100, 206-295 and scripts/unisign_finetune.py:
    the hands and the face receive the wrist and nose features of the body branch without gradient, part embeddings are
    added, and both hands share weights; the RGB branch and the text decoder are not drawn."""
    from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle as R
    W, H, Y0 = 181, 88, -4.5                                        # Y0 < 0: room for a two-row legend
    HT = H - Y0
    SANS = "Arial"
    BLUE, GREEN, GREY = "#256abf", "#1baf7a", "#52514e"
    HAND = "#e4ddf4"                                                # both hands one color: shared weights
    fig = plt.figure(figsize=(FULL, FULL * HT / W))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(Y0, H); ax.axis("off")
    inset = lambda x, y, w, h: fig.add_axes([x / W, (y - Y0) / HT, w / W, h / HT])

    def rbox(x0, y0, x1, y1, fill="none", edge="none", lw=0.7, ls="-", r=1.2, z=1):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}", fc=fill, ec=edge,
                                    lw=lw, ls=ls, zorder=z))

    def panel(x0, y0, x1, y1, num, title, col, fill, fs=7.2, sub=None):   # stage number in a circle
        rbox(x0, y0, x1, y1, fill, col, 1.0)
        if sub:
            ax.text((x0 + x1) / 2, y1 - 6.2, sub, ha="center", va="center", fontsize=6, color=col, family=SANS)
        t = ax.text((x0 + x1) / 2 + 1.6, y1 - 3.3, title, ha="center", va="center", fontsize=fs, fontweight="bold",
                    color=col, family=SANS)
        fig.canvas.draw()
        bb = t.get_window_extent().transformed(ax.transData.inverted())
        ax.add_patch(Circle((bb.x0 - 2.0, y1 - 3.3), 1.45, fc=col, ec="none", zorder=3))
        ax.text(bb.x0 - 2.0, y1 - 3.35, str(num), ha="center", va="center", fontsize=6, fontweight="bold", color="#ffffff",
                family=SANS, zorder=4)

    def block(xc, yc, w, h, text, fill, fs=6.2):                    # flat pastel block without border (Uni-Sign style)
        ax.add_patch(R((xc - w / 2, yc - h / 2), w, h, fc=fill, ec="none", zorder=2))
        ax.text(xc, yc, text, ha="center", va="center", fontsize=fs, color=INK, zorder=3)

    def arrow(x0, y0, x1, y1, col=GREY, label=None, dx=0.8):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=col, lw=0.8, shrinkA=0, shrinkB=0,
                                                            mutation_scale=7), zorder=4)
        if label:
            ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2, label, fontsize=6, color=col, va="center")

    def tokens(x, y, n, fill, edge, vertical=False, group=True, hatch=None):   # tensor: small boxes in a dashed box
        step = 2.9
        for k in range(n):
            ax.add_patch(R((x, y + step * k) if vertical else (x + step * k, y), 2.4, 2.4, fc=fill, ec=edge, lw=0.5,
                           hatch=hatch, zorder=3))
        if group:
            w, h = (2.4, step * (n - 1) + 2.4) if vertical else (step * (n - 1) + 2.4, 2.4)
            rbox(x - 0.9, y - 0.9, x + w + 0.9, y + h + 0.9, edge="#8fb3e3", lw=0.5, ls="--", r=0.9, z=2)

    def slabs(x0, y0, n, w, h, dx, dy, gap):                        # layers as 3D slabs
        for k in range(n):
            x = x0 + gap * k
            ax.add_patch(Polygon([(x, y0 + h), (x + dx, y0 + h + dy), (x + w + dx, y0 + h + dy), (x + w, y0 + h)],
                                 fc="#bfe6d4", ec=GREEN, lw=0.4, zorder=3))
            ax.add_patch(Polygon([(x + w, y0), (x + w + dx, y0 + dy), (x + w + dx, y0 + h + dy), (x + w, y0 + h)],
                                 fc="#9fd8bf", ec=GREEN, lw=0.4, zorder=3))
            ax.add_patch(R((x, y0), w, h, fc="#e3f5ec", ec=GREEN, lw=0.4, zorder=3))

    d = np.load(f"{DATA}/pilot/rtmw_{INPUT_SIGN.split('/')[0]}.npz")
    i0 = d["off"][list(d["relpath"]).index(INPUT_SIGN)]
    frame = lambda t: d["kp"][i0 + t].astype(float)
    kp = frame(INPUT_T)
    BODY = [0, *range(3, 11)]
    PARTS = [("body", 9, BODY, "#e9e8e4"), ("left hand", 21, list(range(91, 112)), HAND),
             ("right hand", 21, list(range(112, 133)), HAND), ("face", 18, U_FACE, "#f5ecc9")]

    def part_icon(a, idx):
        """One body part from the example frame, colored as in Fig. 1, 2 and 5."""
        if idx[0] >= 91:
            col = SERIES[1] if idx[0] == 91 else SERIES[0]
            for p, q in RT_HAND:
                a.plot(kp[[idx[p], idx[q]], 0], kp[[idx[p], idx[q]], 1], color=col, lw=0.8)
        elif idx is BODY:
            col = INK
            for p, q in ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10)):
                a.plot(kp[[p, q], 0], kp[[p, q], 1], color=INK2, lw=0.8)
        else:
            col = INK
            jaw, lips = [k for k in idx if k < 40], [k for k in idx if 83 <= k <= 90]
            a.plot(kp[jaw, 0], kp[jaw, 1], color=INK2, lw=0.6)
            a.plot(kp[lips + lips[:1], 0], kp[lips + lips[:1], 1], color=INK2, lw=0.6)
        a.scatter(kp[idx, 0], kp[idx, 1], s=3, color=col, zorder=3, linewidths=0, marker="s" if idx[0] == 91 else "o")
        lo, hi = kp[idx].min(0), kp[idx].max(0)
        c, sz = (lo + hi) / 2, (hi - lo).max() * 0.62
        a.set_xlim(c[0] - sz, c[0] + sz); a.set_ylim(c[1] + sz, c[1] - sz); a.set_aspect("equal"); a.axis("off")

    def cards(frames, x0, y0, w, h, step):                          # stack of cards (video frames or skeletons)
        out = []
        for k, f in enumerate(frames):
            a = inset(x0 + step[0] * k, y0 + step[1] * (len(frames) - 1 - k), w, h)
            a.set_xticks([]); a.set_yticks([])
            for sd in a.spines.values():
                sd.set_color(AXIS); sd.set_linewidth(0.6)
            out.append(a)
        return out

    with plt.rc_context({"mathtext.fontset": "stix"}):
        # training box
        rbox(35, 75.5, 180, 87, "#ffffff", MUTED, 0.7, "--")
        ax.text(37, 84.6, "Fine-tuning", fontsize=6.6, fontweight="bold", color=INK2, va="center", family=SANS)
        ax.text(37, 81, "All modules are trained end to end except the mT5 token embeddings. AdamW, lr $3\\times10^{-4}$, "
                "cosine schedule, 20 epochs, batch 32, label smoothing 0.2.", fontsize=6, color=INK2, va="center")
        ax.text(37, 77.6, "Up to 64 frames per clip: sampled at random in training, evenly spaced in testing.", fontsize=6,
                color=INK2, va="center")

        # (1) input: SIBI dictionary video frames (not the dataset) -> RTMW -> stack of skeleton frames of signer 1
        panel(1, 12, 26, 73, 1, "Input", GREY, "#ffffff")
        import cv2
        cap = cv2.VideoCapture(os.path.join(DICT, "video", f"{KAMUS_SIGN}.webm"))
        kpose = parts(np.load(os.path.join(DICT, "keypoints", f"{KAMUS_SIGN}.npy"))[KAMUS_T])[0]
        for a, t in zip(cards(range(3), 4.5, 51.7, 9.5, 11.5, (1.8, 1.8)), (KAMUS_T - 16, KAMUS_T - 8, KAMUS_T)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, t)
            ok, bgr = cap.read()
            assert ok, (KAMUS_SIGN, t)
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            Hh, Ww = img.shape[:2]
            sh = kpose[[11, 12], :2] * [Ww, Hh]                         # window as in Fig. 2c, from the shoulders in KAMUS_T
            kx, ky = sh.mean(0)
            kh = 1.15 * np.linalg.norm(sh[0] - sh[1])
            top = int(max(ky - 1.45 * kh, 0)); bot = int(min(top + 2.5 * kh, Hh))
            a.imshow(img[top:bot, max(int(kx - (bot - top) / 2.5), 0):int(kx + (bot - top) / 2.5)])
        cap.release()
        arrow(18.0, 53.0, 21.6, 50.8, col=MUTED)
        ax.text(21.9, 48.9, "$t$", fontsize=6.3, color=INK2)
        ax.text(12.4, 48.0, "video frames", ha="center", fontsize=6, style="italic", family=SANS, color=INK2)
        rbox(3.4, 42.4, 23.6, 46.2, "#ecebe7", "none", r=1.0, z=2)
        ax.text(13.5, 44.3, "RTMW pose estimator", ha="center", va="center", fontsize=5.9, zorder=3)
        arrow(13.5, 42.4, 13.5, 40.6)
        for a, t in zip(cards(range(3), 4.5, 21.6, 12, 14.2, (2.0, 2.3)), (INPUT_T - 10, INPUT_T - 5, INPUT_T)):
            f = frame(t)
            draw_unisign(a, f, [])
            cx, cy = f[[5, 6]].mean(0)
            half = 1.15 * np.linalg.norm(f[5] - f[6])
            a.set_xlim(cx - half, cx + half); a.set_ylim(cy + 0.8 * half, cy - 0.95 * half); a.set_aspect("equal")
            a.set_facecolor("#ffffff")
        arrow(20.8, 23.0, 23.8, 21.0, col=MUTED)
        ax.text(24.2, 21.4, "$T$", fontsize=6.5, color=INK2)
        ax.text(12.2, 19.6, "keypoint sequence", ha="center", fontsize=6, style="italic", family=SANS, color=INK2)
        ax.text(13.5, 16.3, "$x_t\\in\\mathbb{R}^{69\\times3}$, $T\\leq64$", ha="center", fontsize=6.3)
        ax.text(13.5, 13.4, "normalized per part", ha="center", fontsize=6, color=INK2)
        arrow(26, 20.5, 37, 20.5)
        ax.text(30.5, 25.6, "divide\ninto\nparts", ha="center", va="center", fontsize=5.8, style="italic", family=SANS,
                color=INK2, linespacing=1.0)

        # (2) pose encoder per body part (bottom to top, as in Uni-Sign)
        panel(35, 12, 118, 73, 2, "Uni-Sign pose encoder, per body part", "#184f95", "#f7faff")   # named after the source
        xs = (47, 66.5, 86, 105.5)
        for xc, (name, n, idx, fill) in zip(xs, PARTS):
            part_icon(inset(xc - 7, 16, 14, 12.5), idx)
            ax.text(xc, 13.6, f"{name} ({n})", ha="center", fontsize=6, color=INK)
            arrow(xc, 28.8, xc, 30.7, col=BLUE)
            block(xc, 32.8, 15, 4.0, "Linear 3 → 64", fill, 5.9)
            arrow(xc, 34.8, xc, 36.5, col=BLUE)
            block(xc, 39.2, 15, 5.2, "spatial GCN", fill)
            arrow(xc, 41.8, xc, 46.3, col=BLUE)
            block(xc, 49, 15, 5.2, "temporal GCN\n(5 frames)", fill, 5.8)
            arrow(xc, 51.6, xc, 53.3, col=BLUE)
            block(xc, 55.2, 15, 3.6, "mean over points", fill, 5.8)
            arrow(xc, 57, xc, 59.4, col=BLUE, label="256")
        ax.plot([xs[0], xs[3]], [44, 44], color=GREY, lw=0.6, ls=(0, (2.5, 1.6)), zorder=3)   # models.py:240-255
        ax.plot(xs[0], 44, "o", ms=2.2, color=GREY, zorder=4)
        for xc, lab in zip(xs[1:], ("wrist", "wrist", "nose")):
            ax.add_patch(Circle((xc, 44), 1.05, fc="#ffffff", ec=GREY, lw=0.6, zorder=5))
            ax.text(xc, 44.05, "+", ha="center", va="center", fontsize=6, color=GREY, zorder=6)
            ax.text(xc + 1.6, 44, lab, ha="left", va="center", fontsize=5.8, color=GREY, zorder=6,
                    bbox=dict(fc="#f7faff", ec="none", pad=0.3))
        rbox(38, 59.6, 115, 65.6, "#ececea", "none", r=1.8, z=2)        # wide concatenation bar (like Feature Concatenation)
        ax.text(76.5, 62.6, "Concat 4 × 256 + part embedding  →  Linear  →  $e_t\\in\\mathbb{R}^{768}$",
                ha="center", va="center", fontsize=6.3, zorder=3)
        ax.plot([115, 119.5, 119.5], [62.6, 62.6, 42.7], color=GREY, lw=0.8, zorder=4)
        arrow(119.5, 42.7, 122.3, 42.7)

        # (3) mT5 encoder: text prefix from above, pose tokens from the left, layers as 3D slabs, output below
        panel(121, 12, 152, 73, 3, "mT5 encoder", "#127a55", "#f5fbf8", sub="pre-trained; decoder unused")
        tokens(123.4, 37.2, 4, "#e9f0fb", BLUE, vertical=True)
        ax.text(124.6, 33.8, "$e_{1..T}$", ha="center", fontsize=6, color=INK2)
        arrow(126.4, 42.7, 129.2, 42.7, col=GREEN)
        tokens(132.4, 58.0, 4, "#ffffff", MUTED)
        ax.text(136.8, 62.4, "text prefix (8 tokens)", ha="center", fontsize=6, color=INK2)
        arrow(136.8, 57.0, 136.8, 54.6, col=GREEN)
        slabs(129.6, 33.5, 4, 3.0, 16, 2.0, 2.4, 4.3)
        ax.text(139.8, 30.6, "12 Transformer layers", ha="center", fontsize=6, color=INK)
        ax.text(139.8, 27.6, "$d=768$", ha="center", fontsize=6, color=INK2)
        arrow(139.8, 26.2, 139.8, 23.2, col=GREEN)
        tokens(127.4, 19.2, 4, "#f2f2f0", "#c8c7c2", group=False)
        tokens(139.4, 19.2, 4, "#cfe8dc", GREEN, hatch="//////")
        ax.text(131.6, 17.4, "prefix\n(ignored)", ha="center", va="top", fontsize=6, color=MUTED, linespacing=1.0)
        ax.text(143.8, 17.4, "pose\noutputs", ha="center", va="top", fontsize=6, color=INK2, linespacing=1.0)
        arrow(151.6, 20.4, 157, 20.4)

        # (4) our classification head (bottom to top)
        panel(155, 12, 180, 73, 4, "Classifier (ours)", INK, "#f8f8f6", 6.4)
        block(167.5, 21.2, 21, 8.2, "mean over\npose tokens\n(valid frames)", "#e6e6e3", 5.8)
        arrow(167.5, 25.3, 167.5, 30.6)
        block(167.5, 34, 21, 6.6, "Linear 768 → 114", "#e6e6e3", 6)
        arrow(167.5, 37.3, 167.5, 43.6)
        for k, v in enumerate((0.92, 0.35, 0.22, 0.12, 0.06)):
            ax.add_patch(R((159.5, 57.4 - 3.1 * k), 15 * v, 2.2, fc=TEST if k == 0 else "#c9d9ef", ec="none", zorder=3))
        ax.text(167.5, 63.6, "$\\hat{y}\\in\\mathbb{R}^{114}$", ha="center", fontsize=6.4)
        ax.text(167.5, 60.8, "class scores", ha="center", fontsize=6, color=INK2)

        # two-row legend
        rbox(1, -4, 180, 9.6, "#ffffff", AXIS, 0.6)

        def row(y, items):
            x = 5
            for kind, a1, a2, lab in items:
                if kind == "mk":
                    ax.plot(x, y, marker=a1, color=a2, ms=3.5, ls="none")
                elif kind == "sq":
                    ax.add_patch(R((x - 1.2, y - 1.2), 2.4, 2.4, fc=a1, ec=a2[0], lw=0.5, hatch=a2[1]))
                else:
                    ax.plot([x - 2.2, x + 1.2], [y, y], color=GREY, lw=0.6, ls=(0, (2.5, 1.6)))
                    ax.plot(x - 2.2, y, "o", ms=2.2, color=GREY)
                ax.text(x + 2.2, y, lab, fontsize=6, va="center", family=SANS)
                x += 7.5 + len(lab) * 1.2

        row(5.8, [("mk", "o", SERIES[0], "right hand"), ("mk", "s", SERIES[1], "left hand"),
                  ("mk", "o", INK, "body and face keypoints"), ("sq", HAND, ("none", None), "both hands: shared weights"),
                  ("ln", None, None, "body feature added (no gradient)")])
        row(-0.4, [("sq", "#ffffff", (MUTED, None), "text token (frozen embedding)"), ("sq", "#e9f0fb", (BLUE, None), "pose token"),
                   ("sq", "#cfe8dc", (GREEN, "//////"), "encoder output"), ("sq", "#f2f2f0", ("#c8c7c2", None), "output not used")])
    save(fig, "fig_unisign")
