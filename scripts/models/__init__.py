"""Shared input code for the models in this package (scripts/train_slot2.py).

F0 input for models that also use body points: the same transformation as the BiLSTM input (gate_variants.normalize,
'shoulder'), applied to all points. Output indices of `f0()`: MediaPipe pose 0-32, left hand 33-53, right hand 54-74
(MediaPipe order).

    python -c "import sys; sys.path.insert(0, 'scripts'); import models; models.demo()"
"""
import numpy as np

from gate import fill_gaps
from gate_variants import L_SH, R_SH, fill_pose, normalize

LH, RH = 33, 54   # first index of the left and right hand in the f0() output


def split(seq):
    """seq150 (T, 150) -> pose (T, 33, 2), hands (T, 2, 21, 2), hand detections (T, 2). Zero = not detected."""
    T = len(seq)
    h = seq[:, 66:].reshape(T, 2, 21, 2)
    return seq[:, :66].reshape(T, 33, 2), h, np.any(h != 0, axis=(2, 3))


def f0(seq):
    """seq150 -> (T, 75, 2), the F0 input before resampling.

    - Shoulder midpoint c and shoulder width w are averaged over the frames in which both shoulders are detected. Every
      point becomes (p - c) / w. Without any shoulder, c = 0 and w = 1, as for the BiLSTM input.
    - Hands: frames without a detection are interpolated inside the detected range of that hand and set to 0 outside it
      (`fill_gaps` + mask), exactly as for the BiLSTM input.
    - Pose: frames without a pose are interpolated (`fill_pose`). A clip without any pose stays 0.
    """
    pose, h, det = split(seq)
    sh = pose[:, [L_SH, R_SH]]
    ok = np.all(sh != 0, axis=(1, 2))
    c, w = (sh[ok].mean(1).mean(0), np.linalg.norm(sh[ok, 0] - sh[ok, 1], axis=1).mean() + 1e-6) if ok.any() else (0.0, 1.0)
    h, mask = fill_gaps(np.where(det[..., None, None], (h - c) / w, 0), det)
    h = h * mask[:, :, None, None]
    has = np.any(pose != 0, axis=(1, 2))
    pose = (fill_pose(pose, has) - c) / w if has.any() else pose
    return np.concatenate([pose, h.reshape(len(seq), 42, 2)], 1).astype(np.float32)


def demo():
    rg = np.random.default_rng(0)
    seq = rg.uniform(0.2, 0.8, (20, 150)).astype(np.float32)
    seq[3:5, 66:108] = 0                                   # left hand missing mid-clip: interpolated
    seq[:2, 108:] = 0                                      # right hand missing at the start: outside the range, 0
    seq[7, :66] = 0                                        # one frame without pose
    x = f0(seq)
    h, det = normalize(seq, "shoulder")
    gh, mask = fill_gaps(h, det)
    assert np.allclose(x[:, LH:].reshape(20, 2, 21, 2), gh * mask[:, :, None, None], atol=1e-6), "hands != BiLSTM input"
    assert np.all(x[:2, RH:] == 0) and np.allclose(x[3, LH:RH], (x[2, LH:RH] * 2 + x[5, LH:RH]) / 3, atol=1e-5)
    assert np.allclose(x[7, :33], (x[6, :33] + x[8, :33]) / 2, atol=1e-5)
    mid = (x[:, L_SH] + x[:, R_SH]) / 2
    assert abs(np.delete(mid, 7, 0).mean()) < 1e-5     # mean shoulder midpoint of the clip at 0
    print("demo ok: f0 hands = BiLSTM input, same transformation for the pose, pose gaps filled")
