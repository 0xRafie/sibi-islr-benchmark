"""Paired statistics: Cliff's delta, Hodges-Lehmann, Wilcoxon comparisons, Holm correction and the
Friedman/Nemenyi test (moved from analyze.py)."""

import numpy as np

from sibi_islr.evaluation.metrics import per_fold, scores


def cliffs(a, b):
    a, b = np.asarray(a)[:, None], np.asarray(b)[None, :]
    return float((a > b).mean() - (a < b).mean())


def hodges_lehmann(d, alpha=0.05):
    """Hodges-Lehmann estimate and its exact CI, dual to the exact Wilcoxon signed-rank test (no ties, n >= 6)."""
    n = len(d)
    w = np.sort((d[:, None] + d[None, :])[np.triu_indices(n)] / 2)             # n(n+1)/2 Walsh averages
    c = np.zeros(n * (n + 1) // 2 + 1)
    c[0] = 1
    for k in range(1, n + 1):                                                  # null distribution of W+, dynamic programming
        c[k:] = c[k:] + c[:-k]
    k = int(np.searchsorted(np.cumsum(c) / 2 ** n, alpha / 2, side="right")) - 1  # largest k with P(W+ <= k) <= alpha/2
    assert k >= 0, n
    return float(np.median(w)), float(w[k]), float(w[-1 - k])


# Leave-one-signer-out fold = subject - 1. Session pairs (subjects 4-5, 11-12, ..., 21-22): second fold -> first fold
SESSION = {4: 3, 11: 10, 13: 12, 15: 14, 17: 16, 19: 18, 21: 20}


def compare(a_cells, b_cells, n_boot=10000, matched=False, sessions=False, k=0):
    """Model a against reference b, per fold.

    Main analysis: per fold, the mean over all seeds of each model, so a candidate with seed 0 is compared with the
    mean of the BiLSTM's five seeds. matched=True (sensitivity): only (fold, seed) pairs present in both.
    sessions=True (sensitivity, leave-one-signer-out only): the folds of a session pair are averaged into one unit,
    22 folds -> 15 units. k: the tested metric, 0 = accuracy (family a), 1 = macro-F1 (family b); `f1_diff` is always
    macro-F1.
    """
    from scipy.stats import wilcoxon
    if matched:
        common = set(a_cells) & set(b_cells)
        a_cells, b_cells = ({k: c[k] for k in common} for c in (a_cells, b_cells))
    A, B = ([per_fold(c, i) for i in (0, 1)] for c in (a_cells, b_cells))
    folds = sorted(set(A[0]) & set(B[0]))
    unit = (lambda f: SESSION.get(f, f)) if sessions else (lambda f: f)
    units = sorted({unit(f) for f in folds})
    agg = lambda M: np.array([np.mean([M[f] for f in folds if unit(f) == u]) for u in units])
    a, b = agg(A[k]), agg(B[k])
    d = a - b
    rg = np.random.default_rng(0)
    boot = d[rg.integers(0, len(d), (n_boot, len(d)))].mean(1)
    p = float(wilcoxon(a, b).pvalue) if len(d) > 1 and np.any(d != 0) else 1.0
    hl = hodges_lehmann(100 * d) if len(d) >= 6 else (np.nan,) * 3
    return dict(folds=len(units), diff=100 * d.mean(), lo=100 * np.percentile(boot, 2.5), hi=100 * np.percentile(boot, 97.5),
                p=p, cliff=cliffs(a, b), f1_diff=100 * (agg(A[1]) - agg(B[1])).mean(), acc_diff=100 * (agg(A[0]) - agg(B[0])).mean(),
                hl=hl[0], hl_lo=hl[1], hl_hi=hl[2])


def holm(ps):
    """Holm-Bonferroni adjusted p-values, in input order."""
    order, out, run = np.argsort(ps), np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = run
    return out.tolist()


def friedman_nemenyi(scores, alpha=0.05):
    """Descriptive Friedman test over folds with the Nemenyi post-hoc test, and the cliques of a critical-difference
    diagram built from the SAME critical difference, so the diagram cannot contradict the pairwise tests (the
    Nemenyi-versus-CD inconsistency in Ho and Santoso). scores: {model: {fold: value}}; only folds complete for every
    model are used. Rank 1 = best. -> chi2, p, number of folds, mean ranks, CD, pairs that differ (rank difference > CD),
    cliques (rank range <= CD)."""
    from scipy.stats import friedmanchisquare, rankdata, studentized_range
    models = sorted(scores)
    folds = sorted(set.intersection(*(set(scores[m]) for m in models)))
    X = np.array([[scores[m][f] for m in models] for f in folds])
    chi2, p = friedmanchisquare(*X.T)
    R = np.mean([rankdata(-row) for row in X], 0)
    k, n = len(models), len(folds)
    cd = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * n))
    order = np.argsort(R)
    diff = [(models[i], models[j]) for a, i in enumerate(order) for j in order[a + 1:] if R[j] - R[i] > cd]
    cliques = []
    for a, i in enumerate(order):
        grp = [models[j] for j in order[a:] if R[j] - R[i] <= cd]
        if len(grp) > 1 and not any(set(grp) <= set(c) for c in cliques):
            cliques.append(grp)
    return dict(chi2=float(chi2), p=float(p), n=n, ranks=dict(zip(models, map(float, R))), cd=float(cd), different=diff,
                cliques=cliques)
