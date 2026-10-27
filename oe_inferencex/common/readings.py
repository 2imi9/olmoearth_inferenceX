"""The second reading of the labels (--also-accept): class codes, the checked crosswalk, the labels read again."""
import numpy as np


def _code(x, strings=True):
    """A class code as an int: from an int, a whole float, or (with `strings`, as JSON keys are) the string of a whole
    number; None for anything else."""
    if isinstance(x, (bool, np.bool_)):
        return None
    if isinstance(x, str):
        if not strings:
            return None
        try:
            return int(x.strip())
        except ValueError:
            return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return int(v) if np.isfinite(v) and v == int(v) else None


def _also_accept_pairs(also_accept):
    """{predicted class: sorted reference classes also accepted for it} from a mapping of class codes (ints, or their
    strings as JSON keys) to lists of class codes; ValueError saying what is wrong otherwise."""
    shape = 'such as {"3": [4]}: each predicted class code to the list of reference class codes also accepted for it'
    if not isinstance(also_accept, dict):
        raise ValueError(f"also_accept maps class codes to lists of class codes, {shape}; got a "
                         f"{type(also_accept).__name__}")
    pairs = {}
    for key, codes in also_accept.items():
        c = _code(key)
        if c is None:
            raise ValueError(f"also_accept: the key {key!r} is not a class code (a whole number), {shape}")
        if c in pairs:
            raise ValueError(f"also_accept names class {c} twice")
        if not isinstance(codes, (list, tuple)):
            raise ValueError(f"also_accept: class {c} maps to {codes!r}, not a list of class codes, {shape}")
        got = [_code(v, strings=False) for v in codes]
        if any(v is None for v in got):
            bad = next(v for v, g in zip(codes, got) if g is None)
            raise ValueError(f"also_accept: class {c} lists {bad!r}, which is not a class code (a whole number)")
        pairs[c] = sorted(set(got))
    return pairs


def also_accept_reading(wrong, map_class, reference, also_accept, unjudged=None):
    """The labels read a second time, a window of map class c also counted right when its reference class is one
    `also_accept` lists for c (a class listed for itself changes nothing). Returns {"wrong": 0/1 per labelled window
    under this reading, `wrong` as given where unjudged; "pairs": the checked crosswalk (_also_accept_pairs);
    "errors_accepted": {map class: {reference class: windows}}, the judged windows wrong under the strict reading that
    this one accepts (named after the audit's errors_by_reference: in the audit's block, `accepted` is the list of
    codes a class accepts); "n_accepted"; "unused": listed classes no labelled window is mapped as}. ValueError when
    the lengths differ, `wrong` is not 0/1, a judged reference is not a whole number of at least 0, or `wrong`
    disagrees with reference != map_class at a judged window (the first position named).

    A window this reading accepts is wrong under the strict one, so its `wrong` is at most the strict one window by
    window, and the windows that could not be judged are the same in both. Passed to `estimate_error_rate`,
    `certify_zone`, `certify_by_condition` or `sequential.certify_zone_sequential` in place of the strict `wrong`, it
    gets that function's own guarantee for this reading's definition of an error, when `also_accept` was fixed before
    any label was read (ESTIMATE_READING_NOTE, ZONE_READING_NOTE)."""
    w = np.asarray(wrong).ravel()
    mc = np.asarray(map_class).ravel()
    ref = np.asarray(reference).ravel()
    unj = np.zeros(w.size, bool) if unjudged is None else np.asarray(unjudged, bool).ravel()
    if not w.size == mc.size == ref.size == unj.size:
        raise ValueError(f"{w.size} labels but {mc.size}, {ref.size} and {unj.size} map classes, references and flags")
    if not np.isin(w, (0, 1)).all():
        raise ValueError("wrong must be 0 or 1 per window")
    pairs = _also_accept_pairs(also_accept)
    judged = ~unj
    r = np.where(judged, ref, 0).astype(np.float64)               # a ? window's reference is not read
    bad = np.flatnonzero(~(np.isfinite(r) & (r == np.rint(r)) & (r >= 0)))
    if bad.size:
        k = int(bad[0])
        raise ValueError(f"a judged window's reference class is a whole number of at least 0; window {k} has "
                         f"{ref.tolist()[k]!r}")
    # both readings read the same labels: a `wrong` that disagrees with the classes would make them read different ones
    off = np.flatnonzero(judged & ((w == 1) != (ref != mc)))
    if off.size:
        k = int(off[0])
        w_k, c_k, r_k = w.tolist()[k], mc.tolist()[k], ref.tolist()[k]
        raise ValueError(f"wrong disagrees with whether the reference class differs from the map's class at window {k} "
                         f"(wrong {w_k!r}, map class {c_k!r}, reference class {r_k!r}); the second reading reads the "
                         "same labels as the first")
    accepted = np.zeros(w.size, bool)
    for c, codes in pairs.items():
        more = [a for a in codes if a != c]
        if more:
            accepted |= judged & (mc == c) & np.isin(ref, more)
    out = w.copy()
    out[accepted] = 0
    by = {}
    for c in np.unique(mc[accepted]):
        vals, cnt = np.unique(ref[accepted & (mc == c)], return_counts=True)
        by[str(int(c))] = {str(int(v)): int(n) for v, n in zip(vals, cnt)}
    return {"wrong": out, "pairs": {str(c): v for c, v in sorted(pairs.items())}, "errors_accepted": by,
            "n_accepted": int(accepted.sum()), "unused": [str(c) for c in sorted(pairs) if not (mc == c).any()]}
