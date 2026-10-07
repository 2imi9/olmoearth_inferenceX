"""The test map of the README's quick start, and a stand-in for the reviewer who labels it.

    python quickstart_map.py                      # writes scores.tif, other.tif and truth.tif
    python quickstart_map.py --label to_label.csv # fills `wrong` and `reference_class` from truth.tif
                                                  # (`reference_class` alone on a sample drawn with --other)

scores.tif and other.tif are two four-class probability maps of one synthetic scene of 256 x 256 pixels,
shape (4, 256, 256), float32. truth.tif holds the class that is really there. Most errors sit on the class
boundaries and have a low top probability; a few sit inside a class and are confident, which no ranking by
confidence finds.

Everything is built from whole numbers, so the three files are the same on every machine and the commands in the
README print the same lines. Needs numpy and rasterio (pip install "olmoearth-inferencex[geo]").
"""
import csv
import sys

import numpy as np

C, SIDE, PATCH = 4, 256, 4          # classes, pixels per side, pixels per window side
UNIT = 4096                         # a probability is a whole number of 1/4096, so window means are exact in float32


def window_counts(classes):
    """How many pixels of each class every window holds: shape (C, windows, windows)."""
    n = SIDE // PATCH
    blocks = classes.reshape(n, PATCH, n, PATCH).transpose(0, 2, 1, 3).reshape(n, n, PATCH * PATCH)
    return np.stack([(blocks == k).sum(-1) for k in range(C)])


def to_windows(classes):
    """The majority class of each window."""
    return window_counts(classes).argmax(0)


def truth_map():
    """The class per pixel: four regions with curved boundaries and one island. A window split evenly between
    classes is given whole to one of them, so every window has one majority class and nothing is ambiguous."""
    r, c = np.mgrid[0:SIDE, 0:SIDE]
    lower = (r + (c - 128) ** 2 // 256) > 150
    right = (c + (r - 100) ** 2 // 200) > 128
    truth = 2 * lower + right
    truth[(r - 56) ** 2 + (c - 56) ** 2 < 28 ** 2] = 3
    counts = window_counts(truth)
    split = (counts == counts.max(0)).sum(0) > 1
    up = np.repeat(np.repeat(split, PATCH, 0), PATCH, 1)
    whole = np.repeat(np.repeat(counts.argmax(0), PATCH, 0), PATCH, 1)
    return np.where(up, whole, truth).astype(np.uint8)


def score_map(truth, seed):
    """One model's probabilities for the scene. Errors are drawn per window, most often on a class boundary."""
    rs = np.random.RandomState(seed)            # the legacy generator: its stream never changes between numpy versions
    n = SIDE // PATCH
    tw = to_windows(truth)
    pad = np.pad(tw, 2, mode="edge")
    ring = lambda k: np.stack([pad[2 + i:2 + i + n, 2 + j:2 + j + n]                       # noqa: E731
                               for i in range(-k, k + 1) for j in range(-k, k + 1)])
    mixed = truth.reshape(n, PATCH, n, PATCH).transpose(0, 2, 1, 3).reshape(n, n, -1)
    on_boundary = (ring(1) != tw).any(0) | (mixed != mixed[..., :1]).any(-1)
    near_boundary = (ring(2) != tw).any(0) & ~on_boundary

    # which windows the model gets wrong, per mille: 300 on a boundary, 100 next to one, 30 elsewhere
    rate = np.where(on_boundary, 300, np.where(near_boundary, 100, 30))
    wrong = rs.randint(0, 1000, size=(n, n)) < rate
    other_class = (tw + 1 + rs.randint(0, C - 1, size=(n, n))) % C
    # the top probability per window, in 1/4096: high inside a class, lower on a boundary, low where wrong,
    # except one wrong window in six, which is as confident as a correct one
    top = np.where(on_boundary, rs.randint(1900, 3500, size=(n, n)), rs.randint(3000, 4000, size=(n, n)))
    confident_error = rs.randint(0, 6, size=(n, n)) == 0
    top = np.where(wrong & ~confident_error, rs.randint(1750, 2900, size=(n, n)), top)

    up = lambda a: np.repeat(np.repeat(a, PATCH, 0), PATCH, 1)                              # noqa: E731
    predicted = np.where(up(wrong), up(other_class), truth)
    top_px = np.clip(up(top) + rs.randint(-150, 151, size=(SIDE, SIDE)), 1700, 4000)
    rest = UNIT - top_px                        # shared by the three other classes as 1/2, 1/3 and what is left
    parts = np.stack([rest // 2, rest // 3, rest - rest // 2 - rest // 3])
    turn = rs.randint(0, 3, size=(SIDE, SIDE))
    counts = np.zeros((C, SIDE, SIDE), dtype=np.int64)
    for k in range(C):
        slot = (k - predicted - 1) % C          # 0, 1, 2 for the other classes; 3 for the predicted one
        share = np.take_along_axis(parts, ((np.minimum(slot, 2) + turn) % 3)[None], 0)[0]
        counts[k] = np.where(slot == 3, top_px, share)
    assert (counts.sum(0) == UNIT).all() and (counts.argmax(0) == predicted).all()
    return (counts / UNIT).astype(np.float32)


def write_maps():
    import rasterio
    from rasterio.transform import from_origin

    truth = truth_map()
    profile = dict(driver="GTiff", height=SIDE, width=SIDE, crs="EPSG:32632",
                   transform=from_origin(500000, 5000000, 10, 10))
    for name, seed in (("scores.tif", 1), ("other.tif", 2)):
        scores = score_map(truth, seed)
        with rasterio.open(name, "w", count=C, dtype="float32", **profile) as dst:
            dst.write(scores)
        n_wrong = int((to_windows(scores.argmax(0)) != to_windows(truth)).sum())
        n = (SIDE // PATCH) ** 2
        print(f"wrote {name}: {C} classes, {SIDE} x {SIDE} pixels; {n_wrong} of its {n} windows are wrong "
              f"({n_wrong / n:.1%})")
    with rasterio.open("truth.tif", "w", count=1, dtype="uint8", **profile) as dst:
        dst.write(truth, 1)
    print("wrote truth.tif: the class that is really there")


def label(path, truth_path="truth.tif"):
    """What a reviewer does by eye: for each sampled window, write the class that is really there and, on a sample
    of one map, whether the map's class differs from it. A sample drawn with --other (two maps) has no map_class and
    no `wrong`; it takes the class seen alone, and estimate says which map is more accurate."""
    import rasterio

    with rasterio.open(truth_path) as src:
        truth = src.read(1)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    fields = list(rows[0])
    if "reference_class" not in fields:
        fields.append("reference_class")
    one_map = "map_class" in fields          # a pairs CSV crashed here on a KeyError until 2026-10-06
    for row in rows:
        r, c = int(row["pixel_row"]), int(row["pixel_col"])
        seen = int(np.bincount(truth[r:r + PATCH, c:c + PATCH].ravel()).argmax())
        row["reference_class"] = seen
        if one_map:
            row["wrong"] = int(seen != int(row["map_class"]))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"labelled {len(rows)} windows in {path}"
          + (f": {sum(row['wrong'] for row in rows)} wrong" if one_map else ": the class seen in each"))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--label":
        label(sys.argv[2])
    elif len(sys.argv) == 1:
        write_maps()
    else:
        raise SystemExit(__doc__)
