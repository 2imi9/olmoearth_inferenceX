#!/bin/bash
#SBATCH -A p2026_0089_neu
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -J e98read
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/%x-%j.out
# exp98 stage 5 (exp/jobs/E98_README.md): read the collected run into the rasters exp98 reads, with
# `oe-inferencex from-olmoearth`, in parts. The reader holds one CRS's grid in memory up to 4 GiB (MAX_GRID_BYTES in
# oe_inferencex/olmoearth.py), and this map is about 2e8 pixels x 44 bytes (10 float32 bands and an int32 owner), about
# 8.8 GB, and olmoearth_run writes one rslearn dataset root per partition. So every root is split into parts of at most
# E98_PART_GIB (default 3) GiB of grid, each part a set of windows that do not overlap one another, and each part is
# read to its own directory:
#   /home/qi_zim_neu/olmoearth_inferenceX/data/exp98/scores/r<root>_p<part>/   (data/ is gitignored)
#     scores_<EPSG>.tif, condition_<EPSG>.tif, olmoearth_conditions.json, olmoearth_output.json
#   /home/qi_zim_neu/olmoearth_inferenceX/data/exp98/scores/parts.json   which windows each part holds (names only)
# and counts only (roots, parts, windows and grid size per part; no window name, no bound) to exp/out/exp98/e98_read.json.
# exp98 (exp/jobs/e98.sh) reads every part directory as one map: a pixel covered in two parts counts once, and a point
# in two parts takes the one that covers it.
#
# Needs the from-olmoearth reader on main: the job resets the checkout to origin/main and refuses if
# oe_inferencex/olmoearth.py is not there (it is on the from-olmoearth branch until that branch is merged).
#   ssh aicr "sbatch --parsable --dependency=afterok:$COLL_JOB" < exp/jobs/e98_read.sh
# E98_REREAD=1 replaces an earlier read.
set -euo pipefail
REPO=/home/qi_zim_neu/olmoearth_inferenceX
cd "$REPO" || exit 1
git fetch -q origin main
git reset -q --hard origin/main
if [ -n "${E98_SHA:-}" ] && [ "$(git rev-parse HEAD)" != "$E98_SHA" ]; then
  echo "checkout is $(git rev-parse HEAD), not E98_SHA=$E98_SHA"; exit 2
fi
test -f oe_inferencex/olmoearth.py \
  || { echo "oe_inferencex/olmoearth.py is not on origin/main: merge the from-olmoearth branch first"; exit 2; }
SCRATCH=/scratch/qi_zim_neu/olmoearth_inferenceX
export DEST=$REPO/data/exp98/awf_run SCORES=$REPO/data/exp98/scores OUT=$REPO/exp/out/exp98
export E98_PART_BYTES=$(( ${E98_PART_GIB:-3} * 2**30 ))
export PYTHONUNBUFFERED=1 UV_CACHE_DIR=$SCRATCH/uv-cache OMP_NUM_THREADS=8
export PATH="$HOME/.local/bin:$PATH"

echo "== $(date -Is) job ${SLURM_JOB_ID:-none} on $(hostname): exp98 read, commit $(git rev-parse HEAD) =="
test -f "$DEST/records/prepare_inventory.json" || { echo "no collected run at $DEST: run e98_collect.sh first"; exit 1; }
test -f "$OUT/e98_collect.json" || { echo "no $OUT/e98_collect.json: run e98_collect.sh first"; exit 1; }
if [ -e "$SCORES" ]; then
  if [ "${E98_REREAD:-0}" = 1 ]; then rm -rf "$SCORES"; else echo "$SCORES exists; E98_REREAD=1 replaces it"; exit 2; fi
fi
mkdir -p "$SCORES.partial" "$OUT"
uv run --extra geo oe-inferencex from-olmoearth --help > /dev/null \
  || { echo "oe-inferencex from-olmoearth does not run in this checkout"; exit 1; }

uv run --extra geo python - <<'EOF'
import json, os, subprocess
dest, scores, out = os.environ["DEST"], os.environ["SCORES"] + ".partial", os.environ["OUT"]
limit = int(os.environ["E98_PART_BYTES"])
BYTES_PER_PX = 4 * 10 + 4            # the reader's own count for 10 float32 bands and an int32 owner
inv = json.load(open(os.path.join(dest, "records", "prepare_inventory.json")))
assert json.load(open(os.path.join(out, "e98_collect.json"))).get("manifest_verified"), "collect is not verified"


def esc(s):                          # a window id as a literal fnmatch pattern
    return "".join(f"[{ch}]" if ch in "[]*?" else ch for ch in s)


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def union(a, b):
    return b if a is None else [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def meet(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


plan, counts = [], {"roots": [], "part_limit_gib": limit / 2**30}
for ri, r in enumerate(inv["roots"]):
    root = os.path.join(dest, r["root_id"])       # collect copies each root as root_<i>
    wins = []
    for group in sorted(os.listdir(os.path.join(root, "windows"))):
        gd = os.path.join(root, "windows", group)
        for name in sorted(os.listdir(gd)):
            if not os.path.exists(os.path.join(gd, name, "layers", "output", "completed")):
                continue
            m = json.load(open(os.path.join(gd, name, "metadata.json")))
            pr = m["projection"]
            wins.append({"id": f"{group}/{name}", "crs": (pr["crs"], pr["x_resolution"], pr["y_resolution"]),
                         "b": [int(v) for v in m["bounds"]]})
    parts = []
    for crs in sorted({w["crs"] for w in wins}, key=str):
        left = sorted((w for w in wins if w["crs"] == crs), key=lambda w: (w["b"][0], w["b"][1]))
        while left:
            part, box, rest = [], None, []
            for w in left:
                nb = union(box, w["b"])
                if (part and area(nb) * BYTES_PER_PX > limit) or any(meet(w["b"], p["b"]) for p in part):
                    rest.append(w)
                    continue
                part.append(w)
                box = nb
            if area(box) * BYTES_PER_PX > limit:
                raise SystemExit(f"a single window of root {ri} is larger than the part limit")
            parts.append({"windows": [w["id"] for w in part], "grid_gib": area(box) * BYTES_PER_PX / 2**30})
            left = rest
    counts["roots"].append({"root_id": r["root_id"], "windows": len(wins), "parts": [
        {"windows": len(p["windows"]), "grid_gib": round(p["grid_gib"], 2)} for p in parts]})
    for pi, p in enumerate(parts):
        plan.append({"root": r["root_id"], "dir": f"r{ri:02d}_p{pi:02d}", "windows": p["windows"]})
json.dump(plan, open(os.path.join(scores, "parts.json"), "w"), indent=1)
print(json.dumps(counts))
for p in plan:
    cmd = ["oe-inferencex", "from-olmoearth", os.path.join(dest, p["root"]), "--out", os.path.join(scores, p["dir"]),
           "--conditions", "--inputs", "sentinel2_scl", "--window"] + [esc(w) for w in p["windows"]]
    print(f"== {p['dir']}: {len(p['windows'])} windows", flush=True)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)   # its printed command names local paths only
    n = len([f for f in os.listdir(os.path.join(scores, p["dir"])) if f.startswith("scores_")])
    assert n >= 1, f"{p['dir']}: no scores raster written"
counts["parts_read"] = len(plan)
json.dump(counts, open(os.path.join(out, "e98_read.json"), "w"), indent=1)
EOF
mv "$SCORES.partial" "$SCORES"
echo "== $(date -Is) done: $(ls -d "$SCORES"/r*_p* | wc -l) part directories in $SCORES, $OUT/e98_read.json =="
echo "next: E98_MODE=inv sbatch --dependency=afterok:${SLURM_JOB_ID:-<this job>} exp/jobs/e98.sh (E98_SCORES defaults to $SCORES)"
