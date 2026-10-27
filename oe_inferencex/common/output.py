"""What every command writes and checks: the written_by stamp, output paths, numeric arguments, percentages."""
import json
import os

import numpy as np


def _stamp(result, command):
    """`result` with `written_by` as its last key: the package, its version and the command that wrote the file.
    Until 1.14.1 no result file named the release that wrote it, while releases change numbers on purpose (1.14.0
    capped the producer's accuracy point), so a file could not be traced to its code (the OlmoEarth Agent stamps
    its tool outputs; the scan of 23 October 2026). A file another release wrote is read as before."""
    from oe_inferencex import __version__
    result.pop("written_by", None)                              # last, also in a dict copied from a sidecar
    result["written_by"] = {"package": "olmoearth-inferencex", "version": __version__, "command": command}
    return result


# ----------------------------------------------------------------------------- assess
def _pct(x, nd=2):
    """A percentage, or the word undefined. `summary` turns NaN into None, and `None or 0` prints 0.00%, which reads as
    "they never disagree" when the truth is "this could not be computed". This package exists to keep those apart."""
    return "undefined" if x is None else f"{100 * float(x):.{nd}f}%"


def _check_patch(patch, command):
    """--patch 0 was a ZeroDivisionError traceback in assess and sample, and a negative one numpy's "can only specify
    one unknown dimension", where compare refused it by name (2026-10-06)."""
    if patch < 1:
        raise SystemExit(f"{command}: --patch must be at least 1, got {patch}")


def _check_out_dir(out, command):
    """assess and compare write into the directory --out names; an existing file there was a FileExistsError traceback
    from os.makedirs, after the whole run (2026-10-06)."""
    if os.path.exists(out) and not os.path.isdir(out):
        raise SystemExit(f"{command}: --out {out} is a file; name the directory to write into")


def _check_out_file(out, command, example="result.json"):
    """estimate, certify and decide write the file --out names: a directory there, or a missing parent directory, was a
    traceback, and certify had already left its mask beside the directory (2026-10-06). The parent is made, as sample
    makes it. `example` names the kind of file written (boundary writes a band, not a JSON)."""
    if os.path.isdir(out):
        kind = "JSON" if example.endswith(".json") else "file"
        raise SystemExit(f"{command}: --out {out} is a directory; name the {kind} to write, for example "
                         f"{os.path.join(out, example)}")
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)


def _num_arg(value):
    """A number as a printed command must give it after a space: positional and exact, since argparse reads
    '-3.4e+38' as an option (the review of 1.17.2)."""
    v = float(value)
    return np.format_float_positional(v, trim="-") if np.isfinite(v) else repr(v)


def _nodata_arg(value):
    """--nodata as one token with the exact value, as a printed command must give it: argparse takes '-3.4e+38' or
    '-inf' after a space for an option, and :g turned a fill of 1234567 into 1.23457e+06, another value (the review of
    1.17.2)."""
    v = _json_nodata(value)
    return f"--nodata={v if isinstance(v, str) else repr(float(v))}"


def _same_file(a, b):
    """Whether two paths name one file: by the file itself where both exist (another letter case on a case-insensitive
    disk, a symlink), else by the resolved path (the bug hunt of 25 October 2026: `estimate --out S.json` replaced the
    sidecar s.json on macOS)."""
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.realpath(a) == os.path.realpath(b)


def _refuse_overwriting_inputs(command, outputs, inputs):
    """Refuse an output path that is one of the inputs. `assess --condition condition.npy --out .` replaced the user's
    pixel layer with the window grid, exit 0, and the next run was refused on its shape; `estimate s.csv --out s.json`
    replaced the sample's design, and every later estimate died on a KeyError (2026-10-06)."""
    for o in outputs:
        for p in inputs:
            if p and o and isinstance(p, str) and _same_file(o, p):
                raise SystemExit(f"{command}: writing {o} would overwrite the input {p}; name another --out")


def _json_nodata(value):
    """A no-data value as strict JSON holds it: NaN is written "nan" (json.dump wrote the bare token NaN, which a
    strict parser refuses, 2026-10-06); _map_windows reads it back with float()."""
    if value is None:
        return None
    v = float(value)
    # NaN and the infinities as text, which strict JSON (and a certificate's reader) accepts; float() reads them back
    return "nan" if np.isnan(v) else ("inf" if v > 0 else "-inf") if np.isinf(v) else value


def _where(path):
    """An input path as a result or sidecar records it: absolute for a file on disk, as given for a path GDAL reads
    itself (/vsizip/, /vsicurl/, s3://). os.path.abspath broke those: '/vsizip//abs/m.zip/x.tif' lost its '//', and
    's3://b/x.tif' became '<cwd>/s3:/b/x.tif', so a sample drawn on them could not be certified from its sidecar (the
    bug hunt of 2026-10-25)."""
    p = str(path)
    return p if p.startswith("/vsi") or "://" in p else os.path.abspath(p)


def _readable_path(path):
    """A file on disk, or a path GDAL reads itself (/vsizip/, /vsicurl/, s3://): read_raster then says if it cannot."""
    p = str(path)
    return os.path.exists(p) or p.startswith("/vsi") or "://" in p
