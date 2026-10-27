"""The input-condition layer on the command line: --condition and --condition-names, read and checked."""
import numpy as np

from oe_inferencex.common.raster import _same_grid, read_raster


# ----------------------------------------------------------------------------- the input-condition layer
def _condition_names(items, command):
    """{value: name} from `--condition-names 0=clear 1=cloudy`, or None. Unique, non-empty names and the reserved
    "unrecorded" are the API's to check."""
    if not items:
        return None
    names = {}
    for item in items:
        key, sep, name = str(item).partition("=")
        try:
            value = int(key.strip())
        except ValueError:
            value = None
        if not sep or value is None:
            raise SystemExit(f"{command}: --condition-names takes value=name pairs, such as 0=clear 1=cloudy; got {item!r}")
        if value in names:
            raise SystemExit(f"{command}: --condition-names names the value {value} twice")
        if len(name) > 64 or any(not ch.isprintable() or ch in '"\\' for ch in name):
            # a name is printed in every statement made about its condition, a certificate's among them: control
            # characters could rewrite what a terminal shows, and a quote could close the name and write a coverage
            # of its own (reviews of 24 October 2026)
            raise SystemExit(f"{command}: a condition name is at most 64 printable characters, without quotes or "
                             f"backslashes, got {name!r}")
        names[value] = name
    return names


def _read_condition(path, shape, geo, command):
    """The per-pixel input condition layer, checked to lie on the map's grid, as a masked array: the raster's own
    no-data is masked, and NaN and negative values record no condition either (pool_condition). The layer must be
    one band; its values are pool_condition's to check."""
    a, valid, geo_c = read_raster(path, None)
    if a.ndim == 3 and a.shape[0] == 1:                   # a single-band .npy saved as (1, H, W)
        a = a[0]
    if a.ndim != 2:
        raise SystemExit(f"{command}: {path} has shape {a.shape}; the condition layer is one band, one value per pixel "
                         "of the map")
    if a.shape != tuple(shape):
        raise SystemExit(f"{command}: {path} has shape {a.shape}; the map is {tuple(shape)}")
    _same_grid(geo, geo_c, f"{command}: {path}")        # every other refusal of the layer names the command
    return np.ma.masked_array(a, mask=~valid)


def _condition_args(args, command):
    """(path of the condition raster or None, {value: name} or None), read with getattr: demo.py and the tests build
    the argument Namespace by hand, without these attributes."""
    path = getattr(args, "condition", None)
    names = _condition_names(getattr(args, "condition_names", None), command)
    if names and not path:
        raise SystemExit(f"{command}: --condition-names names the values of a --condition raster, and none was given")
    return path, names
