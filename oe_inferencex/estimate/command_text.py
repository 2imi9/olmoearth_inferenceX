"""Words the sample and estimate commands share about an audit's confident zone."""
from oe_inferencex import estimate as est


def _zone_words(zone):
    """How the confident zone was set, as printed after its share of the map: the top third, the top S, or nothing
    for a cut given as a confidence. Windows tied at the cut are in the zone, so on a banded or saturated confidence
    it can hold far more than the share asked for; the words then say how many the ties added (the review of 13
    October 2026: a band with 80% of its pixels at 100 printed a zone of 81.3% of the map as "the top third")."""
    if zone["rule"] != "share":
        return ""
    top = "the top third" if zone["share"] == est.AUDIT_SHARE else f"the top {100 * zone['share']:g}%"
    extra = int(zone["n_zone"]) - int(zone["n_top"])
    if extra > 0:
        return (f" ({top} by rank, {zone['n_top']} window{'' if zone['n_top'] == 1 else 's'}, and {extra} more tied "
                "at the cut)")
    return f" ({top})"
