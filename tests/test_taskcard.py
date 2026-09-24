"""Task-card resolution offline: how a refusal names its cause."""
import pytest
def test_a_rate_limited_listing_is_named_not_reported_as_a_missing_project(monkeypatch):
    """Review of 2026-09-23: under a 403 on the contents API the fallback model.yaml of a variant-only project was
    missing too, and the error asked whether the project name was right."""
    import io
    import urllib.error
    from oe_inferencex import taskcard

    def fake_get(url, timeout=60):
        if url.startswith(taskcard.GH_API):
            raise urllib.error.HTTPError(url, 403, "rate limited", {}, io.BytesIO(b""))
        raise urllib.error.HTTPError(url, 404, "not found", {}, io.BytesIO(b""))
    monkeypatch.setattr(taskcard, "_get", fake_get)
    with pytest.raises(LookupError, match="rate limit"):
        taskcard.project_cards("kenya_lulc_croptype")
