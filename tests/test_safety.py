import numpy as np
from embkit.lib.safety.pii import pii_contains, pii_redact, pii_filter_results
from embkit.lib.safety.repellors import apply_repellors
from embkit.lib.utils import l2n

def test_pii_regex():
    assert pii_contains("Contact: john.doe@example.com")
    assert pii_contains("SSN 123-45-6789")
    assert not pii_contains("No sensitive data")
    red = pii_redact("Mail me at jane@site.org, SSN 123-45-6789, phone +1 (212) 555-1212")
    assert "[REDACTED_EMAIL]" in red and "[REDACTED_SSN]" in red and "[REDACTED_PHONE]" in red


def test_pii_filter_results_masks_snippets():
    rows = [{"id": "1", "snippet": "Call me at 212-555-1212"}, {"id": "2", "snippet": "Clean"}]
    filtered = pii_filter_results(rows)
    assert filtered[0]["snippet"] == "[REDACTED]"
    assert filtered[1]["snippet"] == "Clean"

def test_repellor_penalty():
    D = l2n(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), axis=1)
    scores = np.array([0.9, 0.9], dtype=np.float32)
    B = l2n(np.array([[1.0, 0.0]], dtype=np.float32), axis=1)  # block first direction
    penalized = apply_repellors(scores, D, B, lam=0.5)
    assert penalized[0] < penalized[1]
