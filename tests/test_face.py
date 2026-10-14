"""Model-free face recognition: descriptor determinism, exemplar matching, the
'no ceiling' property (more exemplars -> better match), and zip round-trip.

Uses synthetic structured textures as stand-in 'identities' so the test needs no
real photo and no Haar detection — it exercises the recognition CORE
(descriptor + nearest-neighbor + zip storage), which is the model-free claim.
"""
import numpy as np

from irix import face


def _texture(kind: str, noise: float = 0.0, shift: int = 0, seed: int = 0) -> np.ndarray:
    """A deterministic 128x128 gray 'face': distinct local gradient structure per
    kind, so LBP histograms differ between kinds and stay close within a kind."""
    n = face._FACE_SIZE
    yy, xx = np.mgrid[0:n, 0:n]
    if kind == "h":
        base = (np.sin(yy / 4.0) * 90 + 128)
    elif kind == "v":
        base = (np.sin(xx / 4.0) * 90 + 128)
    elif kind == "diag":
        base = (np.sin((xx + yy) / 4.0) * 90 + 128)
    else:
        base = (np.sin((xx - yy) / 6.0) * 90 + 128)
    if noise:
        rng = np.random.default_rng(seed)
        base = base + rng.normal(0, noise * 255, base.shape)
    base = np.clip(base + shift, 0, 255).astype(np.uint8)
    return base


def test_descriptor_is_deterministic():
    a = _texture("h")
    assert np.array_equal(face.descriptor(a), face.descriptor(a))
    assert face._chi2(face.descriptor(a), face.descriptor(a)) == 0.0


def test_within_class_closer_than_between_class():
    base = face.descriptor(_texture("h"))
    same = face.descriptor(_texture("h", noise=0.05, shift=10, seed=1))
    other = face.descriptor(_texture("v"))
    d_same = face._chi2(base, same)
    d_other = face._chi2(base, other)
    assert d_same < d_other, (d_same, d_other)


def test_gallery_remember_and_recognize(tmp_path):
    g = face.Gallery(str(tmp_path / "people.irix"))
    for s in range(3):
        g.remember(_texture("h", noise=0.04, shift=s * 5, seed=s), "Flouk")
        g.remember(_texture("v", noise=0.04, shift=s * 5, seed=s + 10), "Lee")
    # a fresh exemplar of Flouk must match Flouk, not Lee
    q = _texture("h", noise=0.05, shift=7, seed=99)
    r = g.match(q, threshold=2.0)
    assert r["nearest"] == "Flouk", r
    # an unseen identity is farther than a seen one (discrimination, no hardcoded thresh)
    seen = g.match(_texture("h", noise=0.05, seed=7))["dist"]
    unseen = g.match(_texture("diag"))["dist"]
    assert seen < unseen, (seen, unseen)


def test_no_ceiling_more_exemplars_improve_match(tmp_path):
    g = face.Gallery(str(tmp_path / "grow.irix"))
    q = _texture("h", noise=0.05, shift=3, seed=42)
    g.remember(_texture("h", noise=0.04, seed=0), "X")
    d1 = g.match(q, threshold=2.0)["dist"]
    for s in range(1, 8):                      # see X many more times
        g.remember(_texture("h", noise=0.04, shift=s, seed=s), "X")
    d2 = g.match(q, threshold=2.0)["dist"]
    assert d2 <= d1, (d1, d2)                  # memory grows -> match only improves


def test_zip_persists_across_reopen(tmp_path):
    p = str(tmp_path / "persist.irix")
    g = face.Gallery(p)
    g.remember(_texture("h", noise=0.04), "Flouk")
    g.remember(_texture("v", noise=0.04), "Lee")
    reopened = face.Gallery(p)                 # fresh object, reads the zip
    st = reopened.stats()
    assert st["exemplars"] == 2
    assert set(st["by_name"]) == {"Flouk", "Lee"}
