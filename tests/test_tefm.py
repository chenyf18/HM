"""TEFM unit tests (HM-TEFM-038)."""
import os, sys, unittest
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from libs.modeling.tefm import TEFM, EvidenceHead


class TestTEFM(unittest.TestCase):
    def test_identity_init(self):
        """Zero-init conv2 => TEFM output == input (bit-equal)."""
        t = TEFM(dim=16).eval()
        x = (torch.randn(2, 16, 20), torch.randn(2, 16, 10))
        m = (torch.ones(2, 1, 20, dtype=torch.bool),
             torch.ones(2, 1, 10, dtype=torch.bool))
        out = t(x, m)
        for a, b in zip(out, x):
            self.assertTrue(torch.equal(a, b))

    def test_residual_changes_after_perturb(self):
        t = TEFM(dim=16).eval()
        with torch.no_grad():
            t.conv2.conv.weight.normal_(0, 0.1)
        x = (torch.randn(1, 16, 20),)
        m = (torch.ones(1, 1, 20, dtype=torch.bool),)
        out = t(x, m)
        self.assertFalse(torch.equal(out[0], x[0]))

    def test_shapes_and_mask(self):
        t = TEFM(dim=16)
        x = (torch.randn(2, 16, 20),)
        m = (torch.ones(2, 1, 20, dtype=torch.bool),)
        m[0][:, :, -5:] = False
        out = t(x, m)
        self.assertEqual(out[0].shape, x[0].shape)

    def test_evidence_head(self):
        h = EvidenceHead(dim=16)
        x = (torch.randn(2, 16, 20),)
        m = (torch.ones(2, 1, 20, dtype=torch.bool),)
        out = h(x, m)
        self.assertEqual(out[0].shape, (2, 20))

    def test_param_count(self):
        t = TEFM(dim=384); h = EvidenceHead(dim=384)
        total = sum(p.numel() for p in t.parameters()) + \
                sum(p.numel() for p in h.parameters())
        self.assertLess(total, 2_000_000,
                        f"TEFM+EvidenceHead must be <2M, got {total}")


if __name__ == '__main__':
    unittest.main()
