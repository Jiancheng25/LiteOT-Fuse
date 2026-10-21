import unittest

import torch

from model import LiteOTFuse
from model.LiteOTFuse import SpatiallyConstrainedLocalOT


class LiteOTFuseTests(unittest.TestCase):
    def test_sinkhorn_uniform_marginals(self):
        module = SpatiallyConstrainedLocalOT(8, 4, embedding_dim=4, window_size=4)
        cost = torch.rand(3, 16, 16)
        transport = module._sinkhorn(cost)
        expected = torch.full((3, 16), 1.0 / 16.0)
        # Three iterations are intentionally used for real-time inference, so
        # the first marginal is approximate; the last Sinkhorn update makes
        # the second marginal numerically tight.
        self.assertTrue(torch.allclose(transport.sum(-1), expected, atol=2e-2))
        self.assertTrue(torch.allclose(transport.sum(-2), expected, atol=1e-5))

    def test_output_shape_and_gradients(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for the full-resolution smoke test")
        model = LiteOTFuse().cuda().train()
        image = torch.rand(1, 4, 480, 640, device="cuda")
        output = model(image)
        self.assertEqual(tuple(output.shape), (1, 9, 480, 640))
        output.square().mean().backward()
        self.assertIsNotNone(model.local_ot.source_projection.weight.grad)
        self.assertIsNotNone(model.reliability_fusion.gate[-1].weight.grad)


if __name__ == "__main__":
    unittest.main()
