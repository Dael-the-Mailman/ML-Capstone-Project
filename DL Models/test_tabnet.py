import unittest
import numpy as np
import torch
from network import TabNet

class TestTabNet(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestTabNet, self).__init__(*args, **kwargs)
        param = {
            "input_dim": 2319,
            "output_dim": 1,
            "n_d": 13,
            "n_a": 9,
            "n_steps": 9,
            "gamma": 1.3077154313342185,
            "cat_idxs": [],
            "cat_dims": [],
            "cat_emb_dim": 1,
            "n_independent": 2,
            "n_shared": 2,
            "epsilon": 1e-15,
            "vbs": 128,
            "momentum": 0.03072144877400083
        }
        self.test_model = TabNet(**param)
    
    def test_forward(self):
        x = torch.randn(1024, 2319)
        y, loss= self.test_model(x)
        self.assertEqual(y.shape, (1024, 1))
        self.assertEqual(loss.shape, ())