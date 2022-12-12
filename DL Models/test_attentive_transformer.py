import unittest
import torch
import numpy as np
from network import AttentiveTransformer

class TestAttentiveTransformer(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestAttentiveTransformer, self).__init__(*args, **kwargs)
        self.input_dim = np.random.randint(16,256)
        self.output_dim = np.random.randint(16,256)
        self.vbs = np.random.randint(16,256)
        self.momentum = np.random.rand()
        self.test_model = AttentiveTransformer(
            self.input_dim,
            self.output_dim,
            self.vbs,
            self.momentum
        )

    def test_fc(self):
        self.assertEqual(self.test_model.fc.in_features, self.input_dim)
        self.assertEqual(self.test_model.fc.out_features, self.output_dim)
        self.assertEqual(self.test_model.fc.bias, None)
    
    def test_vbs(self):
        self.assertEqual(self.test_model.bn.vbs, self.vbs)
    
    def test_momentum(self):
        self.assertEqual(self.test_model.bn.momentum, self.momentum)
    
    def test_selector(self):
        x = torch.rand(np.random.randint(16,256), np.random.randint(16,256))
        y = self.test_model.selector(x)
        self.assertEqual(y.shape, x.shape)