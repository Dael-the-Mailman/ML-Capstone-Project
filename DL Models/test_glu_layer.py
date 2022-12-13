import unittest
import numpy as np
import torch
from network import GLU_Layer

class TestGLULayer(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestGLULayer,self).__init__(*args, **kwargs)
        self.input_dim = np.random.randint(16, 256)
        self.output_dim = np.random.randint(16, 256)
        self.test_model = GLU_Layer(
            self.input_dim,
            self.output_dim
        )
    
    def test_forward(self):
        x = torch.rand(128, self.input_dim)
        y = self.test_model(x)
        self.assertEqual(y.shape, (128, self.output_dim))