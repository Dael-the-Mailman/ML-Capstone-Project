import unittest
import numpy as np
import torch
from network import GLU_Block

class TestGLUBlock(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestGLUBlock,self).__init__(*args, **kwargs)
        self.input_dim = np.random.randint(16,256)
        self.output_dim = np.random.randint(16, 256)
        self.n_glu = np.random.randint(1,10)

        self.test_model = GLU_Block(
            self.input_dim,
            self.output_dim,
            self.n_glu
        )
    
    def test_n_glu(self):
        self.assertEqual(len(self.test_model.glu_layers), self.n_glu)
