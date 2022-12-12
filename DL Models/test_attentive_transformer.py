import unittest
import torch
import numpy as np
from network import AttentiveTransformer

class TestAttentiveTransformer(unittest.TestCase):
    def __init__(self):
        self.input_dim = 128
        self.output_dim = 256
        self.test_model = AttentiveTransformer()

    def