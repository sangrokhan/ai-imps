import unittest
import torch
import yaml
from core.registry import MODEL_REGISTRY, RUNNER_REGISTRY
import implementations
import runners

class TestFrameworkIntegrity(unittest.TestCase):
    def test_model_registration(self):
        """Check if models are registered correctly."""
        self.assertIn("den", MODEL_REGISTRY._registry)
        self.assertIn("apd_resnet18", MODEL_REGISTRY._registry)

    def test_runner_registration(self):
        """Check if runners are registered correctly."""
        self.assertIn("den_runner", RUNNER_REGISTRY._registry)
        self.assertIn("supervised", RUNNER_REGISTRY._registry)

    def test_den_instantiation(self):
        """Check if DEN model can be instantiated without abstract method errors."""
        config = {'hidden_dims': [64, 32]}
        model_cls = MODEL_REGISTRY.get("den")
        model = model_cls(config)
        self.assertIsNotNone(model)
        
        # Test forward pass with dummy data
        x = torch.randn(1, 10)
        model.add_task_layer("0", 10, 2)
        out = model(x, task_id="0")
        self.assertEqual(out.shape, (1, 2))

    def test_apd_instantiation(self):
        """Check if APD model can be instantiated."""
        config = {'num_classes': 10}
        model_cls = MODEL_REGISTRY.get("apd_resnet18")
        model = model_cls(config)
        self.assertIsNotNone(model)
        
        # Test forward pass with dummy data (CIFAR size)
        x = torch.randn(1, 3, 32, 32)
        model.add_task(0)
        out = model(x, task_id=0)
        self.assertEqual(out.shape, (1, 10))

if __name__ == "__main__":
    unittest.main()
