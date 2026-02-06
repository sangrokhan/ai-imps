import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.registry import RUNNER_REGISTRY, MODEL_REGISTRY
import implementations

print("Registered Runners:", list(RUNNER_REGISTRY._registry.keys()))
print("Registered Models:", list(MODEL_REGISTRY._registry.keys()))

assert "den_runner" in RUNNER_REGISTRY._registry
assert "apd_resnet18" in MODEL_REGISTRY._registry
print("Verification successful!")
