class Registry:
    """A simple registry to map names to classes."""
    def __init__(self):
        self._registry = {}

    def register(self, name):
        def wrapper(cls):
            self._registry[name] = cls
            return cls
        return wrapper

    def get(self, name):
        if name not in self._registry:
            raise KeyError(f"'{name}' not found in registry.")
        return self._registry[name]

# Global registries
MODEL_REGISTRY = Registry()
RUNNER_REGISTRY = Registry()
DATASET_REGISTRY = Registry()
