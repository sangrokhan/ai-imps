import os
import importlib
import pkgutil

# Automatically import all modules under implementations/
# This ensures that all classes decorated with @MODEL_REGISTRY.register 
# or @RUNNER_REGISTRY.register are properly loaded.

def import_all_implementations():
    # Get the directory of this __init__.py file
    pkg_dir = os.path.dirname(__file__)
    
    # Iterate through all subdirectories and files
    for root, dirs, files in os.walk(pkg_dir):
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                # Convert file path to module path (e.g., implementations.y2015_dqn.model)
                rel_path = os.path.relpath(os.path.join(root, file), os.path.dirname(pkg_dir))
                module_path = rel_path.replace(os.sep, '.')[:-3]
                
                try:
                    importlib.import_module(module_path)
                except Exception as e:
                    # Skip modules that fail to import (e.g., missing dependencies)
                    pass

import_all_implementations()
