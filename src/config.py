import os, yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_config():
    with open(os.path.join(ROOT, "configs", "config.yaml")) as f:
        return yaml.safe_load(f)

def path(rel):
    return os.path.join(ROOT, rel)