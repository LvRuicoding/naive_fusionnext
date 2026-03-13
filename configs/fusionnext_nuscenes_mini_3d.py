import os


_CURRENT_DIR = os.path.dirname(__file__)
_PLUGIN_CONFIG = os.path.join(
    os.path.dirname(_CURRENT_DIR),
    "projects",
    "FusionNeXt",
    "configs",
    "fusionnext_nuscenes_mini_3d.py",
)

with open(_PLUGIN_CONFIG, "r", encoding="utf-8") as f:
    exec(compile(f.read(), _PLUGIN_CONFIG, "exec"))
