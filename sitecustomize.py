import os
import sys
import importlib
from pathlib import Path

# ----------------------------
# 1) Select implementation
# ----------------------------
mode = os.getenv("LEROBOT_IMPL", "train").strip().lower()
workspace = Path(
    os.getenv(
        "LEROBOT_WORKSPACE",
        str(Path(__file__).parent / "lerobot_dataset"),
    )
)

impl_map = {
    "fast_convert": workspace / "lerobot_dataset_fast_convert.py",
    "train": workspace / "lerobot_dataset.py",
}

impl = impl_map.get(mode)
if impl is None:
    raise RuntimeError(f"Unknown LEROBOT_IMPL={mode}. Use one of {list(impl_map.keys())}")
if not impl.exists():
    raise FileNotFoundError(f"LeRobot impl file not found: {impl}")

TARGET = "lerobot.common.datasets.lerobot_dataset"

# ----------------------------
# 2) Import original module
# ----------------------------
orig = importlib.import_module(TARGET)

# ----------------------------
# 3) Compatibility shims
# ----------------------------

# 3.1 LEROBOT_HOME (old scripts rely on it)
if not hasattr(orig, "LEROBOT_HOME"):
    home = os.getenv("HF_LEROBOT_HOME") or os.getenv("LEROBOT_HOME")
    # home = os.getenv("LEROBOT_HOME")
    if home is None:
        home = str(Path.home() / ".cache" / "lerobot")
    orig.LEROBOT_HOME = Path(home)

# 3.2 get_hub_safe_version (older fast_* expect it)
try:
    import lerobot.common.datasets.utils as _ds_utils
    if not hasattr(_ds_utils, "get_hub_safe_version"):
        def get_hub_safe_version(repo_id: str, revision: str | None = None, **kwargs):
            return revision or "main"
        _ds_utils.get_hub_safe_version = get_hub_safe_version
except Exception:
    pass

# ----------------------------
# 4) ★ Root fix: exec impl into orig
# ----------------------------
try:
    code = compile(impl.read_text(encoding="utf-8"), str(impl), "exec")
    exec(code, orig.__dict__)
    sys.modules[TARGET] = orig
    if os.environ.get("LEROBOT_PATCH_PRINTED", "0") != "1":
        print(f"[LeRobot] patched {TARGET} with mode={mode} via exec from {impl}")
        os.environ["LEROBOT_PATCH_PRINTED"] = "1"
except Exception as e:
    sys.modules[TARGET] = orig
    if os.environ.get("LEROBOT_PATCH_PRINTED", "0") != "1":
        print(f"[LeRobot] patch failed for mode={mode} from {impl}: {e}")
        print(f"[LeRobot] fallback: using original {TARGET} with compatibility shims")
        os.environ["LEROBOT_PATCH_PRINTED"] = "1"
