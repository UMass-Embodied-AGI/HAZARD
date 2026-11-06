"""Run an old (pre-pip-package) HAZARD checkout's run_experiments.py unmodified,
with the same runtime fixes run_rule.py applies:

  * launch a TDW build for every controller created with check_version=False
  * big thread stack + high recursion limit (tdw ReplicantBase deepcopy)
  * force use_local_resources=False (old challenge.py hard-codes True, which
    points asset URLs at <cwd>/data/assets, a local cache not in the repo)

Usage:
    python run_old.py --repo ../HAZARD_nov23 --build ~/tdw_builds/v1.12.8/TDW/TDW.exe -- \
        --env_name fire --agent_name rule --use_gt --data_dir data/room_setup_fire/test_set \
        --max_test_episode 25 --port 10860 --output_dir outputs_old/fire
Everything after "--" is passed to the old run_experiments.py.
"""
import argparse
import os
import runpy
import subprocess
import sys
import threading
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--build", required=True)
own, rest = sys.argv[1:], []
if "--" in own:
    i = own.index("--")
    own, rest = own[:i], own[i + 1:]
args = ap.parse_args(own)

repo = Path(args.repo).resolve()
build = Path(args.build).expanduser().resolve()
assert (repo / "run_experiments.py").exists(), repo
assert build.exists(), build
os.chdir(repo)
sys.path.insert(0, str(repo))

# 1. launch a build per controller
from tdw.controller import Controller

_builds = []
_orig_init = Controller.__init__


def _init_and_launch(self, port=1071, check_version=True, launch_build=True):
    if launch_build and not check_version:
        _builds.append(subprocess.Popen([str(build), f"-port={port}"]))
        launch_build = False
    _orig_init(self, port=port, check_version=check_version, launch_build=launch_build)


Controller.__init__ = _init_and_launch

# 1a. 9c003f8 (Sept 22) uses Dict in the *agent_controller.py signatures without importing it
#     (fixed in 5a50c2d by adding it to the typing import). Expose the typing names as builtins.
import builtins
import typing

for _n in ("Dict", "List", "Optional", "Union", "Any", "Tuple"):
    if not hasattr(builtins, _n):
        setattr(builtins, _n, getattr(typing, _n))

# 1b. utils/vision.py at some old commits has a syntax error in its __main__ test block.
#     It only provides the perception Detector (unused with --use_gt), so stub it if it fails.
import importlib
import types

try:
    importlib.import_module("utils.vision")
except SyntaxError as e:
    import utils as _utils

    _stub = types.ModuleType("utils.vision")

    class Detector:  # noqa: D401
        def __init__(self, *a, **k):
            raise RuntimeError("utils.vision stubbed by run_old.py (perception mode unavailable)")

    _stub.Detector = Detector
    sys.modules["utils.vision"] = _stub
    _utils.vision = _stub
    print(f"[run_old] stubbed utils.vision ({e.msg} at line {e.lineno})", flush=True)

# 2. no local asset cache

for mod, cls in [("envs.fire.fireagent_controller", "FireAgentController"),
                 ("envs.flood.floodagent_controller", "FloodAgentController"),
                 ("envs.wind.windagent_controller", "WindAgentController")]:
    try:
        C = getattr(importlib.import_module(mod), cls)
    except Exception as e:  # pragma: no cover
        print(f"[run_old] could not patch {mod}.{cls}: {e!r}", flush=True)
        continue

    def _make(orig):
        def __init__(self, *a, **k):
            k["use_local_resources"] = False
            return orig(self, *a, **k)
        return __init__

    C.__init__ = _make(C.__init__)

import tdw.version
print(f"[run_old] repo {repo.name}, tdw {tdw.version.__version__}, build {build}, args {rest}", flush=True)

# 3. recursion / stack
sys.setrecursionlimit(100_000)
threading.stack_size(255 * 1024 * 1024)
err = []


def _run():
    try:
        sys.argv = ["run_experiments.py"] + rest
        runpy.run_path(str(repo / "run_experiments.py"), run_name="__main__")
    except BaseException as e:
        err.append(e)


try:
    t = threading.Thread(target=_run)
    t.start()
    t.join()
    if err:
        raise err[0]
finally:
    for b in _builds:
        if b.poll() is None:
            b.terminate()
