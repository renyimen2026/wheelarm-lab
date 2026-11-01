import json
import os
from pathlib import Path
import subprocess
import sys

from ament_index_python.packages import get_package_prefix
import pytest


@pytest.mark.parametrize("enabled", [False, True])
def test_only_moveit_libraries_remain_mapped_after_dlclose(tmp_path, enabled):
    fixture = Path(__file__).with_name("lifetime_fixture.cpp")
    names = ["libmoveit_fixture.so", "libmoveit_versioned_fixture.so.1", "libunrelated_fixture.so"]
    libraries = [tmp_path / name for name in names]
    for library in libraries:
        subprocess.run(["c++", "-shared", "-fPIC", str(fixture), "-o", str(library)], check=True)
    environment = os.environ.copy()
    if enabled:
        guard = Path(get_package_prefix("wheel_arm_moveit_config")) / "lib/libwheel_arm_moveit_plugin_lifetime.so"
        environment["LD_PRELOAD"] = str(guard) + ":" + environment.get("LD_PRELOAD", "")
    script = """
import ctypes, _ctypes, json, pathlib, sys
for path in sys.argv[1:]:
    library = ctypes.CDLL(path)
    assert library.lifetime_fixture() == 42
    _ctypes.dlclose(library._handle)
maps = pathlib.Path('/proc/self/maps').read_text()
print(json.dumps([path in maps for path in sys.argv[1:]]))
"""
    result = subprocess.run([sys.executable, "-c", script, *map(str, libraries)],
                            env=environment, text=True, capture_output=True, check=True)
    assert json.loads(result.stdout) == [enabled, enabled, False]
