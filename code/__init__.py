"""Week 1 W and L1 verification pipeline package.

The package name is intentionally ``code`` for this project. Some third-party
libraries import Python's standard-library ``code`` module during initialization;
expose those stdlib attributes here so ``python -m code.w.build`` remains
compatible with those imports.
"""

from __future__ import annotations

import importlib.util
import sysconfig
from pathlib import Path

_stdlib_code_path = Path(sysconfig.get_path("stdlib")) / "code.py"
_stdlib_spec = importlib.util.spec_from_file_location("_hmgraph_stdlib_code", _stdlib_code_path)
if _stdlib_spec and _stdlib_spec.loader:
    _stdlib_code = importlib.util.module_from_spec(_stdlib_spec)
    _stdlib_spec.loader.exec_module(_stdlib_code)
    InteractiveInterpreter = _stdlib_code.InteractiveInterpreter
    InteractiveConsole = _stdlib_code.InteractiveConsole
    compile_command = _stdlib_code.compile_command
