"""Make NVIDIA's pip-installed CUDA DLLs (cuBLAS / cuDNN / NVRTC) loadable.

ctranslate2 ships without bundling CUDA libs and on Windows expects them on the
process DLL search path. The official approach (a system-wide CUDA Toolkit
install) is heavy; the lighter alternative is `pip install nvidia-cublas-cu12
nvidia-cudnn-cu12 nvidia-cuda-nvrtc-cu12`, which lays the DLLs out under
`<site-packages>/nvidia/<lib>/bin/`.

Importing this module at startup is enough to surface those DLLs:
  - prepends each `nvidia/*/bin` dir to %PATH%
  - calls os.add_dll_directory for Python 3.8+ search semantics
"""
from __future__ import annotations

import glob
import os
import sys

_setup_done = False


def setup() -> list[str]:
    global _setup_done
    if _setup_done:
        return []
    nv_dirs: list[str] = []
    for site in [p for p in sys.path if "site-packages" in p]:
        for sub in glob.glob(os.path.join(site, "nvidia", "*", "bin")):
            nv_dirs.append(os.path.abspath(sub))
    if nv_dirs:
        os.environ["PATH"] = os.pathsep.join(nv_dirs + [os.environ.get("PATH", "")])
        for d in nv_dirs:
            try:
                os.add_dll_directory(d)
            except (OSError, AttributeError):
                pass
    _setup_done = True
    return nv_dirs


# Run at import time so anything importing this module before ctranslate2 is set.
setup()
