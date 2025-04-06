from __future__ import division, absolute_import, print_function
import os, sys, glob, subprocess
if hasattr(os, "add_dll_directory"):
    # Add all the DLL directories manually
    # see:
    # https://docs.python.org/3.8/whatsnew/3.8.html#bpo-36085-whatsnew
    # https://stackoverflow.com/a/60803169/19344391
    dll_directory = os.path.dirname(__file__)
    os.add_dll_directory(dll_directory)

    # The user must install the CUDA Toolkit
    cuda_bin = os.path.join(os.environ["CUDA_PATH"], "bin")
    os.add_dll_directory(cuda_bin)


from pyCT.forward import project
from pyCT.backward import reconstruct
from pyCT.parameter import getParameters
from pyCT.phantom import getPhantom
from pyCT.transformation import getTransformation
from pyCT import geometry


IS_WINDOWS = sys.platform == "win32"
CUDA = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
if CUDA is None:
    try:
        which = "where" if IS_WINDOWS else "which"
        nvcc = subprocess.check_output([which, "nvcc"]).decode().rstrip("\r\n")
        CUDA = os.path.dirname(os.path.dirname(nvcc))
    except subprocess.CalledProcessError:
        if IS_WINDOWS:
            cuda_homes = glob.glob("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*")
            if len(cuda_homes) == 0:
                CUDA = ""
            else:
                CUDA = cuda_homes[0]
        else:
            CUDA = "/usr/local/cuda"
        if not os.path.exists(CUDA):
            CUDA = None