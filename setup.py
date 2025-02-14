import os
from setuptools import setup, Extension

__cxx_path__ = "pyxai/sources/solvers/GREEDY/src/"
__cxx_files__ = [__cxx_path__+f for f in os.listdir(__cxx_path__) if f.endswith(".cc")]+[__cxx_path__+"bcp/"+f for f in os.listdir(__cxx_path__+"bcp/") if f.endswith(".cc")]
__cxx_headers__ = [__cxx_path__+f for f in os.listdir(__cxx_path__) if f.endswith(".h")]+[__cxx_path__+"bcp/"+f for f in os.listdir(__cxx_path__+"bcp/") if f.endswith(".h")]

print("__cxx_path__:", __cxx_path__)
print("__cxx_files__:", __cxx_files__)
print("__cxx_headers__:", __cxx_headers__)

from setuptools import setup, Extension

setup(ext_modules=[Extension(
          "c_explainer",
          __cxx_files__,
          language="c++",
          extra_compile_args=["-std=c++11"]
      )],
      headers=__cxx_headers__,
      )
