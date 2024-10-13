import os
from distutils.core import Extension
from setuptools import setup, find_packages

__version__ = open(os.path.join(os.path.dirname(__file__), 'pyxai/version.txt'), encoding='utf-8').read()
__cxx_path__ = "pyxai/sources/solvers/GREEDY/src/"
__cxx_files__ = [__cxx_path__+f for f in os.listdir(__cxx_path__) if f.endswith(".cc")]+[__cxx_path__+"bcp/"+f for f in os.listdir(__cxx_path__+"bcp/") if f.endswith(".cc")]
__cxx_headers__ = [__cxx_path__+f for f in os.listdir(__cxx_path__) if f.endswith(".h")]+[__cxx_path__+"bcp/"+f for f in os.listdir(__cxx_path__+"bcp/") if f.endswith(".h")]

print("__version__", __version__)
print("__cxx_path__:", __cxx_path__)
print("__cxx_files__:", __cxx_files__)
print("__cxx_headers__:", __cxx_headers__)

setup(name='pyxai',
      version=__version__,
      python_requires='>=3.6',
      project_urls={
            'Documentation': 'http://www.cril.univ-artois.fr/pyxai/',
            'Installation': 'http://www.cril.univ-artois.fr/pyxai/documentation/installation/',
            'Git': 'https://github.com/crillab/pyxai'
        },
      author='Gilles Audemard, Steve Bellart, Louenas Bounia, Jean-Marie Lagniez, Pierre Marquis, Nicolas Szczepanski:',
      author_email='audemard@cril.fr, bellart@cril.fr, bounia@cril.fr, lagniez@cril.fr, marquis@cril.fr, szczepanski@cril.fr',
      maintainer='Gilles Audemard, Nicolas Szczepanski',
      maintainer_email='audemard@cril.fr, szczepanski@cril.fr',
      keywords='XAI AI ML explainable learning',
      classifiers=['Topic :: Scientific/Engineering :: Artificial Intelligence', 'Topic :: Education'],
      packages=find_packages(),  # exclude=["problems/g7_todo/"]),
      package_dir={'pyxai': 'pyxai'},
      install_requires=['lxml', 'numpy', 'wheel', 'pandas', 'termcolor', 'shap', 'wordfreq', 'python-sat[pblib,aiger]', 'xgboost==1.7.3', 'pycsp3', 'matplotlib', 'dill', 'lightgbm', 'docplex', 'ortools', 'packaging'],
      extras_require={
        "gui": ['pyqt6'],
      },
      include_package_data=True,
      description='Explaining Machine Learning Classifiers in Python',
      long_description=open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='utf-8').read(),
      long_description_content_type='text/markdown',
      license='MIT',
      ext_modules=[Extension(
          "c_explainer",
          __cxx_files__,
          language="c++",
          extra_compile_args=["-std=c++11"]
      )],
      headers=__cxx_headers__,
      platforms='LINUX')
