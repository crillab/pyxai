from distutils.core import setup, Extension


def main():
    setup(name="c_explainer",
          version="1.0.0",
          description="Python interface for the c_explainer C library function",
          author="<your name>",
          author_email="your_email@gmail.com",
          ext_modules=[Extension(
              "c_explainer",
              ["sources/solvers/GREEDY/src/bt_wrapper.cc", "sources/solvers/GREEDY/src/Explainer.cc", "sources/solvers/GREEDY/src/Tree.cc",
               "sources/solvers/GREEDY/src/Node.cc"],
              language="c++")])


if __name__ == "__main__":
    main()
