
# PyXAI - Python eXplainable AI

![PyXAI](http://www.cril.univ-artois.fr/pyxai/assets/figures/pyxai.png)

Documentation: [http://www.cril.univ-artois.fr/pyxai/](http://www.cril.univ-artois.fr/pyxai/)

Git: [https://github.com/crillab/pyxai](https://github.com/crillab/pyxai)

PyXAI is a <a href="https://www.python.org/">Python</a> library (version 3.6 or later) allowing to bring explanations of various forms from classifiers resulting of machine learning techniques  (Decision Tree, Random Forest, Boosted Tree).

More precisely, several types of explanations for the classification task of a given instance X can be computed:

<ul>
  <li>Abductive explanations for X are intended to explain why X has been classified in the way it has been classified by the ML model (thus, addressing the “Why?” question).</li>
  <li>Contrastive explanations for X is to explain why X has not been classified by the ML model as the user expected it (thus, addressing the “Why not?” question).</li>
</ul>

<p>
In addition to finding explanations, PyXAI also contains methods that perform operations (production, saving, loading) on models and instances. 
Currently, these helping methods are available using two ML libraries:
</p>
<ul>
  <li><a href="https://scikit-learn.org/stable/">Scikit-learn</a>: a software machine learning library</li> 
  <li><a href="https://xgboost.readthedocs.io/en/stable/">XGBoost</a>: an optimized distributed gradient boosting library</li>
</ul> 

<p>
Note that it is quite possible to find explanations of models coming from other libraries.
</p>

<p>
As an illustration, below, you can find an example of use:
</p>

```python
from pyxai import Learning, Explainer, Tools

learner = Learning.Scikitlearn("../dataset/iris.csv")
model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.DT)
instance, prediction = learner.get_instances(model, n=1, correct=True, predictions=[0])

explainer = Explainer.initialize(model, instance)
print("instance:", instance)
print("implicant:", explainer.binary_representation)

sufficient_reason = explainer.sufficient_reason(n=1)
print("sufficient_reason:", sufficient_reason)
print("to_features:", explainer.to_features(sufficient_reason))

instance, prediction = learner.get_instances(model, n=1, correct=False)
explainer.set_instance(instance)
contrastive_reason = explainer.contrastive_reason()
print("contrastive reason", contrastive_reason)
print("to_features:", explainer.to_features(contrastive_reason))
```

# Installation

## Installation from PyPi

The Python Package Index (PyPi) is the easiest way of installing PyXAI.

Note that you need first Python 3 (version 3.6, or later) to be installed.
You can do it, for example, from [python.org](https://www.python.org/downloads/).

See the Virtual Environment section if you want to install PyXAI inside a Python virtual environment.

### Installing PyXAI (Linux)

Check if 'python3-pip' is installed. If it is not the case, execute:

```console
sudo apt install python3-pip
```

Or check if you have the last installed version:

```
python3 -m pip install --upgrade pip
```

Then, install PyXAI with the command 'pip3':

```console
python3 -m pip install pyxai
```

### Installing PyXAI (Mac OS)

PyXAI is currently partially compatible with Mac OS but without PyPi (see this section). You can also use a docker container that runs a jupyter notebook with all features.

### Installing PyXAI (Windows)

PyXAI is currently not compatible with Windows (work in progress). Instead, you can use a docker container with PyXAI inside.

### Updating the Version of PyXAI (for PyPi)

For updating your version of PyXAI, simply execute:

For linux/Mac:

```console
python3 -m pip install -U pyxai
```

## Installation (alternative) by Cloning from GitHub

An alternative to PyPi is to clone the code from GitHub.

Here is an illustration for linux. We assume that Python 3 is installed, and consequently 'pip3' is also installed.
In a console, type:

```console
git clone https://gitlab.univ-artois.fr/expekctation/software/PyLearningExplanation.git
```

You may need to update the environment variable 'PYTHONPATH', by typing for example:

```console
export PYTHONPATH="${PYTHONPATH}:${PWD}/.."
```

Get the last version of pip:

```console
python3 -m pip install --upgrade pip
```

There are a few packages that PyXAI depends on that must be installed:

```console
python3 -m pip install numpy
python3 -m pip install wheel
python3 -m pip install pandas
python3 -m pip install termcolor
python3 -m pip install shap
python3 -m pip install wordfreq
python3 -m pip install python-sat[pblib,aiger]
python3 -m pip install xgboost
python3 -m pip install lxml
python3 -m pip install pycsp3
python3 -m pip install matplotlib
```

To compile the c++ code (python C extensions):

```console
python3 setup.py install --user
```

Of course, for this step, you need a C++ compiler.

Unfortunately, the compiled C extensions are not take into account in a virtual environment, therefore you must type
(we consider here that the virtual environment is in the 'env' directory and you are in the 'PyXAI' directory):

```console
cp build/lib.linux-x86_64-3.6/c_explainer.cpython-36m-x86_64-linux-gnu.so env/lib/python3.6/site-packages/.
```

This last command depend of your python version (here: 3.6).

Finally, you can test an example:

```console
python3 examples/DT/BuilderOrchids.py 
```

## Using a Docker Image

A docker container is available on Git ([https://github.com/crillab/pyxai](https://github.com/crillab/pyxai)). 
It launches a Jupyter notebook that supports all PyXAI features.

Below is the code line to build the container:
```
docker build -t pyxai .
```

And run the container (we consider that the working directory is the current one):
```
docker run -it -p 8888:8888 -v $PWD:/data pyxai```
```

## Virtual Environment

Create and activate a new virtual environment:

```console
sudo apt-get install python3.6-venv
python3.6 -m venv env
source env/bin/activate
```

Update pip:

```console
python3.6 -m pip install -U pip
```

With this new version of pip, it is possible that you have to clear the pip cache:

```console
python3 -m pip cache purge
```

Now you can do the "Installation from PyPi" or the "Installation (alternative) by Cloning from GitHub".

Note that if you want install dependencies without internet connection, you can build a requirement.txt file:

```console
python3.6 -m pip freeze > requirements.txt 
python3.6 -m pip download -r requirements.txt -d requirements-download/
pip install -r requirements.txt --find-links=requirements-download --no-index
```

For deactivating the environment:

```console
deactivate
```