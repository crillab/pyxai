
<p align="center">
  <div style="text-align:center">
    <img src="https://drive.google.com/uc?export=download&id=1R8PuTqnwQwZracP39JYmz91KnINiwt6e" />
  </div>
</p>

# PyXAI - Python eXplainable AI

- Documentation: [http://www.cril.univ-artois.fr/pyxai/](http://www.cril.univ-artois.fr/pyxai/)
- Git: [https://github.com/crillab/pyxai](https://github.com/crillab/pyxai)
- Installation: [http://www.cril.univ-artois.fr/pyxai/documentation/installation/](http://www.cril.univ-artois.fr/pyxai/documentation/installation/)

<h3>What is PyXAI ?</h3>
<p align="justify">
<b>PyXAI (Python eXplainable AI)</b> is a <a href="https://www.python.org/">Python</a> library (version 3.6 or later) allowing to bring explanations of various forms suited to <b>(regression or classification) tree-based ML models</b> (Decision Trees, Random Forests, Boosted Trees, ...). In contrast to many approaches to XAI (SHAP, Lime, ...), PyXAI algorithms are <b>model-specific</b>. Furthermore, PyXAI algorithms <b>guarantee certain properties</b> about the explanations generated, that can be of several types:
</p>
<ul>
  <li><b>Abductive explanations</b> for an instance $X$ are intended to explain why $X$ has been classified in the way it has been classified by the ML model (thus, addressing the “Why?” question). For the regression tasks, abductive explanations for $X$ are intended to explain why the regression value on $X$ is in a given interval.</li>
  <li><b>Contrastive explanations</b> for $X$ is to explain why $X$ has not been classified by the ML model as the user expected it (thus, addressing the “Why not?” question).</li>
</ul>

> <b> New features in version 1.0.0:</b>
> <ul>
>   <li>Regression for Boosted Trees with XGBoost or LightGBM</li>
>   <li>Adding Theories (knowledge about the dataset)</li>
>   <li>Easier model import (automatic detection of model types)</li>
>   <li>PyXAI's Graphical User Interface (GUI): displaying, loading and saving explanations. </li>
>   <li>Supports multiple image formats for imaging datasets</li>
>   <li>Supports data pre-processing (tool for preparing and cleansing a dataset)</li>
>   <li>Unit Tests with the unittest module</li>
> </ul> 

<figure>
  <img src="http://www.cril.univ-artois.fr/pyxai/assets/figures/pyxai2.png" alt="pyxai" />
  <figcaption>PyXAI's main steps for producing explanations.</figcaption>
</figure>

<figure>
  <img src="http://www.cril.univ-artois.fr/pyxai/assets/figures/GUI.png" alt="pyxai" />
  <img src="http://www.cril.univ-artois.fr/pyxai/assets/figures/GUI2.png" alt="pyxai" />
  <figcaption>PyXAI's Graphical User Interface (GUI) for visualizing explanations.</figcaption>
</figure>

<h3>What is the difference between PyXAI and other methods ?</h3>
<p align="justify">

The most popular approaches (SHAP, Lime, ...) to XAI <b>are model-agnostic, but do not offer any guarantees</b> of rigor. A number of <a href="https://arxiv.org/pdf/2307.07514.pdf">works</a> have highlighted several misconceptions about informal approaches to XAI (see the <a href="https://www.cril.univ-artois.fr/pyxai/papers/">related papers</a>). Contrastingly, <b>PyXAI algorithms rely on logic-based, model-precise</b> approaches for computing explanations. Although formal explainability has a number of drawbacks, particularly in terms of the computational complexity of logical reasoning needed to derive explanations, <b>steady progress has been made since its inception</b>. 
</p>


<h3>Which models can be explained with PyXAI ?</h3>
<p align="justify">
Models are the resulting objects of an experimental ML protocol through a chosen <b>cross-validation method</b> (for example, the result of a training phase on a classifier). Importantly, in PyXAI, there is a complete separation between the learning phase and the explaining phase: <b>you produce/load/save models, and you find explanations for some instances given such models</b>. Currently, with PyXAI, you can use methods to find explanations suited to different <b>ML models for classification or regression tasks</b>:
</p>
<ul>
  <li><a href="https://en.wikipedia.org/wiki/Decision_tree_learning">Decision Tree</a> (DT)</li> 
  <li><a href="https://en.wikipedia.org/wiki/Random_forest">Random Forest</a> (RF)</li>
  <li><a href="https://en.wikipedia.org/wiki/Gradient_boosting">Boosted Tree (Gradient boosting)</a> (BT)</li>
</ul> 
<p align="justify">
In addition to finding explanations, PyXAI also provides methods that perform operations (production, saving, loading) on models and instances. Currently, these methods are available for three <b>ML libraries</b>:
</p>
<ul>
  <li><a href="https://scikit-learn.org/stable/">Scikit-learn</a>: a software machine learning library</li> 
  <li><a href="https://xgboost.readthedocs.io/en/stable/">XGBoost</a>: an optimized distributed gradient boosting library</li>
  <li><a href="https://lightgbm.readthedocs.io/en/stable/">LightGBM</a>: a gradient boosting framework that uses tree based learning algorithms</li>
</ul> 
<p align="justify">
It is possible to also leverage PyXAI to find explanations suited to models learned using other libraries.
</p>

<h3>What does this website offer ?</h3>
<p align="justify">
In this website, you can find all what you need to know about PyXAI, with more than 10 <a href="https://jupyter.org/">Jupyter</a> Notebooks, including:
</p>
<ul>
 <li>The <a href="https://www.cril.univ-artois.fr/pyxai/documentation/installation/">installation guide</a> and the <a href="https://www.cril.univ-artois.fr/pyxai/documentation/quickstart/">quick start</a></li>
 
  <li>About obtaining models:</li>
  <ul>
  <li>How to <b>prepare and clean a dataset</b> using the PyXAI <a href="https://www.cril.univ-artois.fr/pyxai/documentation/preprocessor/">preprocessor</a> object?</li>
  <li>How to <b>import a model</b>, whatever its format? <a href="https://www.cril.univ-artois.fr/pyxai/documentation/importing/"> Importing Models</a> </li>
  <li>How to <b>generate a model using a ML cross-validation method</b>? <a href="https://www.cril.univ-artois.fr/pyxai/documentation/learning/generating/">Generating Models</a> </li>
  
  <li>How to <b>build a model from trees directly built by the user</b>? <a href="https://www.cril.univ-artois.fr/pyxai/documentation/learning/builder/">Building Models</a></li>
  <li>How to <b>save and load models</b> with the PyXAI learning module? <a href="https://www.cril.univ-artois.fr/pyxai/documentation/saving/">Saving/Loading Models</a></li>
  </ul>

<li>About obtaining explanations:</li>
  <ul>
  <li>The <b>concepts of the PyXAI explainer module</b>: <a href="https://www.cril.univ-artois.fr/pyxai/documentation/explainer/concepts/">Concepts</a> </li>
  <li>How to use a <b>time limit</b>? <a href="https://www.cril.univ-artois.fr/pyxai/documentation/explainer/time_limit/">Time Limit</a> </li>
  
  <li>The PyXAI library offers the possibility to process user preferences (<b>prefer some explanations to others or exclude some features</b>): <a href="https://www.cril.univ-artois.fr/pyxai/documentation/explainer/preferences/">Preferences</a> </li>

  <li><b>Theories are knowledge about the dataset.</b> PyXAI offers the possibility of encoding a theory when calculating explanations in order to avoid calculating impossible explanations: <a href="https://www.cril.univ-artois.fr/pyxai/documentation/explainer/theories/">Theories</a> </li>

  <li>How to <b>compute explanations for classification tasks</b>? <a href="https://www.cril.univ-artois.fr/pyxai/documentation/classification/">Explaining Classification</a> </li>
  
  <li>How to <b>compute explanations for regression tasks</b>? <a href="https://www.cril.univ-artois.fr/pyxai/documentation/regression/">Explaining Regression</a> </li>
  
  </ul>

 <li>How to use the <b>PyXAI's Graphical User Interface (GUI)</b> for <a href="https://www.cril.univ-artois.fr/pyxai/documentation/visualization/">visualizing explanations</a>?</li>
 
 
</ul>

<h3>How to use PyXAI ?</h3>
<p align="justify">
Here is an example (it comes from the <a href="https://www.cril.univ-artois.fr/pyxai/documentation/quickstart">Quick Start page</a>):
</p>
<h4 class="example">PyXAI in action</h4>

```python
from pyxai import Learning, Explainer

learner = Learning.Scikitlearn("tests/iris.csv", learner_type=Learning.CLASSIFICATION)
model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.DT)
instance, prediction = learner.get_instances(model, n=1, correct=True, predictions=[0])

explainer = Explainer.initialize(model, instance)
print("instance:", instance)
print("binary representation:", explainer.binary_representation)

sufficient_reason = explainer.sufficient_reason(n=1)
print("sufficient_reason:", sufficient_reason)
print("to_features:", explainer.to_features(sufficient_reason))

instance, prediction = learner.get_instances(model, n=1, correct=False)
explainer.set_instance(instance)
contrastive_reason = explainer.contrastive_reason()
print("contrastive reason", contrastive_reason)
print("to_features:", explainer.to_features(contrastive_reason, contrastive=True))

explainer.visualisation.gui()
```

<img src="https://www.cril.univ-artois.fr/pyxai/assets/figures/pyxaiGUI.png" alt="pyxai" />

<p>As illustrated by this example, with a few lines of code, PyXAI allows you to train a model, extract instances, and get explanations about the classifications made.</p>

<br /><br />
<p align="center">
    <a href="http://www.cril.univ-artois.fr"><img width="140px" src="https://www.cril.univ-artois.fr/pyxai/assets/figures/cril.png" /></a>
    <a href="https://www.cnrs.fr/"><img width="80px" style="width: 80px;" src="https://www.cril.univ-artois.fr/pyxai/assets/figures/cnrs.png" /></a>
    <a href="https://www.confiance.ai/"><img width="140px" style="width: 120px;" src="https://www.cril.univ-artois.fr/pyxai/assets/figures/confianceai.jpg" /></a>
    <a href="https://www.hautsdefrance.fr/"><img width="120px" style="width: 80px;" src="https://www.cril.univ-artois.fr/pyxai/assets/figures/logo_HDF.svg" /></a>
    <a href="http://univ-artois.fr"><img width="120px" src="https://www.cril.univ-artois.fr/pyxai/assets/figures/artois.png" /></a>
</p>

