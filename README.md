
<p align="center">
  <div style="text-align:center">
    <img src="http://www.cril.univ-artois.fr/pyxai/assets/figures/logo.png" alt="pyxai" width="200"/>
  </div>
</p>

# PyXAI - Python eXplainable AI

- Documentation: [http://www.cril.univ-artois.fr/pyxai/](http://www.cril.univ-artois.fr/pyxai/)
- Git: [https://github.com/crillab/pyxai](https://github.com/crillab/pyxai)
- Pypi: [hhttps://pypi.org/project/pyxai/](https://pypi.org/project/pyxai/)
- Installation: [http://www.cril.univ-artois.fr/pyxai/documentation/installation/](http://www.cril.univ-artois.fr/pyxai/documentation/installation/)

<h3>What is PyXAI ?</h3>
<p align="justify">
<b>PyXAI (Python eXplainable AI)</b> is a <a href="https://www.python.org/">Python</a> library (version 3.6 or later) allowing to bring formal explanations suited to <b>(regression or classification) tree-based ML models</b> (Decision Trees, Random Forests, Boosted Trees, ...). PyXAI generates explanations that are <b>post-hoc and local</b>. In contrast to many popular approaches to XAI (SHAP, LIME, ...), PyXAI generates explanations that are also <b>correct</b>. Being correct (aka sound or faithful) indicates that the explanations that are provided actually <b>reflect the exact behaviour of the model by guaranteeing certain properties</b> about the explanations generated. They can be of several types:  
</p>

<ul>
  <li><b>Abductive explanations</b> for an instance $X$ are intended to explain why $X$ has been classified in the way it has been classified by the ML model (thus, addressing the “Why?” question). In the regression case, abductive explanations for $X$ are intended to explain why the regression value of $X$ belongs to a given interval.</li>
  <li><b>Contrastive explanations</b> for $X$ are intended to explain why $X$ has not been classified by the ML model as the user expected it (thus, addressing the “Why not?” question).</li>
</ul>

<p align="justify">
PyXAI also includes algorithms for <b>correcting tree-based models</b> when their predictions conflict with pieces of user knowledge. This more tricky facet of XAI is seldom offered by existing XAI systems. When some domain knowledge is available and a prediction (or an explanation) contradicts it, the model must be corrected. <b>Rectification</b> is a principled approach for such a correction operation.
</p>

> <b> New features in version 1.1:</b>
> <ul>
>   <li>Rectification for DT (Decision Tree) and RF (Random Forest) models dedicated to binary classification.</li>
>   <li>Visualization displayed in a notebook or on screen, and now also for time series problems.</li> 
>   <li>Enhanced compatibility with Mac OS and Windows</li>
> </ul> 

> <b> New features in version 1.0:</b>
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
  <img src="http://www.cril.univ-artois.fr/pyxai/assets/figures/base.png" alt="pyxai" width="600"/>
  <figcaption>User interaction with PyXAI.</figcaption>
</figure>

<figure>
  <img src="http://www.cril.univ-artois.fr/pyxai/assets/figures/GUI.png" alt="pyxai" width="580"/>
  <figcaption>PyXAI's Graphical User Interface (GUI) for visualizing explanations.</figcaption>
</figure>

<figure>
  <img src="http://www.cril.univ-artois.fr/pyxai/assets/figures/timeserie.png" alt="pyxai" />
  <figcaption>Visualization in a notebook of an explanation for an instance from a time series problem.</figcaption>
</figure>

<h3>What is the difference between PyXAI and other methods ?</h3>
<p align="justify">

The most popular approaches (SHAP, LIME, ...) to XAI <b>are model-agnostic, but they do not offer any guarantees</b> of rigor. 
A number of works by <a href="https://arxiv.org/pdf/2307.07514.pdf">Marques-Silva and Huang</a>, <a href="https://www.ijcai.org/proceedings/2020/726">Ignatiev</a> have highlighted several misconceptions about such approaches to XAI. Correctness is paramount when dealing with high-risk or sensitive applications, which is the type of applications that are targeted by PyXAI. When the correctness property is not satisfied, one can find ”counterexamples” for the explanations that are generated, i.e., pairs of instances sharing an explanation but leading to distinct predictions. Contrastingly, <b>PyXAI algorithms rely on logic-based, model-precise</b> approaches for computing explanations. Although formal explainability has a number of drawbacks, particularly in terms of the computational complexity of logical reasoning needed to derive explanations, <b>steady progress has been made since its inception</b>. 
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
 
  <li>About models:</li>
  <ul>
  <li>How to <b>prepare and clean a dataset</b> using the PyXAI <a href="https://www.cril.univ-artois.fr/pyxai/documentation/preprocessor/">preprocessor</a> object?</li>
  <li>How to <b>import a model</b>, whatever its format? <a href="https://www.cril.univ-artois.fr/pyxai/documentation/importing/"> Importing Models</a> </li>
  <li>How to <b>generate a model using a ML cross-validation method</b>? <a href="https://www.cril.univ-artois.fr/pyxai/documentation/learning/generating/">Generating Models</a> </li>
  
  <li>How to <b>build a model from trees directly built by the user</b>? <a href="https://www.cril.univ-artois.fr/pyxai/documentation/learning/builder/">Building Models</a></li>
  <li>How to <b>save and load models</b> with the PyXAI learning module? <a href="https://www.cril.univ-artois.fr/pyxai/documentation/saving/">Saving/Loading Models</a></li>
  </ul>

<li>About explanations:</li>
  <ul>
  <li>The <b>concepts of the PyXAI explainer module</b>: <a href="https://www.cril.univ-artois.fr/pyxai/documentation/explainer/concepts/">Concepts</a> </li>
  <li>How to use a <b>time limit</b>? <a href="https://www.cril.univ-artois.fr/pyxai/documentation/explainer/time_limit/">Time Limit</a> </li>
  
  <li>The PyXAI library offers the possibility to process user preferences (<b>prefer some explanations to others or exclude some features</b>): <a href="https://www.cril.univ-artois.fr/pyxai/documentation/explainer/preferences/">Preferences</a> </li>

  <li><b>Theories are knowledge about the dataset.</b> PyXAI offers the possibility of encoding a theory when calculating explanations in order to avoid calculating impossible explanations: <a href="https://www.cril.univ-artois.fr/pyxai/documentation/explainer/theories/">Theories</a> </li>

  <li>How to <b>compute explanations for classification tasks</b>? <a href="https://www.cril.univ-artois.fr/pyxai/documentation/classification/">Explaining Classification</a> </li>
  
  <li>How to <b>compute explanations for regression tasks</b>? <a href="https://www.cril.univ-artois.fr/pyxai/documentation/regression/">Explaining Regression</a> </li>
  
  </ul>
  <li>About rectification:</li>
  <ul>
    <li>How to <b>rectify a DT model</b>? <a href="https://www.cril.univ-artois.fr/pyxai//documentation/rectification/decision_tree/">Rectification for Decision Tree</a> </li>
    <li>How to <b>rectify a RF model</b>? <a href="https://www.cril.univ-artois.fr/pyxai//documentation/rectification/random_forest/">Rectification for Random Forest</a> </li>
  </ul>
 <li>About visualization:</li>
  <ul>
    <li>How to generate images to represent explanations (<b>in a notebook or save them in png format</b>)? <a href="https://www.cril.univ-artois.fr/pyxai/documentation/visualization/visualization/">Visualization of Explanations (Without GUI)</a>?</li>
    <li>How to use the <b>PyXAI's Graphical User Interface (GUI)</b> for <a href="https://www.cril.univ-artois.fr/pyxai/documentation/visualization/visualizationGUI/">visualizing explanations</a>?</li>
  </ul>
 
 
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

explainer.visualisation.screen(instance, contrastive_reason, contrastive=True)
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

