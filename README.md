
# PyXAI - Python eXplainable AI

- Documentation: [http://www.cril.univ-artois.fr/pyxai/](http://www.cril.univ-artois.fr/pyxai/)
- Git: [https://github.com/crillab/pyxai](https://github.com/crillab/pyxai)
- Installation: [http://www.cril.univ-artois.fr/pyxai/documentation/installation/](http://www.cril.univ-artois.fr/pyxai/documentation/installation/)

<figure>
  <img src="https://lh3.googleusercontent.com/drive-viewer/AITFw-xLC9-pvcsp0MGlTOBODqrs8aJogGpnAlAnVrh41EetySebz-VNzJW9PkHLmYUIBb_SaqlOpGBLsAm8IY5WIo73xNj0=s1600" alt="pyxai" />
  <figcaption>PyXAI's main steps for producing explanations.</figcaption>
</figure>
<figure>
  <img src="https://lh3.googleusercontent.com/drive-viewer/AITFw-yqn-ZOIW2u7a2XxVH9UNcr5SQQnxUH8b1wfLoReVa2f7zm68-S4GAbr7RWUYW1lKLJ957gLPaFn3077l4qZXUyv82T=s1600" alt="pyxai" />
  <img src="https://lh3.googleusercontent.com/drive-viewer/AITFw-xDdbVt_DCAmsvJhRlMj3jxgADUVkFzHbnxmQnabdjfuPaylcyeHTyBgDZs4Xna_N_oT6pwxXBv_ls2nqRUwd8RiWgM=s1600" alt="pyxai" />
  <figcaption>PyXAI's Graphical User Interface (GUI) for visualizing explanations.</figcaption>
</figure>

<h3>What is PyXAI ?</h3>
<p align="justify">
<b>PyXAI (Python eXplainable AI)</b> is a <a href="https://www.python.org/">Python</a> library (version 3.6 or later) allowing to bring explanations of various forms suited to <b>(regression or classification) tree-based ML models</b> (Decision Trees, Random Forests, Boosted Trees, ...). In contrast to many approaches to XAI (SHAP, Lime, ...), PyXAI algorithms are <b>model-specific</b>. Furthermore, PyXAI algorithms <b>guarantee certain properties</b> about the explanations generated, that can be of several types:
</p>
<ul>
  <li><b>Abductive explanations</b> for an instance $X$ are intended to explain why $X$ has been classified in the way it has been classified by the ML model (thus, addressing the “Why?” question). For the regression tasks, abductive explanations for $X$ are intended to explain why the regression value on $X$ is in a given interval.</li>
  <li><b>Contrastive explanations</b> for $X$ is to explain why $X$ has not been classified by the ML model as the user expected it (thus, addressing the “Why not?” question).</li>
</ul>

<h3>What is the difference between PyXAI and other methods ?</h3>
<p align="justify">

The most popular approaches (SHAP, Lime, ...) to XAI <b>are model-agnostic, but do not offer any guarantees</b> of rigor. A number of <a href="https://arxiv.org/pdf/2307.07514.pdf">works</a> have highlighted several misconceptions about informal approaches to XAI (see the <a href="{{ site.baseurl }}/papers/">related papers</a>). Contrastingly, <b>PyXAI algorithms rely on logic-based, model-precise</b> approaches for computing explanations. Although formal explainability has a number of drawbacks, particularly in terms of the computational complexity of logical reasoning needed to derive explanations, <b>steady progress has been made since its inception</b>. 
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
 <li>The <a href="{{ site.baseurl }}/documentation/installation/">installation guide</a> and the <a href="{{ site.baseurl }}/documentation/quickstart/">quick start</a></li>
 
  <li>About obtaining models:</li>
  <ul>
  <li>How to <b>prepare and clean a dataset</b> using the PyXAI <a href="{{ site.baseurl }}/documentation/preprocessor/">preprocessor</a> object?</li>
  <li>How to <b>import a model</b>, whatever its format? <a href="{{ site.baseurl }}/documentation/importing/"> Importing Models</a> </li>
  <li>How to <b>generate a model using a ML cross-validation method</b>? <a href="{{ site.baseurl }}/documentation/learning/generating/">Generating Models</a> </li>
  
  <li>How to <b>build a model from trees directly built by the user</b>? <a href="{{ site.baseurl }}/documentation/learning/builder/">Building Models</a></li>
  <li>How to <b>save and load models</b> with the PyXAI learning module? <a href="{{ site.baseurl }}/documentation/saving/">Saving/Loading Models</a></li>
  </ul>

<li>About obtaining explanations:</li>
  <ul>
  <li>The <b>concepts of the PyXAI explainer module</b>: <a href="{{ site.baseurl }}/documentation/explainer/concepts/">Concepts</a> </li>
  <li>How to use a <b>time limit</b>? <a href="{{ site.baseurl }}/documentation/explainer/time_limit/">Time Limit</a> </li>
  
  <li>The PyXAI library offers the possibility to process user preferences (<b>prefer some explanations to others or exclude some features</b>): <a href="{{ site.baseurl }}/documentation/explainer/preferences/">Preferences</a> </li>

  <li><b>Theories are knowledge about the dataset.</b> PyXAI offers the possibility of encoding a theory when calculating explanations in order to avoid calculating impossible explanations: <a href="{{ site.baseurl }}/documentation/explainer/theories/">Theories</a> </li>

  <li>How to <b>compute explanations for classification tasks</b>? <a href="{{ site.baseurl }}/documentation/classification/">Explaining Classification</a> </li>
  
  <li>How to <b>compute explanations for regression tasks</b>? <a href="{{ site.baseurl }}/documentation/regression/">Explaining Regression</a> </li>
  
  </ul>

 <li>How to use the <b>PyXAI's Graphical User Interface (GUI)</b> for <a href="{{ site.baseurl }}/documentation/visualization/">visualizing explanations</a>?</li>
 
 
</ul>

<h3>How to use PyXAI ?</h3>
<p align="justify">
Here is an example (it comes from the <a href="{{ site.baseurl }}/documentation/quickstart">Quick Start page</a>):
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

explainer.show()
```

<img src="https://lh3.googleusercontent.com/drive-viewer/AITFw-xbHs56zfQ_EHQ0-XqdHxy7mdL3fBxFRVnfW6pPCCCpSg89GStqQCBD5ElFLn3NaZmB-2mwY9hdu5TH0gPajOI2xwSCJQ=s1600" alt="pyxai" />

<p>As illustrated by this example, with a few lines of code, PyXAI allows you to train a model, extract instances, and get explanations about the classifications made.</p>

<br /><br />
<div class="center logo">
    <a href="http://www.cril.univ-artois.fr"><img src="https://lh3.googleusercontent.com/drive-viewer/AITFw-wsxZnVjsY1ypy7nGs2m__Iz5pDphw1wbc3a78HHVVqBhAFOx35hcvCGFaTfgDFlqGB_ChMWBfC-tlXUfX0twpqAnNfVg=s2560" /></a>
    <a href="https://www.irt-systemx.fr/"><img style="width: 80px;" src="https://lh3.googleusercontent.com/drive-viewer/AITFw-xuRWtP8WNuRXXaff32Tzd7OT4guc8vNEeXurAKIQiaeuIdeEYXo9hiA1HeGCgUY7I7NeT70U5yQt5BbwK6H4lv5jabQA=s2560" /></a>
    <a href="https://www.cnrs.fr/"><img style="width: 80px;" src="https://lh3.googleusercontent.com/drive-viewer/AITFw-xBV_ILK1g_mKMJ0Hk0wJtFmdKLnAT68QA7fMa5i663Tbla_Q2RjALnH6cER8BGAPThh_ZaOKpcO9ggkI1DAmU4zaEG=s1600" /></a>
    <a href="https://www.confiance.ai/"><img style="width: 80px;" src="https://lh3.googleusercontent.com/drive-viewer/AITFw-wiEyfiP29DKvwP5webvNRDXwXsS1PxQnTIZEdMpQ9xV9JN23-86HOqzNEBi9F4Ng8h-Kd8W5NKaWqefnGhhhQmxneu=s1600" /></a>
    <a href="http://univ-artois.fr"><img src="https://lh3.googleusercontent.com/drive-viewer/AITFw-wA-x2qgHNNrxLEaI33jDH64TM7sudMsTt781ICTAvzBsPaEtL2Ky_1Ba-QWm6YyqCmTuFGpylJ2sSXRgjzu7BM7iC8Xg=s2560" /></a>
</div>

