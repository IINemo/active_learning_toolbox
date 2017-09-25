Active learning toolbox
=======================

This repo contains some query strategies and utils for active learning, as well
as the widget for dataset annotation in Jupyter IDE. The repo has tight
integration with [libact](https://github.com/ntucllab/libact) Python library.

 

Example of active learning annotation of MNIST dataset with the Jupyter widget.

![](https://github.com/IINemo/jupyter_al_annotator/blob/master/docs/al.png?raw=true)

Active learning
===============

Active learning (AL) is an interactive approach to simultaneously building a
labeled dataset and training a machine learning model. AL algorithm:

1.  A relatively large unlabeled dataset is gathered.

2.  A domain expert labels a few positive examples in the dataset.

3.  A classifier is trained on labeled samples.

4.  The classifier is applied to the rest of the corpus.

5.  Few most “useful” examples are selected (e.g., that increase classification
    performance).

6.  The examples labeled by the expert are added to the training set.

7.  Goto 3.

The procedure repeats until the performance of the classifier stops improving or
the expert is bored.

Requirements
============

1.  Python 3.6 (the package has not been tested with earlier versions)

2.  numpy (1.12.1)

3.  pandas (0.20.1)

4.  sklearn (0.18.1)

5.  scipy (0.19.0)

6.  Pillow (4.2.1)

7.  Jupyter (4.3.0)

8.  LibAct from the [fork](https://github.com/windj007/libact) (`pip install
    git+https://github.com/windj007/libact`)

Installation
============

Enabling widgets in Jupyter IDE
-------------------------------

The Jupyter widgets are not enabled by default. To install and activate them do
the following.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pip install ipywidgets
jupyter nbextension enable --py --sys-prefix widgetsnbextension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For further details, please, refer to [jupyter-widgets
repo](https://github.com/jupyter-widgets/ipywidgets).

Installing the library and the widget
-------------------------------------

To install the library and the widget execute in command line with root
priviledges:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pip install git+https://github.com/IINemo/active_learning_toolbox
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Usage
=====

See example for [MNIST dataset
annotation](https://github.com/IINemo/active_learning_toolbox/blob/master/examples/MNIST_annotation.ipynb)
and example for [20 newsgroups
annotation](https://github.com/IINemo/active_learning_toolbox/blob/master/examples/20newsgroups.ipynb).

Cite
====

If you use active learning toolbox in academic works, please cite (to be
published):

 

*BibTex:*

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@inproceedings{suvorovshelmanov2017ainl,
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    title={Active Learning with Adaptive Density Weighted Sampling for Information Extraction from Scientific Papers},
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    author={Roman Suvorov and Artem Shelmanov and Ivan Smirnov},
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    booktitle={Proceedings of AINL: Artificial Intelligence and Natural Language Conference},
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    publisher = {Springer, Communications in Computer and Information Science},
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    year={2017}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 

*Russian GOST:*

Suvorov R., Shelmanov A., Smirnov I. Active learning with adaptive density
weighted sampling for information extraction from scientific papers //
Proceedings of AINL: Artificial Intelligence and Natural Language Conference. —
Springer, Communications in Computer and Information Science, 2017.
