.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/py_semtools.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/py_semtools
    .. image:: https://readthedocs.org/projects/py_semtools/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://py_semtools.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/py_semtools/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/py_semtools
    .. image:: https://img.shields.io/pypi/v/py_semtools.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/py_semtools/
    .. image:: https://img.shields.io/conda/vn/conda-forge/py_semtools.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/py_semtools
    .. image:: https://pepy.tech/badge/py_semtools/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/py_semtools
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/py_semtools

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

===========
py_semtools
===========


    Library to handle ontologies that allows queries and calculations (information coefficients, semantic similarity, ontology representations, etc) in a easy way. It can load any ontology that complies the obo format supported by OBO Foundry.


This library has the following capabilities:
* Ontology queries: It can perform term id or name translations, search the last stable id of a given term, infer term parents, etc.
* Associate items to terms: You can load term profiles associated to items such as genes, patients, etc and obtain the more specific terms, get information coefficient based on the item frequency, calculate semantic similarity against other items, etc.
* Ontology representations: There are methods to represent the specificity and frequency of the used terms in a set of items to visualize the ontology distribution in a given dataset.
* Text similarity analysis with SentenceTransformers (SBERT)

