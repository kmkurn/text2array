text2array
==========

*Convert your NLP text data to arrays!*

.. image:: https://img.shields.io/pypi/pyversions/text2array.svg?style=flat
   :target: https://img.shields.io/pypi/pyversions/text2array.svg?style=flat
   :alt: Python versions

.. image:: https://img.shields.io/pypi/v/text2array.svg?style=flat
   :target: https://pypi.org/project/text2array
   :alt: PyPI project

.. image:: https://img.shields.io/travis/kmkurn/text2array.svg?style=flat
   :target: https://travis-ci.org/kmkurn/text2array
   :alt: Build status

.. image:: https://img.shields.io/readthedocs/text2array.svg?style=flat
   :target: https://text2array.readthedocs.io
   :alt: Documentation status

.. image:: https://img.shields.io/coveralls/github/kmkurn/text2array.svg?style=flat
   :target: https://coveralls.io/github/kmkurn/text2array
   :alt: Code coverage

.. image:: https://img.shields.io/pypi/l/text2array.svg?style=flat
   :target: https://www.apache.org/licenses/LICENSE-2.0
   :alt: License

.. image:: https://cdn.rawgit.com/syl20bnr/spacemacs/442d025779da2f62fc86c2082703697714db6514/assets/spacemacs-badge.svg
   :target: http://spacemacs.org
   :alt: Built with Spacemacs

**text2array** helps you process your NLP text dataset into Numpy ndarray objects that are
ready to use for e.g. neural network inputs. **text2array** handles data shuffling,
batching, padding, and converting into arrays. Say goodbye to these tedious works!

Documentation
=============

https://text2array.readthedocs.io

Contributing
============

Pull requests are welcome! To start contributing, first install flit_.

::

    pip install flit

Next, install this library and its dependencies in development mode.

::

    flit install --symlink

Lastly, setup the pre-commit hook.

::

    ln -s ../../pre-commit.sh .git/hooks/pre-commit

Tests, the linter, and the type checker can be run with ``pytest``, ``flake8``, and ``mypy``
respectively.

License
=======

Apache License, Version 2.0


.. _flit: https://pypi.org/project/flit/
