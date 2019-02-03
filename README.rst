text2array
==========

*Convert your NLP text data to arrays!*

.. image:: https://badge.fury.io/py/text2array.svg
   :target: https://badge.fury.io/py/text2array

.. image:: https://travis-ci.org/kmkurn/text2array.svg?branch=master
   :target: https://travis-ci.org/kmkurn/text2array

.. image:: https://readthedocs.org/projects/text2array/badge/?version=latest
   :target: https://text2array.readthedocs.io/en/latest/?badge=latest

.. image:: https://coveralls.io/repos/github/kmkurn/text2array/badge.svg?branch=master
   :target: https://coveralls.io/github/kmkurn/text2array?branch=master

.. image:: https://cdn.rawgit.com/syl20bnr/spacemacs/442d025779da2f62fc86c2082703697714db6514/assets/spacemacs-badge.svg
   :target: http://spacemacs.org

**text2array** helps you process your NLP text dataset into Numpy ndarray objects that are
ready to use for e.g. neural network inputs. **text2array** handles data shuffling,
batching, padding, and converting into arrays. Say goodbye to these tedious works!

Installation
============

**text2array** requires at least Python 3.6 and can be installed via pip::

    $ pip install text2array

Documentation
=============

https://text2array.readthedocs.io/en/latest/

Contributing
============

Pull requests are welcome! To start contributing, make sure to install all the dependencies.

::

    $ pip install -r requirements.txt

Then install this library as editable package.

::

    $ pip install -e .

Next, setup the pre-commit hook.

::

    $ ln -s ../../pre-commit.sh .git/hooks/pre-commit

Tests and the linter can be run with ``pytest`` and ``flake8`` respectively. The latter also
runs ``mypy`` for type checking.

License
=======

MIT
