<h1 align="center">TPSC</h1>
<p align="center">
A Python library that allows the computation of Hubbard model related functions and quantities (such as the self-energy and Green's function) using the Two-Particle-Self-Consistent (TPSC) approach first described in
[<a href="https://arxiv.org/abs/cond-mat/9702188">Vick and Tremblay, 1997</a>].
</p>

## Table of contents

- [Installation](#installation)
- [Documentation and tutorials](#documentation-and-tutorials)
- [Examples](#examples)
- [TODO](#TODO)


## Installation

This package will be soon available on PyPI. 
Meanwhile, you can install it by cloning this repository and using pip:

```bash
git clone https://github.com/amstremblay/TPSC
cd TPSC
pip install .
```

## Documentation and tutorials

TODO Create docs and tutorials, then add link here.


## Examples

The `examples/` directory contains some examples for you to experiment with and get familiar with different use cases.

The quickest way to start doing calculations using TPSC is to call the `TPSC` executable from the command line and provide input parameters in a JSON `para.json` file (see the `examples` directory):

```bash
TPSC para.json
```

TPSC could also be use in Python scripts for a finner control over input parameters or to post-process the results of TPSC calculations such as plotting observables.


## TODO

* Release on PyPi
* Automated tests with `pytest`
* Automated documentation on GitHub
* More examples
* JSON output
* Inclure TPSC+ ?
* Inclure binding avec TRIQS ?
