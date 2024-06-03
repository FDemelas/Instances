# Instances.jl
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://fdemelas.github.io/Instances/dev/)
[![Documentation](https://github.com/FDemelas/Instances/actions/workflows/documentation.yml/badge.svg?branch=master)](https://github.com/FDemelas/Instances/actions/workflows/documentation.yml)


This package provides a complement of 

https://github.com/FDemelas/Learning_Lagrangian_Multipliers.jl/tree/main

Here we develop the encoding of the instances and the resolution of the Lagrangian Sub-Problem and the Continuous Relaxation.
For the moment this package supports these problems:
- Multi-Commodity Network Design,
- Generalized Assignment,
- Capacitated Warehouse Location Problem,
- Unit Commitment Problem.

## Documentation

More specific information can be found in the corresponding documentation.
To build the documentation you can use the following command from the main folder:

```shell
julia --project=docs/ docs/make.jl
```

and you will find it in the folder docs/build/ as html files.

## License

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
