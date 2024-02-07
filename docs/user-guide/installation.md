# Installation

ScippNeutron is available on Pip and Conda.
If you need Mantid, you need to use Conda because Mantid is not available on Pip.

`````{tab-set}
````{tab-item} pip
```sh
pip install scippneutron
```

This will install both ScippNeutron and its dependencies which include Scipp and ScippNexus.

By default, this will only install minimal requirements.
If you wish to use plotting features, you can install all the optional dependencies by doing

```sh
pip install scippneutron[all]
```

See the [scipp documentation](https://scipp.github.io/getting-started/installation.html)
for more about extra dependencies.
````
````{tab-item} conda
```sh
conda install -c conda-forge -c scipp scippneutron
```

This will install ScippNeutron and all its dependencies.
However, if you need Mantid, you need to install that separately, e.g., using

```sh
conda install -c conda-forge -c scipp -c mantid mantid scippneutron
```
````
`````
