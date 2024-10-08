# Helicast

Welcome to the Helicast repo! 

Helicast is a library that aims at facilitating data science and machine learning 
workflows for the data science team at [Helicon](https://helicon.ai/). The library is 
in active development.


The documentation can be built using the makefile! Run ``make docs`` to compile the
documentation and open the documentation in your web browser. If you want to open
the documentation without re-compiling it, you can use ``make open_docs``.

> [!NOTE]  
> The repo has been moved to github recently. We will host the documentation with 
> readthedocs or github pages. Stay tuned =) 


## 1. Installation
You can install the library using pip
```bash
pip install git+https://github.com/heliconhq/helicast.git@XYZ
```
where ``XYZ`` can be a commit hash, a branch name or a tag. We recommend using tags
as they refer to immutable snapshots of the library.

You can also include the library in your dependencies, e.g., in your ``pyproject.toml``
file,
```toml
[project]
# ...
dependencies = [
    # ...
    "helicast @ git+https://github.com/heliconhq/helicast.git@XYZ",
    #...
]
```


## 2. Development
We're using [rye](https://rye.astral.sh/) for environment management. You can clone/fork
the repo and then run ``rye sync``.