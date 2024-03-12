# Helicast

For environment management, we recommend [miniforge](https://github.com/conda-forge/miniforge).


## 1 Installation

### 1.1 Install as a Python library (static)

You can install `helicast` as a python library in any of your project! The package is still
in active development, so we recommend that you pin the version when installing. At this
stage, the package is in a private repo, which means that you should have access to 
GitLab via SSH for the following to work.

To install the package, you can use
```bash
pip install git+ssh://git@gitlab.com/trelltech/data-science/helicast.git@XYZ
```
where `XYZ` is the tag, branch name or commit hash. If you omit `@XYZ`, it will install
the main branch (we do not recommend doing that)

### 1.2 Install as a Python library (editable)
(Not recommended!)


If you want to install the `helicast` python library for your project but still have
access to the latest change quickly, you can install it using [pip editable install](https://setuptools.pypa.io/en/latest/userguide/development_mode.html). The work flow will look like the following:

1. Clone the `helicast` repository somewhere in you computer, e.g.,
`git clone git@gitlab.com:trelltech/data-science/helicast.git ..`
2. Navigate to the helicast repo you just cloned, e.g., `cd ../helicast`
3. Activate your project environment, e.g., `conda activate myproject`
3. Install `helicast` in editable mode, `pip install -e .`

With this setup, the `helicast` library installed in your conda environment `myproject`
will reflect the state of the repo on your local filesystem! For instance, if you change
branch, using `helicast` in your project will reflect the state of `helicast` in that
specific branch! This set up will make your project less reproducible. 


### 1.3 Install the `helicast` development environment

If you want to develop `helicast`, you can do so quite easily :) 
1. Clone the repo on your local machine `git clone git@gitlab.com:trelltech/data-science/helicast.git`
2. Run the bash script `./conda_setup.sh`

The script will create an environment `helicast` in which python 3.10 and `helicast`
dependencies will be installed and `helicast` will be installed in editable mode. You can
change the name of the conda environment passing `--env-name ENV_NAME` to the script. If
an environment with the same name exist, the script will forcefully overwrite it!