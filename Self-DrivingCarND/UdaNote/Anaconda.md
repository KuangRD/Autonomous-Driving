# Anaconda
## Managing Environment

```sh
conda create -n some_envir python=3
conda env remove -n env_name
```

```sh
source activate some_envir
source deactivate some_envir
```
Export the package list of current environment
```sh
conda env export > environment.yaml.
```
Load from some existed environment.
```sh
conda env create -f environment.yaml
```
Listing environments
```sh
conda env list
```
When sharing your code on GitHub, it's good practice to make an environment file and include it in the repository. This will make it easier for people to install all the dependencies for your code. I also usually include a pip `requirements.txt` file using `pip freeze` [learn more here](https://pip.pypa.io/en/stable/reference/pip_freeze/) for people not using conda.

## Managing package
```sh
conda install numpy
conda install numpy=1.10
conda install numpy pandas matplotlib
conda install jupyter notebook
```
show the list of all the installed package name and version
```sh
conda list
```
```sh
conda upgrade --all
conda update --all
conda remove numpy
conda search nump
```
Conda also automatically installs dependencies for you. For example scipy depends on numpy, it uses and requires numpy. If you install just scipy (conda install scipy), Conda will also install numpy if it isn't already installed.



## Conda & Pip
You’re probably already familiar with pip, it’s the default package manager for Python libraries. Conda is similar to pip except that the available packages are focused around data science while pip is for general use. However, conda is not Python specific like pip is, it can also install non-Python packages. It is a package manager for any software stack. That being said, not all Python libraries are available from the Anaconda distribution and conda. You can (and will) still use pip alongside conda to install packages.

## Others


If you are seeing the following "conda command not found" and are using ZShell, you have to do the following:

Add `export PATH="/Users/username/anaconda/bin:$PATH"` to your .zsh_config file.
