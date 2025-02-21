# SnapKin

SnapKin: Protein phosphorylation site prediction for phosphoproteomic data using an ensemble deep learning approach.


## Requirements

The following are the dependencies required to run the model 

```
    python = 3.8
    tensorflow = 2.2.0
    numpy >= 1.19.4
    pandas >= 1.1.5
```
### Useful Packages 

SnapKin can be used in R and an example workflows can be found in the articles.
The following R packages are used in the example R workflow:

```
    PhosR        : (sequence information scoring and kinase-substrate labelling)
    r-reticulate : (integrates Python into R)
    dplyr        : (dataframe manipulation)
```

**Note.** 
Please use the development version of `PhosR` on Github by installing using *devtools*

```
    install.packages('devtools')
    devtools::install_github("PYangLab/PhosR")
```


### Conda 

We recommend installing the necessary dependencies via Conda (refer to [Install Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)).

#### Installation: Commandline

The following code snippet is for initialising and activating a Conda environment on the commandline for Tensorflow with CPU:

        conda env create -f environment.yml
        conda activate SnapKin

This installs the necessary dependencies in a new environment and activates it.

For GPU support, use *environment-gpu.yml* and *activate SnapKin-GPU*. 
**Note.** Our method for GPU support is not tested for MacOS, but CPU support is available for MacOS.

#### Installation: R 

A helper function is included to install the appropriate conda environment in R by running the following code.

``` 
    install.packages('r-reticulate')
    SnapKin::installSnapkin(useGPU=FALSE)
```

For non-MacOS users, Tensorflow-GPU may be installed by using *useGPU=TRUE*.

## Example

Please follow the Python or R work for testing on the example data:

* [SnapKin Workflow in Python](https://pyanglab.github.io/SnapKin/articles/SnapKin_WorkFlow_Python.html)

* [SnapKin Workflow in R](https://pyanglab.github.io/SnapKin/articles/SnapKin_Workflow_R.html)


