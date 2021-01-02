# SnapKin

SnapKin: Protein phosphorylation site prediction for phosphoproteomic data using an ensemble deep learning approach.

## Description


## Requirements

The following are the dependencies required to run the model 

```
    tensorflow = 2.0.0
    numpy >= 1.19.4
    pandas >= 1.1.5
```

### Conda 

We recommend installing the necessary dependencies via Conda (refer to [Install Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)).
The following code snippet is for initialising and activating a Conda environment on the commandline for Tensorflow with CPU:

        conda env create -f environment.yml
        conda activate SnapKin

This installs the necessary dependencies in a new environment and activates it.
For GPU support, use *environment-gpu.yml* and *activate SnapKin-GPU*.

## Example 

Documentation of the SnapKin package and example usage can be found in the following [README](./Package/README.md).
