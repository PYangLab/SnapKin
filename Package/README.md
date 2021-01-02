# SnapKin Package

This repository contains code for the SnapKin package which may be used to predict kinase substrate sites from phosphoproteomic data.

**Note.** The package requires dependencies mentioned in the root directory of this repository.

## Usage 

To use SnapKin, download this repository and go to the downloaded folder on the commandline (eg. Terminal).

Create an *arguments* text file [(mentioned below)](#Arguments-File) and run using python3 on the commandline.

Example usage with provided example data.
Note, point the terminal to the **Python** folder before running the example code.

    ~/SnapKin/Package/Python $ python3 Trainer.py ../Example_Python/arguments.txt

**Note.** It is recommended to move the data into the downloaded folder when running SnapKin for ease of use such as the `Example_Data` file.

## Data Format 

The training and test files must be a particular format as described below.

An example of the format and preprocessing can be found in [Example_Data/Example_Data_Preprocessing.Rmd](./Example_Data/Example_Data_Preprocessing.Rmd).

### Training 

The training dataset must be of the following format:

- site: how each observation will be identified. 
- phosphoproteomic data 
- sequence score 
- y: class labels 

where the phosphoproteomic data and sequence score are normalised using min-max (0-1) normalisation, and the columns *y* and *site* must be included. 
### Test

The test dataset must be of the following format:

- site: how each observation will be identified.
- phosphoproteomic data 
- sequence score

## Arguments File

The arguments file is used to provide necessary information, such as hyperparamters and file paths, to run SnapKin. 

An example `arguments.txt` file is provided in [Example_Data](./Example_Data/arguments.txt). 

The arguments that may be used in the arguments file are the following:

    train_fp (required)          :: file path to training csv file
    test_fp  (required)          :: file path to test csv file (requires site column for labels)
    save_dir (required)          :: folder path where predictions are saved 
    save_test_name               :: filename for predictions  (default is prediction.csv)
    verbose                      :: training output and logging - (0) none, (1) progress bar, (2) full output 
    num_layers                   :: number of layers in feed forward network 
    num_snapshots                :: number of snapshots to train 
    num_epochs                   :: number of epochs for each snapshot 
    batch_size                   :: the size of each batch during training 
    learning_rate                :: the maximum learning rate - high learning rates may mean the network won't converge 

**Note.** The file path depends on the current location on the commandline when running SnapKin.

### Example arguments file 

Below is an example of a simple arguments file saved as a txt file. 

    train_fp=Example_Data/Example_ESC_MTOR_train.csv
    test_fp=Example_Data/Example_ESC_MTOR_test.csv
    save_dir=Example_Data

A more complicated arguments file can be found in `Example_Data`.