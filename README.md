# COMP 4447 Final Project: Object Recognition with Traditional Models

## File Structure

`svm_rf.py`: code to load the data, augment data, train model, test model, and tune hyperparameters.

`visual.py`: code to visualize the result.

`Figures`: contains all visualization plots.

`results.txt`: a log file to save all training and testing information.

`cifar-10-data`: the data folder for model training and testing. The image dataset is too large. Please read Data part to download it.

`requirements.txt`: all required packages to run this code.

`COMP 4447 Final Project.docx`: report for final project.

`COMP 4447 Midterm Presentation.pptx`: slides for final presentation.

`rf_weight.pkl` and `svm_weight.pkl`: pre-trained weight after training. They are over 2GB, so it is impossible to upload them. But, running `svm_rf.py` can help generate two pickle files.

## Dataset

This project studies CIFAR-10. We download it from [here](https://www.cs.toronto.edu/~kriz/cifar.html).

Click `CIFAR-10 python version`.

Name it `cifar-10-data`.

## Setup and initialize virtual environment

```bash
sudo apt update
sudo apt install python3-venv
# Enter yes or y if asked
python3 -m venv C4447
# Enter yes or y if asked
. C4447/bin/activate
```

## Install required packages

```bash
python3 -m pip install -r requirements.txt
```

## Run File

To train or test model

```bash
python3 svm_rf.py
```

To visualize the model

```bash
python3 visual.py
```
