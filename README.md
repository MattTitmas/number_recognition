# number_recognition
### Number recognition using a neural network

## Installation:
Python dependencies:
```
pip install -r requirements.txt
```

## Usage
```
python -m main -h
```
Output:
```
usage: main.py [-h] -t TRAIN -o OUTPUT [-e EPOCHS] [-hn HIDDEN_NODES [HIDDEN_NODES ...]]

Train a number recognition model.

options:
  -h, --help            show this help message and exit
  -t TRAIN, --train TRAIN
                        Location of training file.
  -o OUTPUT, --output OUTPUT
                        Directory of output weights.
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs to train with.
  -hn HIDDEN_NODES [HIDDEN_NODES ...], --hidden_nodes HIDDEN_NODES [HIDDEN_NODES ...]
                        Number of hidden nodes per layer.

```