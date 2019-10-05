# A RetinaNet-based algorithm for mammography
This algorthm was RetinaNet based on ResNet-152 to detect breast cancer in mammograms.

All you need is to put your mammograms into the directory of "testimages" and run "inference.py".
Inferenced images will be stored into "inference" directory.

This algorithm is not for a diagnostic use.

## Install

- Ensure git and git-lfs is installed.
- Ensure numpy and tensorflow-gpu is installed.
- Clone this repository.
- `python setup.py build_ext --inplace` to compile Cython code.

## Usage
`python inference.py`

## Enviroment
This algorithm was built in the TensorFlow framework (https://www.tensorflow.org/) with the Keras wrapper library (https://keras.io/).

- tensorflow-gpu 1.10.1
- Keras 2.2.4

## Research paper
This algorithm was published on XXX.
