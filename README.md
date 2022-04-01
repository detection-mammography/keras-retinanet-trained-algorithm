# A RetinaNet-based algorithm for mammography
This algorthm was RetinaNet based on ResNet-152 to detect breast cancer in mammograms.

All you need is to put your mammograms into the directory of "testimages" and run "inference.py".
Inferenced images will be stored into "inference" directory.

This algorithm is not for a diagnostic use.

## Install

- Ensure numpy and tensorflow-gpu is installed.
- Clone this repository.
- `python setup.py build_ext --inplace` to compile Cython code.
- Download a H5 file (resnet152_pascal.h5) into `snapshots` directory.

H5 file can be downloaded from:
https://github.com/detection-mammography/keras-retinanet-trained-algorithm/releases/tag/v0.1-mammography


## Usage
`python inference.py`

## Enviroment
This algorithm was built in the TensorFlow framework (https://www.tensorflow.org/) with the Keras wrapper library (https://keras.io/).

- tensorflow-gpu 1.10.1
- Keras 2.2.4

## Research paper
This algorithm was published on https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0265751.

## Special thanks
This research is supported by following contributors.

YEBIS.XYZ, tech vein, Osakan Space, Workflow-Design LLC  
Chinatsu Aida, Hitoshi Ando, Yukiko Ida, Taichi Kakinuma, Taroo Takahara, Yosie Kizaki, Eri Kusano, Ayako Yamakake
