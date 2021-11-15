# Web application built with Python and Flask. 
## This web application was built for the thesis work "Cooperative edge deepfake detection" which can be found at http://urn.kb.se/resolve?urn=urn:nbn:se:hj:diva-53790.

### Purpose
The webapplication can be used to convert ml-weights (generated with the use of Darknet YOLOv2) to Keras models and create a bagging ensemble 
from the models.

#### Requirements
* Python 3.7.1
* Tensorflow 1.15
* Keras 2.3.1
* h5py 2.10.0
* Pillow

#### Dataset
A subset of the dataset from https://www.kaggle.com/xhlulu/140k-real-and-fake-faces is included in the repository.

#### Credits
- This project makes use of YAD2k which can be found at: https://github.com/allanzelener/YAD2K
- This project makes use of YOLOw-Keras which can be found at: https://github.com/miranthajayatilake/YOLOw-Keras
