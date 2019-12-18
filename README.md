# ML-Project2

Authors : Damien Ronssin, Jérémie Canoni--Meynet, Benjamin Pedrotti

## Installation

In order to be able to run the code, you will need the following libraries :
* Numpy `pip install numpy`
* pandas `pip install pandas`
* Tensorflow `pip install tensorflow`
* Keras `pip install keras`
* Symspell `pip install symspellpy`
* nltk `pip install nltk`
* gensim `pip install gensim`

Some additional downloads are required for nltk and Symspell. Note that due to the computational cost of the whole processing, the notebooks have been executed using google colaboratory.

## Usage

### Pre-processing

You will find in the file `cleaning.py` all the pre-processing methods explained in the report. This file generates three `.npy` files  with pre-processed data : the training dataset and its labels and the test dataset.
The following files are creating when running `cleaning.py`: 
* `data_train_pr_f_sl5.npy`
* `data_test_pr_f_sl5.npy`
* `labels_train_f_sl5.npy`

Concerning the generation of the raw dataset (please refer to the report), you will need to call the function '''python raw_data(path_g, full)''' contained in the file `helpers.py`. where `path`

### Hyper-parameter tuning

In the notebooks `CNN_Tuning.ipynb` and `LSTM_Tuning.ipynb` you will find the hyper-parameter tuning for two different deep learning architectures : convolutional network and LSTM network respectively.




### Reproducibilty 

You will be able to reproduce our best results by running the notebook `CNN.ipynb`. 

