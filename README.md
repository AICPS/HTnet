# HTnet: Transfer Learning for Golden Chip-Free Hardware Trojan Detection
This is the companion repository for [our paper](https://link_to_paper) titled "HTnet: Transfer Learning for Golden Chip-Free Hardware Trojan Detection" published in [The Design, Automation, and Test in Europe (DATE'2021)] conference. 


## Data 
The data used in this project comes from the following source: 
* The [Our archive](http://ieee-dataport.org/3599), which contains side-channel signals for multiple hardware Trojan Infected circuits. 

## Code 
The code is divided as follows: 
* The [trainOnAllExecptTwoModality_different_classes.py](https://link_to_) python file to train base HTnet. 
* The [tl_learning_tests_TwoModality_double_net.py](https://link_to_) python file to use the base base HTnet for feature extraction. 
* The [utils](https://link_to_) folder contains the necessary functions to read the datasets and summarize the results.
* The [classifiers](https://link_to_) folder contains python files  deep neural network models. The [htnet.py] is used when finding the best model structure. The [htnetNarrowed.py](https://link_to_) is the final model published in the paper and used for training the base model.  The [transferLearningDoubleNetAnomalyDetectorGroup.py](https://link_to_) present the double net transfer learning approach provided in the paper. 

To run the code, the data should be placed in the HT_Data folder. Then, the base model should be trained and tested as follows: 
```
python3 trainOnAllExecptTwoModality_different_classes.py
python3 tl_learning_tests_TwoModality_double_net.py

```

## Prerequisites
All python packages needed are listed in [pip-requirements.txt](https://github.com/hfawaz/dl-4-tsc/blob/master/utils/pip-requirements.txt) file and can be installed simply using the pip command. 

* [numpy](http://www.numpy.org/)  
* [pandas](https://pandas.pydata.org/)  
* [sklearn](http://scikit-learn.org/stable/)  
* [scipy](https://www.scipy.org/)  
* [matplotlib](https://matplotlib.org/)  
* [tensorflow-gpu](https://www.tensorflow.org/)  
* [keras](https://keras.io/)  
* [h5py](http://docs.h5py.org/en/latest/build.html)
* [keras_contrib](https://www.github.com/keras-team/keras-contrib.git)

## Acknowledgment
This work was supported by
the Office of Naval Research (ONR) under Award N00014-17-1-2499.

## Reference

If you re-use this work, please cite:

```
@article{faezi2021htnet,
  Title                    = {HTnet: Transfer Learning for Golden Chip-Free Hardware Trojan Detection},
  Author                   = {Faezi, Sina and Yasaei, Rozhin and Al Faruque, Mohammad},
  journal                = {The Design, Automation, and Test in Europe},
  Year                     = {2021}
}
```
