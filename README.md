# Missingness Augmentation: A General Approach for Improving Generative Imputation Models

Before each training epoch, we use the outputs of the model to expand the incomplete samples, 
and then construct a special reconstruction loss for these augmented samples. 
The final optimization objective of the model is composed of the original loss plus the reconstruction loss for augmented samples. 
Due to the introduction of extra incomplete samples during the training process, 
the proposed approach is named Missingness Augmentation (MA).

In this code, we provide the original GAIN code and the corresponding modified code in our method, 
to run them separately and compare the difference in results. 
You can see that our proposed method improves the results.
We provide the original data used in the paper in the `datasets` file.

## Setup
This code was tested on Windows,
Python 3.6.4, tensorflow-gpu 2.0.0,
cuda_10.0.130_411.31_win10 and 
cudnn-10.0-windows10-x64-v7.4.2.24.


## Experiments

## Missing Feature Imputation

To impute missing features with GAIN one can cancel the `GAIN(miss_data_x, gain_parameters)` comment in `main.py` and use following commands:
```
python main.py
```
Or to impute missing features with GAIN+ cancel the `MA_GAIN(miss_data_x, gain_parameters)` comment in `main.py`.
