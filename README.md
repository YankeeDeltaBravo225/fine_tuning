# Kyoko/Noriko classifier by VGG16 + Finetuning

A fork project from https://github.com/kazuki-hayakawa/fine_tuning 

This program classifies given charasters with a small number of samples and fare accuracy by VGG16 + fine-tuinig

# Requirements

* Python  
  * 3.6 or later
* TensorFlow  
  * 1.4.0
* Pillow  
  * 4.2.0 or later
* Keras  
  * 2.1.0 or later

# Ｆｉｌｅｓ

* train.py　　
  * Generate classifier
* test.py　　
  * Predict by the generated classifier
* config　　
  * Configuration of classification targets

# How to use

Place your images below

* training 　images
  * data/training/kyoko 
  * data/training/noriko 

* validation images
  * data/validation/kyoko 
  * data/validation/noriko 

Then run train.py, it would take around 10minutes (Geforce970).  
`python train.py`

If the training is done, predict by test.py. 

`python test.py [image directory to predict]`

Images in the specified directry will be shown with the predict result.

# Changing classification target

* Create image directory and place your images there 
  * /data/training/
  * /data/validation/

* Change the following line on conf.py
  * ```python:conf.py
    classes = ['kyoko', 'noriko']
    ```

# Improving accuracy

Following would take effect.

* Adjust data extention parameter of ImageDataGenerator　　
  * Refer to https://keras.io/ja/preprocessing/image/ for detail
* Make epoch size in conf.py a greater value
  * Training time will be longer as well
  * ```python:conf.py
    nb_epoch = 30
    ```
